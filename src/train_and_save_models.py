"""
Model Training and Serialization Script for NETRA 1.0 - IMT-2030

Trains models on IMT-2030 compliant dataset and saves them for deployment.
Creates both stable (v1) and intentionally degraded (v2) models for testing
rollback functionality.

Citations:
- ITU-R M.2160: "Framework and overall objectives of the future development of IMT
  for 2030 and beyond" (September 2023)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# IMT-2030 feature columns
IMT2030_FEATURE_COLS = [
    "time_of_day",
    "usage_scenario",
    "throughput_mbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_rate",
    "reliability_target",
    "connection_density_km2",
    "mobility_kmph",
    "area_traffic_capacity_score",
    "ai_load_score",
    "resilience_score",
]

TARGET_COL = "future_load_target"


def prepare_data():
    """Load and prepare IMT-2030 dataset."""
    df = pd.read_csv("data/traffic_synthetic.csv")

    # Encode categorical columns
    if "usage_scenario" in df.columns:
        le = LabelEncoder()
        df["usage_scenario"] = le.fit_transform(df["usage_scenario"])

    X = df[IMT2030_FEATURE_COLS]
    y = df[TARGET_COL]

    return X, y


def train_stable_model(X, y):
    """Train stable model (good performance) with feature scaling."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SGDRegressor(
        max_iter=1000,
        learning_rate="adaptive",
        eta0=0.01,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )
    model.fit(X_scaled, y)

    # Store scaler with model for inference
    model.scaler = scaler

    return model


def train_bad_model(X, y):
    """
    Train intentionally degraded model for testing rollback.
    Uses shuffled labels to simulate poor performance.
    """
    # Shuffle labels to create "bad" model
    y_shuffled = np.random.permutation(y)

    model = SGDRegressor(
        max_iter=100,  # Fewer iterations
        learning_rate="constant",
        eta0=0.001,    # Lower learning rate
        random_state=42,
        warm_start=True,
    )
    model.partial_fit(X[:len(X)//2], y_shuffled[:len(X)//2])
    return model


def save_model(model, path, feature_names):
    """Save model in FL-compatible format."""
    model_data = {
        "coef": model.coef_,
        "intercept": model.intercept_,
        "feature_names": feature_names,
        "model_version": "IMT-2030-v1.0",
        "scaler_mean": model.scaler.mean_ if hasattr(model, 'scaler') and model.scaler else None,
        "scaler_scale": model.scaler.scale_ if hasattr(model, 'scaler') and model.scaler else None,
    }
    joblib.dump(model_data, path)
    print(f"Saved model to {path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("NETRA 1.0 - Model Training & Serialization (IMT-2030)")
    print("=" * 60)

    # Prepare data
    X, y = prepare_data()
    print(f"Loaded {len(X)} samples with {len(IMT2030_FEATURE_COLS)} features")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Train and save stable model (v1)
    print("\nTraining stable model (v1)...")
    stable_model = train_stable_model(X, y)
    save_model(stable_model, "models/model_v1.pkl", IMT2030_FEATURE_COLS)

    # Train and save bad model (v2) for testing
    print("\nTraining degraded model (v2) for rollback testing...")
    bad_model = train_bad_model(X, y)
    save_model(bad_model, "models/model_v2_bad.pkl", IMT2030_FEATURE_COLS)

    # Also save as FL global model format
    save_model(stable_model, "models/fl_global_model.pkl", IMT2030_FEATURE_COLS)

    print("\n[OK] Model training complete!")
    print("  - models/model_v1.pkl: Stable baseline model")
    print("  - models/model_v2_bad.pkl: Degraded model for testing")
    print("  - models/fl_global_model.pkl: FL global model")


if __name__ == "__main__":
    main()
