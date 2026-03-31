"""
Centralized Training Script for NETRA 2.0 - IMT-2030

Trains a baseline model using the IMT-2030 compliant dataset.
This script serves as a reference implementation for the centralized
training approach, complementing the federated learning pipeline.

Citations:
- ITU-R M.2160: "Framework and overall objectives of the future development of IMT
  for 2030 and beyond" (September 2023)
- ITU-R IMT-2030 TPR: "Technical Performance Requirements for IMT-2030" (February 2026)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, max_error
from sklearn.linear_model import SGDRegressor
from math import sqrt

# IMT-2030 feature columns (excluding target and metadata)
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


def train_and_eval(model_type="sgd"):
    """
    Train and evaluate model on IMT-2030 dataset.

    Args:
        model_type: Type of model to train ("sgd" for SGDRegressor, "rf" for RandomForest)

    Returns:
        Trained model
    """
    # Load IMT-2030 dataset
    df = pd.read_csv("data/traffic_synthetic.csv")

    # Encode categorical columns
    if "usage_scenario" in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["usage_scenario"] = le.fit_transform(df["usage_scenario"])

    # Extract features and target
    X = df[IMT2030_FEATURE_COLS]
    y = df[TARGET_COL]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model
    if model_type == "sgd":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SGDRegressor(
            max_iter=1000,
            learning_rate="adaptive",
            eta0=0.01,
            random_state=42,
            early_stopping=True,
        )
        # Train model with scaled features
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            n_jobs=-1,
            random_state=42,
        )
        # Train model (RF doesn't need scaling)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Evaluate
    preds = model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    max_err = max_error(y_test, preds)

    print(f"Model: {model_type}")
    print(f"  Samples: {len(df)} (train={len(X_train)}, test={len(X_test)})")
    print(f"  Features: {len(IMT2030_FEATURE_COLS)}")
    print(f"  RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.4f}, MAPE={mape:.1f}%, Max Error={max_err:.1f}")

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("NETRA 2.0 - Centralized Training (IMT-2030)")
    print("=" * 60)

    # Train SGD model (for FL compatibility)
    train_and_eval("sgd")

    # Train Random Forest (for comparison)
    train_and_eval("rf")
