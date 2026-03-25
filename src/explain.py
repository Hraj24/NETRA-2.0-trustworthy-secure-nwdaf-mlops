"""
SHAP Explainability Module for NETRA 1.0 - IMT-2030

Generates SHAP-based explanations for IMT-2030 traffic predictions.
Produces global summary plots and local bar charts for model interpretability.

Citations:
- ITU-R M.2160: "Framework and overall objectives of the future development of IMT
  for 2030 and beyond" (September 2023)
"""

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
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


def load_imt2030_model():
    """Load the FL global model from disk."""
    model_data = joblib.load("models/fl_global_model.pkl")
    model = SGDRegressor()
    model.coef_ = np.array(model_data["coef"])
    model.intercept_ = np.array(model_data["intercept"])
    return model


def prepare_data():
    """Load and prepare IMT-2030 dataset for SHAP analysis."""
    df = pd.read_csv("data/traffic_synthetic.csv")

    # Encode categorical columns
    if "usage_scenario" in df.columns:
        le = LabelEncoder()
        df["usage_scenario"] = le.fit_transform(df["usage_scenario"])

    return df


def generate_shap_explanations(n_samples=100, seed=42):
    """
    Generate SHAP explanations for IMT-2030 model.

    Args:
        n_samples: Number of samples for SHAP analysis
        seed: Random seed for reproducibility

    Returns:
        Tuple of (shap_values, data, feature_names)
    """
    # Load model and data
    model = load_imt2030_model()
    df = prepare_data()

    # Prepare features
    X = df[IMT2030_FEATURE_COLS].values

    # Create background data for masker
    background = np.zeros((50, len(IMT2030_FEATURE_COLS)))
    masker = shap.maskers.Independent(background)

    # Create SHAP explainer
    explainer = shap.LinearExplainer(model, masker)

    # Compute SHAP values for sample
    X_sample = X[:n_samples]
    shap_values = explainer.shap_values(X_sample)

    return shap_values, X_sample, IMT2030_FEATURE_COLS


def plot_global_summary(shap_values, X, feature_names, output_path="reports/shap_summary_fl.png"):
    """
    Generate global SHAP summary plot.

    Args:
        shap_values: SHAP values array
        X: Feature data
        feature_names: List of feature names
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        color=plt.cm.viridis,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved global summary: {output_path}")


def plot_local_bar(shap_values, X, feature_names, sample_idx=0, output_path="reports/shap_local_bar_fl.png"):
    """
    Generate local SHAP bar plot for a single sample.

    Args:
        shap_values: SHAP values array
        X: Feature data
        feature_names: List of feature names
        sample_idx: Index of sample to explain
        output_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))
    shap.plots.bar(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=shap_values[sample_idx].sum(),  # Linear explainer
            data=X[sample_idx],
            feature_names=feature_names,
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved local bar chart: {output_path}")


def main():
    """Main SHAP explanation pipeline."""
    print("=" * 60)
    print("NETRA 1.0 - SHAP Explainability (IMT-2030)")
    print("=" * 60)

    # Generate SHAP values
    print("Computing SHAP values...")
    shap_values, X, feature_names = generate_shap_explanations(n_samples=100)

    # Ensure reports directory exists
    import os
    os.makedirs("reports", exist_ok=True)

    # Generate global summary
    print("Generating global summary plot...")
    plot_global_summary(shap_values, X, feature_names)

    # Generate local bar chart
    print("Generating local bar chart...")
    plot_local_bar(shap_values, X, feature_names)

    print("\n[OK] SHAP explanations generated successfully!")
    print("  - reports/shap_summary_fl.png: Global feature importance")
    print("  - reports/shap_local_bar_fl.png: Local explanation sample")


if __name__ == "__main__":
    main()
