"""
SHAP Logger for NETRA 2.0 - IMT-2030

Logs SHAP explanation values for audit and analysis.
Updated for IMT-2030 schema with 12 features.

Citations:
- ITU-R M.2160: "Framework and overall objectives of the future development of IMT
  for 2030 and beyond" (September 2023)
"""

import csv
import os
from datetime import datetime


class ShapLogger:
    """
    Logs SHAP values for IMT-2030 predictions.

    Tracks per-feature SHAP contributions and rollback state
    for explainability auditing.
    """

    def __init__(self, path="logs/shap_log.csv"):
        self.path = path
        os.makedirs("logs", exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                # IMT-2030 feature headers (12 features)
                writer.writerow([
                    "timestamp",
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
                    "rollback_active"
                ])

    def log(self, shap_values: dict, rollback_active: bool):
        """
        Log SHAP values for a prediction.

        Args:
            shap_values: Dictionary mapping feature names to SHAP values
            rollback_active: Whether rollback is currently active
        """
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                shap_values.get("time_of_day", 0),
                shap_values.get("usage_scenario", 0),
                shap_values.get("throughput_mbps", 0),
                shap_values.get("latency_ms", 0),
                shap_values.get("jitter_ms", 0),
                shap_values.get("packet_loss_rate", 0),
                shap_values.get("reliability_target", 0),
                shap_values.get("connection_density_km2", 0),
                shap_values.get("mobility_kmph", 0),
                shap_values.get("area_traffic_capacity_score", 0),
                shap_values.get("ai_load_score", 0),
                shap_values.get("resilience_score", 0),
                rollback_active
            ])
