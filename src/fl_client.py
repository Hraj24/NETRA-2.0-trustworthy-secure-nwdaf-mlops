"""
Federated Learning Client for NETRA 2.0 - IMT-2030

Implements a Flower FL client that trains on domain-specific IMT-2030 data.
Each client represents a network domain with specific usage scenarios:
- Domain A: IC + HRLLC (high-performance scenarios)
- Domain B: MC + UC (massive/connectivity scenarios)
- Domain C: AIAC + ISAC (AI/sensing scenarios)

Citations:
- ITU-R M.2160: "Framework and overall objectives of the future development of IMT
  for 2030 and beyond" (September 2023)
"""

import csv
import os
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from math import sqrt


# IMT-2030 feature columns (12 features)
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


class TrafficClient(fl.client.NumPyClient):
    """
    Flower client for IMT-2030 traffic prediction.

    Trains an SGDRegressor on domain-specific data and participates
    in federated averaging to build a global model.
    """

    def __init__(self, path: str, domain_name: str):
        self.domain_name = domain_name
        df = pd.read_csv(path)

        # Encode categorical columns (usage_scenario -> numeric)
        for col in df.columns:
            if df[col].dtype == object:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

        # Extract features and target
        X = df[IMT2030_FEATURE_COLS].values
        y = df["future_load_target"].values

        # Scale features (IMPORTANT for FL convergence)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Train/test split (80/20)
        n = int(0.8 * len(X))
        self.X_train, self.X_test = X[:n], X[n:]
        self.y_train, self.y_test = y[:n], y[n:]

        # Initialize SGD model (matches server architecture)
        self.model = SGDRegressor(
            max_iter=1,
            learning_rate="constant",
            eta0=0.01,
            random_state=42,
        )

        # Initial local training
        self.model.partial_fit(self.X_train, self.y_train)

        # RMSE log
        self.log_file = f"logs/fl_rmse_{domain_name}.csv"
        os.makedirs("logs", exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(["round", "rmse"])

        self.round = 0

    def get_parameters(self, config):
        """Return model parameters for federation."""
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        """Load aggregated parameters from server."""
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        """
        Perform local training on received parameters.

        Args:
            parameters: Aggregated model parameters from server
            config: Training configuration

        Returns:
            Updated parameters, sample count, metrics
        """
        self.round += 1
        self.set_parameters(parameters)
        self.model.partial_fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """
        Evaluate model on local test set.

        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Tuple of (loss, sample count, metrics dict)
        """
        self.set_parameters(parameters)
        preds = self.model.predict(self.X_test)
        rmse = sqrt(mean_squared_error(self.y_test, preds))

        # Log RMSE per round
        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([self.round, rmse])

        return float(rmse), len(self.X_test), {"rmse": rmse}


def start_client(csv_path: str, domain_name: str):
    """
    Start the Flower client for a specific domain.

    Args:
        csv_path: Path to domain-specific CSV data file
        domain_name: Domain identifier (domain_a, domain_b, domain_c)
    """
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=TrafficClient(csv_path, domain_name),
    )


if __name__ == "__main__":
    # Example usage:
    # python -m src.fl_client --domain domain_a
    import argparse

    parser = argparse.ArgumentParser(description="NETRA 2.0 FL Client")
    parser.add_argument("--domain", type=str, required=True,
                        help="Domain name (domain_a, domain_b, domain_c)")
    args = parser.parse_args()

    domain_map = {
        "domain_a": "data/domain_a.csv",
        "domain_b": "data/domain_b.csv",
        "domain_c": "data/domain_c.csv",
    }

    if args.domain not in domain_map:
        print(f"Error: Unknown domain '{args.domain}'")
        print("Valid domains: domain_a, domain_b, domain_c")
        exit(1)

    print(f"Starting FL client for {args.domain}...")
    start_client(domain_map[args.domain], args.domain)
