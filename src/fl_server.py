"""
Federated Learning Server for NETRA 2.0 - IMT-2030

Implements a Flower FL server with custom FedAvg strategy that saves
the global model after each aggregation round.

The server coordinates training across 3 domains:
- Domain A: IC + HRLLC (high-performance scenarios)
- Domain B: MC + UC (massive/connectivity scenarios)
- Domain C: AIAC + ISAC (AI/sensing scenarios)

Citations:
- ITU-R M.2160: "Framework and overall objectives of the future development of IMT
  for 2030 and beyond" (September 2023)
"""

import flwr as fl
import joblib
from typing import List, Tuple, Optional
from flwr.common import Parameters, FitRes
from flwr.server.client_proxy import ClientProxy


# IMT-2030 feature count (12 features)
# time_of_day, usage_scenario, throughput_mbps, latency_ms, jitter_ms,
# packet_loss_rate, reliability_target, connection_density_km2,
# mobility_kmph, area_traffic_capacity_score, ai_load_score, resilience_score
NUM_FEATURES = 12


# --------------------------------------------------
# Custom FedAvg Strategy to Save Global Model
# --------------------------------------------------
class SaveModelFedAvg(fl.server.strategy.FedAvg):
    """
    Extended FedAvg strategy that persists the global model
    after each federation round.

    Saves model in joblib format compatible with the inference API.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, dict]]:
        """
        Aggregate client updates and save global model.

        Args:
            server_round: Current federation round number
            results: List of (client, fit_result) tuples
            failures: List of exceptions from failed clients

        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics) or None
        """
        aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated is None:
            return None

        # Unpack return value
        aggregated_parameters, aggregated_metrics = aggregated

        # Convert Parameters -> NumPy arrays
        weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

        # Save global FL model in IMT-2030 format
        model_data = {
            "coef": weights[0],
            "intercept": weights[1],
            "num_features": NUM_FEATURES,
            "model_version": f"FL-IMT2030-Round{server_round}",
        }

        joblib.dump(model_data, "models/fl_global_model.pkl")
        print(f"[SERVER] Saved FL global model at round {server_round}")

        return aggregated_parameters, aggregated_metrics


# --------------------------------------------------
# Start Flower server
# --------------------------------------------------
if __name__ == "__main__":
    strategy = SaveModelFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    print("=" * 60)
    print("NETRA 2.0 - FL Server (IMT-2030)")
    print("=" * 60)
    print("Waiting for clients: domain_a, domain_b, domain_c")
    print("Model: SGDRegressor with 12 IMT-2030 features")
    print()

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
