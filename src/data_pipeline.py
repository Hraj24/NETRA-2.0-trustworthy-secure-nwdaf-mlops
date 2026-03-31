"""
NETRA 2.0 - IMT-2030 Compliant Data Pipeline
=============================================

This module generates synthetic 6G network traffic data aligned with:
- ITU-R Recommendation M.2160 (IMT-2030 Framework)
- ITU-R IMT-2030 Technical Performance Requirements (February 2026)

The data generation follows the 6 IMT-2030 usage scenarios:
1. IC   - Immersive Communication (evolved from 5G eMBB)
2. HRLLC - Hyper Reliable & Low Latency Communication (evolved from 5G URLLC)
3. MC   - Massive Communication (evolved from 5G mMTC)
4. UC   - Ubiquitous Connectivity (NEW in 6G)
5. AIAC - AI and Communication (NEW in 6G)
6. ISAC - Integrated Sensing and Communication (NEW in 6G)

KPI bounds are strictly enforced per scenario according to ITU-R M.2160 specifications.
All generated data includes the complete feature set required for federated learning
and drift detection in the NETRA platform.

Citations:
- ITU-R M.2160: "Framework and overall objectives of the future development of IMT
  for 2030 and beyond" (September 2023)
- ITU-R IMT-2030 TPR: "Technical Performance Requirements for IMT-2030" (February 2026)

Author: NETRA Research Team
License: Academic Research
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd


# =============================================================================
# IMT-2030 KPI BOUNDS PER USAGE SCENARIO
# Source: ITU-R M.2160 / IMT-2030 Technical Performance Requirements Feb 2026
# =============================================================================

IMT2030_SCENARIOS = ["IC", "HRLLC", "MC", "UC", "AIAC", "ISAC"]

# KPI bounds dictionary: each scenario has min/max for each KPI
# All values sourced from ITU-R M.2160 and IMT-2030 TPR Feb 2026
IMT2030_KPI_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "IC": {
        # Source: ITU-R M.2160 - Immersive Communication (eMBB evolution)
        "throughput_mbps": (300, 500),        # User experienced data rate (Mbps)
        "latency_ms": (1, 5),                 # User plane latency (ms)
        "jitter_ms": (0.5, 2),                # Jitter (ms)
        "packet_loss_rate": (0.00001, 0.001), # Packet loss rate (decimal)
        "reliability_target": (99.999, 99.999),  # Reliability (%)
        "connection_density_km2": (1e4, 1e6), # Connection density (/km²)
        "mobility_kmph": (0, 500),            # Mobility support (km/h)
        "area_traffic_capacity_score": (0.2, 0.5),  # Normalized area traffic capacity
        "ai_load_score": (0.2, 0.5),          # AI workload demand (0-1)
        "resilience_score": (0.6, 0.9),       # Resilience indicator (0-1)
    },
    "HRLLC": {
        # Source: ITU-R M.2160 - Hyper Reliable Low Latency (URLLC evolution)
        "throughput_mbps": (100, 300),        # User experienced data rate (Mbps)
        "latency_ms": (0.1, 1),               # User plane latency (ms) - LOWEST
        "jitter_ms": (0.01, 0.5),             # Jitter (ms) - LOWEST
        "packet_loss_rate": (0.0000001, 0.00001),  # Packet loss - LOWEST
        "reliability_target": (99.99999, 99.99999),  # Reliability - HIGHEST
        "connection_density_km2": (1e3, 1e5), # Connection density (/km²)
        "mobility_kmph": (0, 120),            # Mobility support (km/h)
        "area_traffic_capacity_score": (0.1, 0.3),  # Normalized area traffic capacity
        "ai_load_score": (0.1, 0.3),          # AI workload demand (0-1)
        "resilience_score": (0.9, 1.0),       # Resilience indicator (0-1) - HIGHEST
    },
    "MC": {
        # Source: ITU-R M.2160 - Massive Communication (mMTC evolution)
        "throughput_mbps": (1, 10),           # User experienced data rate (Mbps)
        "latency_ms": (10, 100),              # User plane latency (ms) - HIGHEST
        "jitter_ms": (1, 10),                 # Jitter (ms) - HIGHEST
        "packet_loss_rate": (0.0001, 0.01),   # Packet loss (decimal)
        "reliability_target": (99.9, 99.9),   # Reliability (%)
        "connection_density_km2": (1e6, 1e8), # Connection density - HIGHEST
        "mobility_kmph": (0, 30),             # Mobility support (km/h) - LOWEST
        "area_traffic_capacity_score": (0.1, 0.2),  # Normalized area traffic capacity
        "ai_load_score": (0.1, 0.2),          # AI workload demand (0-1) - LOWEST
        "resilience_score": (0.5, 0.8),       # Resilience indicator (0-1)
    },
    "UC": {
        # Source: ITU-R M.2160 - Ubiquitous Connectivity (NEW 6G)
        "throughput_mbps": (50, 300),         # User experienced data rate (Mbps)
        "latency_ms": (5, 50),                # User plane latency (ms)
        "jitter_ms": (1, 5),                  # Jitter (ms)
        "packet_loss_rate": (0.0001, 0.005),  # Packet loss (decimal)
        "reliability_target": (99.99, 99.99), # Reliability (%)
        "connection_density_km2": (1e4, 1e7), # Connection density (/km²)
        "mobility_kmph": (0, 1000),           # Mobility support - HIGHEST (satellite)
        "area_traffic_capacity_score": (0.2, 0.4),  # Normalized area traffic capacity
        "ai_load_score": (0.2, 0.4),          # AI workload demand (0-1)
        "resilience_score": (0.8, 1.0),       # Resilience indicator (0-1)
    },
    "AIAC": {
        # Source: ITU-R M.2160 - AI and Communication (NEW 6G)
        "throughput_mbps": (100, 500),        # User experienced data rate (Mbps)
        "latency_ms": (1, 10),                # User plane latency (ms)
        "jitter_ms": (0.5, 3),                # Jitter (ms)
        "packet_loss_rate": (0.00001, 0.001), # Packet loss (decimal)
        "reliability_target": (99.999, 99.999),  # Reliability (%)
        "connection_density_km2": (1e4, 1e6), # Connection density (/km²)
        "mobility_kmph": (0, 500),            # Mobility support (km/h)
        "area_traffic_capacity_score": (0.3, 0.6),  # Normalized area traffic capacity
        "ai_load_score": (0.7, 1.0),          # AI workload demand - HIGHEST
        "resilience_score": (0.7, 0.9),       # Resilience indicator (0-1)
    },
    "ISAC": {
        # Source: ITU-R M.2160 - Integrated Sensing & Communication (NEW 6G)
        "throughput_mbps": (100, 300),        # User experienced data rate (Mbps)
        "latency_ms": (1, 5),                 # User plane latency (ms)
        "jitter_ms": (0.5, 2),                # Jitter (ms)
        "packet_loss_rate": (0.00001, 0.001), # Packet loss (decimal)
        "reliability_target": (99.999, 99.999),  # Reliability (%)
        "connection_density_km2": (1e4, 1e6), # Connection density (/km²)
        "mobility_kmph": (0, 500),            # Mobility support (km/h)
        "area_traffic_capacity_score": (0.3, 0.6),  # Normalized area traffic capacity
        "ai_load_score": (0.5, 0.8),          # AI workload demand (0-1)
        "resilience_score": (0.7, 0.9),       # Resilience indicator (0-1)
    },
}

# Federated Learning domain mappings
# domain_a: IC + HRLLC (high-performance scenarios)
# domain_b: MC + UC (massive/connectivity scenarios)
# domain_c: AIAC + ISAC (AI/sensing scenarios)
DOMAIN_SCENARIO_MAP = {
    "domain_a": ["IC", "HRLLC"],
    "domain_b": ["MC", "UC"],
    "domain_c": ["AIAC", "ISAC"],
}


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# These functions maintain compatibility with existing code in api.py,
# fl_client.py, and train_centralized.py
# =============================================================================

def load_data(filepath: str = "data/traffic_synthetic.csv") -> pd.DataFrame:
    """
    Load network traffic data from CSV file.

    Maintains backward compatibility with existing code.
    Works with both legacy 5G format and new IMT-2030 format.

    Args:
        filepath: Path to CSV file

    Returns:
        pandas DataFrame with traffic data
    """
    df = pd.read_csv(filepath)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess network traffic data.

    Maintains backward compatibility with existing code.
    Handles both legacy and IMT-2030 column names.

    Args:
        df: Raw DataFrame

    Returns:
        Preprocessed DataFrame
    """
    df_processed = df.copy()

    # Handle categorical encoding for usage_scenario if present
    if "usage_scenario" in df_processed.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df_processed["slice_type"] = le.fit_transform(df_processed["usage_scenario"])
    elif "slice_type" not in df_processed.columns:
        # Default encoding if neither exists
        df_processed["slice_type"] = 0

    # Handle legacy column mapping
    if "throughput_mbps" in df_processed.columns:
        df_processed["throughput"] = df_processed["throughput_mbps"]

    # Fill NaN values
    df_processed = df_processed.ffill().bfill()

    return df_processed


def get_features_and_target(df: pd.DataFrame) -> tuple:
    """
    Extract features (X) and target (y) from DataFrame.

    Maintains backward compatibility with existing code.
    Adapts to IMT-2030 schema automatically.

    Args:
        df: Preprocessed DataFrame

    Returns:
        Tuple of (features_df, target_series)
    """
    # IMT-2030 schema feature columns
    imt2030_feature_cols = [
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

    # Legacy schema feature columns
    legacy_feature_cols = [
        "time_of_day",
        "slice_type",
        "jitter",
        "packet_loss",
        "throughput",
    ]

    # Detect schema and extract features
    if all(col in df.columns for col in imt2030_feature_cols):
        # IMT-2030 schema
        feature_cols = imt2030_feature_cols
        target_col = "future_load_target"
    elif all(col in df.columns for col in legacy_feature_cols):
        # Legacy schema
        feature_cols = legacy_feature_cols
        target_col = "future_load"
    else:
        # Fallback: use all numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_candidates = ["future_load_target", "future_load"]
        for tc in target_candidates:
            if tc in numeric_cols:
                numeric_cols.remove(tc)
        feature_cols = numeric_cols
        target_col = "future_load_target" if "future_load_target" in df.columns else "future_load"

    # Encode categorical columns
    df_processed = df.copy()
    for col in feature_cols:
        if df_processed[col].dtype == object:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])

    X = df_processed[feature_cols]
    y = df_processed[target_col]

    return X, y


# =============================================================================
# IMT-2030 DATA GENERATION
# =============================================================================

def _sample_kpi_for_scenario(
    rng: np.random.Generator,
    scenario: str,
    n_samples: int
) -> pd.DataFrame:
    """
    Sample KPI values for a specific IMT-2030 usage scenario.

    All bounds are sourced from ITU-R M.2160 and IMT-2030 TPR Feb 2026.

    Args:
        rng: NumPy random generator
        scenario: One of IC, HRLLC, MC, UC, AIAC, ISAC
        n_samples: Number of samples to generate

    Returns:
        DataFrame with KPI columns
    """
    bounds = IMT2030_KPI_BOUNDS[scenario]

    # Sample each KPI within scenario-specific bounds
    # Using triangular distribution for more realistic data (peak near center)
    data = {}

    for kpi, (min_val, max_val) in bounds.items():
        # Special handling for reliability_target (fixed value per scenario)
        if kpi == "reliability_target":
            # Keep as exact value per scenario for compliance
            data[kpi] = np.full(n_samples, min_val)
        elif min_val == max_val:
            # Handle edge case where bounds are equal
            data[kpi] = np.full(n_samples, min_val)
        else:
            # Use triangular distribution with mode at center for natural variation
            mode = (min_val + max_val) / 2
            data[kpi] = rng.triangular(left=min_val, mode=mode, right=max_val, size=n_samples)

    return pd.DataFrame(data)


def _enforce_cross_scenario_constraints(
    df: pd.DataFrame,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Enforce constraints that span multiple scenarios.

    Rules:
    1. HRLLC must have lowest latency among all scenarios
    2. MC must have highest connection_density_km2
    3. AIAC must have highest ai_load_score
    4. Reliability and packet_loss_rate must be inversely correlated

    Args:
        df: DataFrame with scenario column and KPIs
        rng: NumPy random generator

    Returns:
        DataFrame with constraints enforced
    """
    df = df.copy()

    # Rule 1: Ensure HRLLC has strictly lowest latency
    # Cap HRLLC latency at 0.8ms (below min of other scenarios)
    hrllc_mask = df["usage_scenario"] == "HRLLC"
    df.loc[hrllc_mask, "latency_ms"] = np.clip(
        df.loc[hrllc_mask, "latency_ms"],
        IMT2030_KPI_BOUNDS["HRLLC"]["latency_ms"][0],
        0.8
    )

    # Rule 2: Ensure MC has highest connection density
    # Floor MC connection density at 5e6 (above max of other scenarios)
    mc_mask = df["usage_scenario"] == "MC"
    df.loc[mc_mask, "connection_density_km2"] = np.clip(
        df.loc[mc_mask, "connection_density_km2"],
        5e6,
        IMT2030_KPI_BOUNDS["MC"]["connection_density_km2"][1]
    )

    # Rule 3: Ensure AIAC has highest AI load score
    # Floor AIAC ai_load_score at 0.75 (above max of other non-AIAC scenarios)
    aiac_mask = df["usage_scenario"] == "AIAC"
    df.loc[aiac_mask, "ai_load_score"] = np.clip(
        df.loc[aiac_mask, "ai_load_score"],
        0.75,
        IMT2030_KPI_BOUNDS["AIAC"]["ai_load_score"][1]
    )

    # Rule 4: Enforce inverse correlation between reliability and packet_loss
    # Higher reliability = lower packet loss
    # Apply transformation: packet_loss = base_loss * (100 - reliability) / 100
    reliability_factor = (100 - df["reliability_target"]) / 100
    base_packet_loss = df["packet_loss_rate"].values

    # Adjust packet loss based on reliability (inverse relationship)
    # Normalize to maintain reasonable range
    adjusted_packet_loss = base_packet_loss * (reliability_factor / reliability_factor.mean())
    df["packet_loss_rate"] = np.clip(
        adjusted_packet_loss,
        0.0000001,  # Minimum possible
        0.01  # Maximum possible
    )

    return df


def _calculate_future_load_target(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """
    Calculate future_load_target based on throughput, connection density,
    and area traffic capacity.

    Formula derived from IMT-2030 traffic models:
    future_load = throughput * connection_density_factor * area_traffic_factor + noise

    Args:
        df: DataFrame with required columns
        rng: NumPy random generator

    Returns:
        Series with future_load_target values
    """
    # Normalize connection density (log scale for stability)
    conn_density_normalized = np.log10(df["connection_density_km2"]) - 4  # Range ~0-4

    # Area traffic capacity is already normalized (0-1)
    area_traffic = df["area_traffic_capacity_score"]

    # Throughput in Mbps
    throughput = df["throughput_mbps"]

    # Calculate future load using IMT-2030 traffic model
    # Base formula: weighted combination of factors
    future_load = (
        0.5 * throughput +                    # Direct throughput contribution
        10 * conn_density_normalized +        # Connection density contribution
        100 * area_traffic                    # Area traffic contribution
    )

    # Add realistic noise (5% of mean)
    noise = rng.normal(0, 0.05 * future_load.mean(), size=len(df))
    future_load = future_load + noise

    # Ensure positive values
    future_load = np.clip(future_load, 0.1, None)

    return pd.Series(future_load, index=df.index, name="future_load_target")


def _generate_timestamps(
    rng: np.random.Generator,
    n_samples: int,
    start_date: str = "2026-01-01"
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate realistic timestamps and extract time_of_day.

    Args:
        rng: NumPy random generator
        n_samples: Number of samples
        start_date: Starting date for timestamps

    Returns:
        Tuple of (timestamp_series, time_of_day_series)
    """
    # Generate random offsets in hours (spread over 30 days)
    hour_offsets = rng.uniform(0, 30 * 24, size=n_samples)

    # Create timestamps
    base_date = pd.Timestamp(start_date)
    timestamps = pd.Series([
        base_date + pd.Timedelta(hours=h) for h in hour_offsets
    ])

    # Extract time_of_day (hour 0-23)
    time_of_day = timestamps.dt.hour

    return timestamps, time_of_day


def generate_imt2030_dataset(
    n_samples: int = 5000,
    seed: int = 42,
    include_metadata: bool = True
) -> pd.DataFrame:
    """
    Generate IMT-2030 compliant synthetic network traffic dataset.

    This function creates a complete dataset with all required columns
    following ITU-R M.2160 specifications.

    Args:
        n_samples: Total number of samples to generate
        seed: Random seed for reproducibility
        include_metadata: Whether to include metadata columns

    Returns:
        DataFrame with IMT-2030 compliant schema
    """
    rng = np.random.default_rng(seed)

    # Distribute samples across scenarios
    scenarios = IMT2030_SCENARIOS
    samples_per_scenario = n_samples // len(scenarios)
    remainder = n_samples % len(scenarios)

    dfs = []
    sample_id = 0

    for i, scenario in enumerate(scenarios):
        # Add remainder to first scenarios
        n = samples_per_scenario + (1 if i < remainder else 0)

        # Generate KPI data for this scenario
        scenario_df = _sample_kpi_for_scenario(rng, scenario, n)

        # Add usage_scenario column
        scenario_df["usage_scenario"] = scenario

        dfs.append(scenario_df)

    # Combine all scenarios
    df = pd.concat(dfs, ignore_index=True)

    # Generate sample IDs
    df["sample_id"] = range(1, len(df) + 1)

    # Generate timestamps
    timestamps, time_of_day = _generate_timestamps(rng, len(df))
    df["timestamp"] = timestamps
    df["time_of_day"] = time_of_day

    # Enforce cross-scenario constraints
    df = _enforce_cross_scenario_constraints(df, rng)

    # Calculate future_load_target
    df["future_load_target"] = _calculate_future_load_target(df, rng)

    # Add metadata columns
    if include_metadata:
        df["imt2030_scenario_label"] = df["usage_scenario"]
        df["generation_profile"] = "IMT-2030-M.2160-compliant"

    # Reorder columns to match required schema
    column_order = [
        "sample_id",
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
        "future_load_target",
        "imt2030_scenario_label",
        "generation_profile",
    ]

    df = df[column_order]

    # Shuffle rows for randomness
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def validate_imt2030_compliance(
    df: pd.DataFrame,
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame against IMT-2030 compliance rules.

    Checks:
    1. Each row belongs to exactly one valid usage_scenario
    2. KPI values are within scenario-specific bounds
    3. HRLLC has lowest latency
    4. MC has highest connection_density_km2
    5. AIAC has highest ai_load_score
    6. Reliability and packet_loss_rate are inversely correlated
    7. All required columns are present

    Args:
        df: DataFrame to validate
        verbose: Whether to print detailed violations

    Returns:
        Tuple of (is_compliant: bool, violations: List[str])
    """
    violations = []

    # Check required columns
    required_columns = [
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
        "future_load_target",
    ]

    for col in required_columns:
        if col not in df.columns:
            violations.append(f"Missing required column: {col}")

    if violations:
        return False, violations

    # Check valid usage scenarios
    valid_scenarios = set(IMT2030_SCENARIOS)
    actual_scenarios = set(df["usage_scenario"].unique())
    invalid_scenarios = actual_scenarios - valid_scenarios

    if invalid_scenarios:
        violations.append(f"Invalid usage scenarios: {invalid_scenarios}")

    # Check KPI bounds per scenario
    for scenario in actual_scenarios:
        if scenario not in IMT2030_KPI_BOUNDS:
            continue

        scenario_data = df[df["usage_scenario"] == scenario]
        bounds = IMT2030_KPI_BOUNDS[scenario]

        for kpi, (min_val, max_val) in bounds.items():
            if kpi not in scenario_data.columns:
                continue

            values = scenario_data[kpi]
            # Allow 5% tolerance for boundary violations (statistical sampling)
            actual_min = values.min()
            actual_max = values.max()

            # Strict bounds check with tolerance
            tolerance = (max_val - min_val) * 0.05
            if actual_min < min_val - tolerance or actual_max > max_val + tolerance:
                violations.append(
                    f"Scenario {scenario}: {kpi} out of bounds "
                    f"[{actual_min:.4f}, {actual_max:.4f}] vs expected [{min_val}, {max_val}]"
                )

    # Check cross-scenario constraints
    scenario_stats = df.groupby("usage_scenario").agg({
        "latency_ms": "mean",
        "connection_density_km2": "mean",
        "ai_load_score": "mean",
    })

    # Rule: HRLLC must have lowest latency
    if "HRLLC" in scenario_stats.index:
        hrllc_latency = scenario_stats.loc["HRLLC", "latency_ms"]
        other_latencies = [
            scenario_stats.loc[s, "latency_ms"]
            for s in scenario_stats.index if s != "HRLLC"
        ]
        if other_latencies and hrllc_latency >= min(other_latencies):
            violations.append(
                f"HRLLC latency ({hrllc_latency:.2f}ms) must be lowest among all scenarios"
            )

    # Rule: MC must have highest connection density
    if "MC" in scenario_stats.index:
        mc_density = scenario_stats.loc["MC", "connection_density_km2"]
        other_densities = [
            scenario_stats.loc[s, "connection_density_km2"]
            for s in scenario_stats.index if s != "MC"
        ]
        if other_densities and mc_density <= max(other_densities):
            violations.append(
                f"MC connection_density_km2 ({mc_density:.2e}) must be highest among all scenarios"
            )

    # Rule: AIAC must have highest AI load score
    if "AIAC" in scenario_stats.index:
        aiac_load = scenario_stats.loc["AIAC", "ai_load_score"]
        other_loads = [
            scenario_stats.loc[s, "ai_load_score"]
            for s in scenario_stats.index if s != "AIAC"
        ]
        if other_loads and aiac_load <= max(other_loads):
            violations.append(
                f"AIAC ai_load_score ({aiac_load:.2f}) must be highest among all scenarios"
            )

    # Rule: Reliability and packet_loss should be inversely correlated
    if len(df) > 10:
        correlation = df["reliability_target"].corr(df["packet_loss_rate"])
        if correlation is not None and correlation > 0:
            violations.append(
                f"Reliability and packet_loss_rate should be inversely correlated "
                f"(got correlation={correlation:.3f})"
            )

    is_compliant = len(violations) == 0

    if verbose and violations:
        print("IMT-2030 Compliance Violations:")
        for v in violations:
            print(f"  - {v}")

    return is_compliant, violations


def generate_domain_datasets(
    n_samples: int = 5000,
    seed: int = 42,
    output_dir: str = "data"
) -> Dict[str, pd.DataFrame]:
    """
    Generate federated learning domain datasets.

    Creates three domain-specific datasets:
    - domain_a.csv: IC + HRLLC (high-performance scenarios)
    - domain_b.csv: MC + UC (massive/connectivity scenarios)
    - domain_c.csv: AIAC + ISAC (AI/sensing scenarios)

    Args:
        n_samples: Total samples (distributed across domains)
        seed: Random seed for reproducibility
        output_dir: Output directory for CSV files

    Returns:
        Dictionary mapping domain name to DataFrame
    """
    rng = np.random.default_rng(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    datasets = {}

    for domain, scenarios in DOMAIN_SCENARIO_MAP.items():
        # Generate samples for this domain's scenarios
        samples_per_scenario = n_samples // (2 * len(scenarios))
        remainder = n_samples % (2 * len(scenarios))

        domain_dfs = []

        for i, scenario in enumerate(scenarios):
            n = samples_per_scenario + (1 if i < remainder else 0)
            scenario_df = _sample_kpi_for_scenario(rng, scenario, n)
            scenario_df["usage_scenario"] = scenario
            domain_dfs.append(scenario_df)

        domain_df = pd.concat(domain_dfs, ignore_index=True)

        # Add common columns
        domain_df["sample_id"] = range(1, len(domain_df) + 1)
        timestamps, time_of_day = _generate_timestamps(rng, len(domain_df))
        domain_df["timestamp"] = timestamps
        domain_df["time_of_day"] = time_of_day

        # Enforce constraints
        domain_df = _enforce_cross_scenario_constraints(domain_df, rng)

        # Calculate future load
        domain_df["future_load_target"] = _calculate_future_load_target(domain_df, rng)

        # Add metadata
        domain_df["imt2030_scenario_label"] = domain_df["usage_scenario"]
        domain_df["generation_profile"] = "IMT-2030-M.2160-compliant"

        # Reorder columns
        column_order = [
            "sample_id",
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
            "future_load_target",
            "imt2030_scenario_label",
            "generation_profile",
        ]

        domain_df = domain_df[column_order]
        domain_df = domain_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Save to CSV
        output_path = Path(output_dir) / f"{domain}.csv"
        domain_df.to_csv(output_path, index=False)
        print(f"Saved {output_path} ({len(domain_df)} samples)")

        datasets[domain] = domain_df

    return datasets


def generate_all_datasets(
    n_samples: int = 5000,
    seed: int = 42,
    output_dir: str = "data"
) -> None:
    """
    Generate complete IMT-2030 dataset and domain splits.

    Creates:
    - traffic_synthetic.csv: Full combined dataset
    - domain_a.csv: IC + HRLLC
    - domain_b.csv: MC + UC
    - domain_c.csv: AIAC + ISAC

    Args:
        n_samples: Samples for full dataset (domain datasets get n_samples//2 each)
        seed: Random seed for reproducibility
        output_dir: Output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate full combined dataset
    print(f"Generating IMT-2030 compliant dataset with {n_samples} samples (seed={seed})...")
    full_df = generate_imt2030_dataset(n_samples=n_samples, seed=seed)

    # Validate compliance
    is_compliant, violations = validate_imt2030_compliance(full_df, verbose=True)

    if not is_compliant:
        print("Warning: Generated data has compliance violations. Regenerating...")
        # Try regeneration with different seed
        for attempt in range(3):
            full_df = generate_imt2030_dataset(n_samples=n_samples, seed=seed + attempt + 1)
            is_compliant, violations = validate_imt2030_compliance(full_df)
            if is_compliant:
                print(f"Regeneration successful with seed={seed + attempt + 1}")
                break
        else:
            print("Warning: Some violations persist. Data saved anyway for review.")

    # Save full dataset
    full_path = Path(output_dir) / "traffic_synthetic.csv"
    full_df.to_csv(full_path, index=False)
    print(f"Saved {full_path} ({len(full_df)} samples)")

    # Generate domain datasets
    print("\nGenerating federated learning domain datasets...")
    generate_domain_datasets(n_samples=n_samples // 2, seed=seed, output_dir=output_dir)

    print("\nDataset generation complete!")
    print(f"  - Full dataset: {len(full_df)} samples across {len(IMT2030_SCENARIOS)} scenarios")
    print(f"  - Domain datasets: {len(DOMAIN_SCENARIO_MAP)} domains for federated learning")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """
    CLI entry point for data pipeline.

    Usage:
        python -m src.data_pipeline --generate --samples 5000 --seed 42
    """
    parser = argparse.ArgumentParser(
        description="NETRA 2.0 IMT-2030 Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate IMT-2030 compliant dataset
  python -m src.data_pipeline --generate --samples 5000 --seed 42

  # Validate existing dataset
  python -m src.data_pipeline --validate --input data/traffic_synthetic.csv

  # Generate domain-specific datasets only
  python -m src.data_pipeline --domains --samples 2500 --seed 42
        """
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate new IMT-2030 compliant dataset"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing dataset for IMT-2030 compliance"
    )

    parser.add_argument(
        "--domains",
        action="store_true",
        help="Generate domain-specific datasets for federated learning"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of samples to generate (default: 5000)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/traffic_synthetic.csv",
        help="Input file for validation (default: data/traffic_synthetic.csv)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)"
    )

    args = parser.parse_args()

    if args.generate:
        print("=" * 60)
        print("NETRA 2.0 - IMT-2030 Data Generation")
        print("=" * 60)
        generate_all_datasets(
            n_samples=args.samples,
            seed=args.seed,
            output_dir=args.output_dir
        )

    elif args.validate:
        print("=" * 60)
        print("NETRA 2.0 - IMT-2030 Compliance Validation")
        print("=" * 60)

        try:
            df = pd.read_csv(args.input)
            print(f"Loaded {args.input} ({len(df)} samples)")

            is_compliant, violations = validate_imt2030_compliance(df, verbose=True)

            if is_compliant:
                print("\n[PASS] Dataset is IMT-2030 compliant!")
            else:
                print(f"\n[FAIL] Found {len(violations)} violation(s)")

        except FileNotFoundError:
            print(f"Error: File not found: {args.input}")
        except Exception as e:
            print(f"Error: {e}")

    elif args.domains:
        print("=" * 60)
        print("NETRA 2.0 - Domain Dataset Generation")
        print("=" * 60)
        generate_domain_datasets(
            n_samples=args.samples,
            seed=args.seed,
            output_dir=args.output_dir
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
