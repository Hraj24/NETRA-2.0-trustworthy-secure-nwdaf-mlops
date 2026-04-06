import numpy as np
from collections import deque
from typing import Optional, Dict

# Scenario-adaptive drift thresholds
# HRLLC requires earliest detection (most sensitive), UC most tolerant
DRIFT_THRESHOLDS = {
    "HRLLC": {"threshold": 0.15, "min_windows": 2},  # Ultra-sensitive for low-latency
    "ISAC": {"threshold": 0.20, "min_windows": 2},   # Sensitive for sensing+comms
    "IC": {"threshold": 0.25, "min_windows": 3},     # Immersive comms
    "AIAC": {"threshold": 0.30, "min_windows": 3},   # AI autonomous control
    "MC": {"threshold": 0.35, "min_windows": 3},     # Mobile broadband
    "UC": {"threshold": 0.40, "min_windows": 4},     # Ubiquitous (most tolerant)
}


class MovingAverageDriftDetector:
    """
    Detects concept drift using sliding window mean comparison.
    Supports scenario-adaptive thresholds for IMT-2030 6G use cases.

    Drift is confirmed only after consecutive violations.
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 0.3,
        min_drift_windows: int = 3,
        scenario: Optional[str] = None
    ):
        self.window_size = window_size
        self.scenario = scenario

        # Use scenario-specific thresholds if available
        if scenario and scenario in DRIFT_THRESHOLDS:
            config = DRIFT_THRESHOLDS[scenario]
            self.threshold = config["threshold"]
            self.min_drift_windows = config["min_windows"]
        else:
            self.threshold = threshold
            self.min_drift_windows = min_drift_windows

        self.window = deque(maxlen=window_size)
        self.reference_mean = None
        self.drift_counter = 0

    def update(self, value: float) -> bool:
        self.window.append(value)

        # Not enough data yet
        if len(self.window) < self.window_size:
            return False

        current_mean = np.mean(self.window)

        # Initialize baseline
        if self.reference_mean is None:
            self.reference_mean = current_mean
            return False

        relative_change = abs(current_mean - self.reference_mean) / (
            self.reference_mean + 1e-6
        )

        if relative_change > self.threshold:
            self.drift_counter += 1
        else:
            # Adapt baseline if system stabilizes
            self.drift_counter = 0
            self.reference_mean = current_mean

        return self.drift_counter >= self.min_drift_windows

    def set_scenario(self, scenario: str):
        """Update detector configuration for a new scenario."""
        self.scenario = scenario
        if scenario in DRIFT_THRESHOLDS:
            config = DRIFT_THRESHOLDS[scenario]
            self.threshold = config["threshold"]
            self.min_drift_windows = config["min_windows"]

    def get_config(self) -> Dict:
        """Return current detector configuration."""
        return {
            "scenario": self.scenario,
            "threshold": self.threshold,
            "min_drift_windows": self.min_drift_windows,
            "window_size": self.window_size,
        }
