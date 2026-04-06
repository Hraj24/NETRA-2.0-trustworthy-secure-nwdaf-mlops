"""
Create per-scenario fine-tuned model heads.

In production, these would be trained on scenario-specific data.
For now, we create small offset adjustments to simulate domain adaptation.
"""

import joblib
import numpy as np

# Scenario heads: small coefficient adjustments per scenario
# These simulate fine-tuning on domain-specific traffic patterns
SCENARIO_HEADS = {
    "HRLLC": {
        "coef": np.array([0.1, 0.5, -0.2, -0.8, -0.3, -0.4, 0.1, 0.05, -0.1, -0.05, -0.2, 0.1]),
        "intercept": -2.0,  # Lower baseline for ultra-low latency scenario
    },
    "IC": {
        "coef": np.array([0.2, 0.1, 0.4, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.3, 0.1, 0.05]),
        "intercept": 5.0,  # Higher baseline for immersive comms
    },
    "MC": {
        "coef": np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.05, 0.15, 0.1, 0.1, 0.1, 0.05]),
        "intercept": 3.0,
    },
    "UC": {
        "coef": np.array([0.05, 0.1, 0.15, 0.05, 0.05, 0.05, 0.02, 0.2, 0.05, 0.05, 0.05, 0.02]),
        "intercept": 1.0,  # Lower density baseline
    },
    "AIAC": {
        "coef": np.array([0.1, 0.15, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1, 0.5, 0.1]),
        "intercept": 4.0,  # AI load is key feature
    },
    "ISAC": {
        "coef": np.array([0.1, 0.2, 0.25, -0.3, -0.2, -0.2, 0.1, 0.05, 0.15, 0.2, 0.1, 0.3]),
        "intercept": 2.0,  # Sensing + comms integration
    },
}

# Save scenario heads
joblib.dump(SCENARIO_HEADS, "models/scenario_heads.pkl")
print("Created models/scenario_heads.pkl with per-scenario fine-tuned offsets")
print("\nScenario heads created:")
for scenario, head in SCENARIO_HEADS.items():
    print(f"  {scenario}: intercept offset = {head['intercept']}")
