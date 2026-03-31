# uvicorn src.api:app --reload --port 8000
# http://127.0.0.1:8000/docs --> Open In Chrome using Swagger UI


from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import shap
from datetime import datetime

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from src.drift import MovingAverageDriftDetector
from src.drift_adaptive import ADWINDriftDetector, DDMDriftDetector
from src.rollback import ModelManager
from src.drift_logger import DriftLogger
from src.shap_logger import ShapLogger

from fastapi.middleware.cors import CORSMiddleware
import os

# =========================================================
# App
# =========================================================
app = FastAPI(
    title="Trustworthy NWDAF ML-Ops API (IMT-2030)",
    version="4.0.0",
    description=(
        "Federated Learning inference with hybrid drift detection, "
        "rollback, auto-recovery, and SHAP-based explainability. "
        "Supports both legacy 5G (eMBB/URLLC/mMTC) and IMT-2030 6G "
        "(IC/HRLLC/MC/UC/AIAC/ISAC) usage scenarios."
    ),
)

# CORS: read allowed origins from env (comma-separated), fallback to localhost
_default_origins = "http://localhost:5173,http://localhost:5174,http://localhost:3000"
_cors_origins = os.getenv("CORS_ORIGINS", _default_origins)
ALLOWED_ORIGINS = [o.strip() for o in _cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load FL Global Model
# =========================================================
FL_MODEL_PATH = "models/fl_global_model.pkl"
MODEL_VERSION = "FL-v2.0-IMT2030"

fl_model_data = joblib.load(FL_MODEL_PATH)

fl_model = SGDRegressor()
fl_model.coef_ = np.array(fl_model_data["coef"])
fl_model.intercept_ = np.array(fl_model_data["intercept"])

# Load scaler if available
if "scaler_mean" in fl_model_data and fl_model_data["scaler_mean"] is not None:
    scaler = StandardScaler()
    scaler.mean_ = np.array(fl_model_data["scaler_mean"])
    scaler.scale_ = np.array(fl_model_data["scaler_scale"])
else:
    scaler = None

model_manager = ModelManager(
    stable_model=fl_model,
    primary_model=fl_model
)

# =========================================================
# Drift Detectors (HYBRID STRATEGY)
# =========================================================
ma_detector = MovingAverageDriftDetector(
    window_size=50,
    threshold=0.3,
    min_drift_windows=3
)

adwin_detector = ADWINDriftDetector(delta=0.002)
ddm_detector = DDMDriftDetector()

PRIMARY_DETECTOR_NAME = "MovingAverage"
drift_logger = DriftLogger("drift_events.csv")

# =========================================================
# SHAP Explainer (GLOBAL, SAFE)
# =========================================================
# Updated for IMT-2030: 12 features instead of 6
background = np.zeros((50, 12))
shap_masker = shap.maskers.Independent(background)

shap_explainer = shap.LinearExplainer(
    model_manager.primary_model,
    shap_masker
)

shap_logger = ShapLogger()

# IMT-2030 feature names for SHAP explanations
SHAP_FEATURE_NAMES = [
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

# =========================================================
# Input / Output Schemas
# =========================================================

class TrafficSampleLegacy(BaseModel):
    """Legacy 5G schema for backward compatibility"""
    time_of_day: int = Field(..., ge=0, le=23)
    slice_type: str = Field(..., description="eMBB / URLLC / mMTC")
    jitter: float = Field(..., ge=0)
    packet_loss: float = Field(..., ge=0, le=100)
    throughput: float = Field(..., gt=0)


class TrafficSampleIMT2030(BaseModel):
    """IMT-2030 (6G) schema with full KPI set"""
    time_of_day: int = Field(..., ge=0, le=23)
    usage_scenario: str = Field(..., description="IC / HRLLC / MC / UC / AIAC / ISAC")
    throughput_mbps: float = Field(..., gt=0, description="User experienced data rate (Mbps)")
    latency_ms: float = Field(..., gt=0, description="User plane latency (ms)")
    jitter_ms: float = Field(..., ge=0, description="Jitter (ms)")
    packet_loss_rate: float = Field(..., ge=0, le=1, description="Packet loss rate (decimal)")
    reliability_target: float = Field(..., ge=99.0, le=100.0, description="Reliability target (%)")
    connection_density_km2: float = Field(..., gt=0, description="Connection density (/km²)")
    mobility_kmph: float = Field(..., ge=0, description="Mobility support (km/h)")
    area_traffic_capacity_score: float = Field(..., ge=0, le=1, description="Area traffic capacity (normalized)")
    ai_load_score: float = Field(..., ge=0, le=1, description="AI workload demand (0-1)")
    resilience_score: float = Field(..., ge=0, le=1, description="Resilience indicator (0-1)")


class PredictionResponse(BaseModel):
    predicted_future_load: float
    explanation: str
    model_version: str
    warning: str | None = None


class ExplainResponse(BaseModel):
    prediction: float
    shap_values: dict
    note: str


# =========================================================
# Utilities
# =========================================================

# Legacy slice type mapping (5G)
SLICE_MAP_LEGACY = {"eMBB": 0, "URLLC": 1, "mMTC": 2}

# IMT-2030 usage scenario mapping (6G)
SCENARIO_MAP_IMT2030 = {
    "IC": 0,
    "HRLLC": 1,
    "MC": 2,
    "UC": 3,
    "AIAC": 4,
    "ISAC": 5,
}


def generate_explanation(sample: TrafficSampleIMT2030) -> str:
    """Generate human-readable explanation based on IMT-2030 KPIs."""
    reasons = []

    # Check IMT-2030 specific thresholds
    if sample.jitter_ms > 5:
        reasons.append("high jitter")
    if sample.packet_loss_rate > 0.01:
        reasons.append("high packet loss")
    if sample.latency_ms > 50:
        reasons.append("high latency")
    if sample.throughput_mbps < 10:
        reasons.append("low throughput")
    if sample.ai_load_score > 0.8:
        reasons.append("high AI workload")

    if reasons:
        return f"Predicted load may increase due to {', '.join(reasons)}."
    return "Network conditions appear stable with moderate predicted load."


def build_feature_vector_legacy(sample: TrafficSampleLegacy) -> np.ndarray:
    """Build feature vector from legacy 5G schema."""
    slice_encoded = SLICE_MAP_LEGACY.get(sample.slice_type, 0)

    return np.array([[
        sample.time_of_day,
        slice_encoded,
        sample.jitter,
        sample.packet_loss,
        sample.throughput,
        0.0  # placeholder (matches FL training)
    ]])


def build_feature_vector_imt2030(sample: TrafficSampleIMT2030) -> np.ndarray:
    """Build feature vector from IMT-2030 schema."""
    scenario_encoded = SCENARIO_MAP_IMT2030.get(sample.usage_scenario, 0)

    return np.array([[
        float(sample.time_of_day),
        float(scenario_encoded),
        sample.throughput_mbps,
        sample.latency_ms,
        sample.jitter_ms,
        sample.packet_loss_rate,
        sample.reliability_target,
        sample.connection_density_km2,
        sample.mobility_kmph,
        sample.area_traffic_capacity_score,
        sample.ai_load_score,
        sample.resilience_score,
    ]])


def build_feature_vector(sample) -> np.ndarray:
    """Build feature vector from either legacy or IMT-2030 sample."""
    if isinstance(sample, TrafficSampleIMT2030):
        return build_feature_vector_imt2030(sample)
    else:
        return build_feature_vector_legacy(sample)


# =========================================================
# Prediction Endpoint
# =========================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(sample: TrafficSampleIMT2030):
    """
    Predict future network traffic load with drift-aware inference.

    Accepts IMT-2030 (6G) schema with full KPI set.
    Returns prediction with model version and drift warning if applicable.
    """
    X = build_feature_vector(sample)

    # Apply scaler if available
    if scaler is not None:
        X = scaler.transform(X)

    # SLA metric for drift detection (using jitter + packet loss)
    sla_metric = sample.jitter_ms + sample.packet_loss_rate

    # Drift signals from hybrid detectors
    ma_drift = ma_detector.update(sla_metric)
    adwin_drift = adwin_detector.update(sla_metric)
    ddm_drift = ddm_detector.update(sla_metric)

    recovered = False

    # ---- Rollback ONLY on MA drift ----
    if ma_drift and not model_manager.rollback_active:
        model_manager.rollback()

        drift_logger.log(
            detector_name=PRIMARY_DETECTOR_NAME,
            sla_metric=sla_metric,
            model_version=MODEL_VERSION,
            action="rollback"
        )

    # ---- Auto-recovery ----
    recovered = model_manager.try_recover(system_stable=not ma_drift)

    if recovered:
        drift_logger.log(
            detector_name=PRIMARY_DETECTOR_NAME,
            sla_metric=sla_metric,
            model_version=MODEL_VERSION,
            action="recovered"
        )

    # ---- Prediction ----
    prediction = model_manager.predict(X)[0]

    # ---- Warning message ----
    warning = None
    if model_manager.rollback_active:
        warning = "⚠ Concept drift detected. Rolled back to stable model."
    elif recovered:
        warning = "✅ System stabilized. Primary model restored."
    elif adwin_drift or ddm_drift:
        warning = "ℹ Early drift warning detected (adaptive detector)."

    return PredictionResponse(
        predicted_future_load=float(prediction),
        explanation=generate_explanation(sample),
        model_version=f"{MODEL_VERSION} ({model_manager.status()})",
        warning=warning
    )


# =========================================================
# Explain Endpoint (SHAP)
# =========================================================
@app.post("/explain", response_model=ExplainResponse)
def explain(sample: TrafficSampleIMT2030):
    """
    Get SHAP-based feature attributions for a prediction.

    Returns per-feature contribution values explaining how each
    IMT-2030 KPI influenced the prediction.
    """
    X = build_feature_vector(sample)

    # Apply scaler if available
    if scaler is not None:
        X = scaler.transform(X)

    prediction = model_manager.predict(X)[0]

    shap_values_full = shap_explainer.shap_values(X)
    shap_values = shap_values_full[0]  # IMT-2030 has 12 features

    shap_dict = {
        name: float(val)
        for name, val in zip(SHAP_FEATURE_NAMES, shap_values)
    }

    # Log SHAP for rollback correlation
    shap_logger.log(
        shap_values=shap_dict,
        rollback_active=model_manager.rollback_active
    )

    return ExplainResponse(
        prediction=float(prediction),
        shap_values=shap_dict,
        note="SHAP values indicate per-feature contribution to the prediction"
    )


# =========================================================
# Health Endpoint
# =========================================================
@app.get("/health")
def health():
    """Return system health, model version, and drift detector status."""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "primary_detector": PRIMARY_DETECTOR_NAME,
        "rollback_active": model_manager.rollback_active,
        "recovery_counter": model_manager.recovery_counter,
    }
