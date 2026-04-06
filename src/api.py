# uvicorn src.api:app --reload --port 8000
# http://127.0.0.1:8000/docs --> Open In Chrome using Swagger UI


from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import joblib
import numpy as np
import shap
from datetime import datetime
import hashlib
import time
from collections import deque
import os
import csv
from pathlib import Path
import httpx

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from src.drift import MovingAverageDriftDetector
from src.drift_adaptive import ADWINDriftDetector, DDMDriftDetector
from src.rollback import ModelManager
from src.drift_logger import DriftLogger
from src.shap_logger import ShapLogger

from fastapi.middleware.cors import CORSMiddleware

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

# Webhook configuration for drift/rollback notifications
DRIFT_WEBHOOK_URL = os.getenv("DRIFT_WEBHOOK_URL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load FL Global Model + Per-Scenario Heads
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

# Per-scenario model heads (fine-tuned offsets from global model)
# Each scenario has a small coefficient adjustment for domain adaptation
SCENARIO_HEADS_PATH = "models/scenario_heads.pkl"
scenario_heads: dict = {}

if os.path.exists(SCENARIO_HEADS_PATH):
    scenario_heads = joblib.load(SCENARIO_HEADS_PATH)
    # scenario_heads = {"HRLLC": {"coef": [...], "intercept": ...}, ...}


class ScenarioModelManager:
    """
    Model manager with per-scenario fine-tuned heads.
    Falls back to global model if scenario head not available.
    """

    def __init__(self, global_model, scenario_heads: dict, stable_model=None):
        self.global_model = global_model
        self.stable_model = stable_model or global_model
        self.scenario_heads = scenario_heads
        self.current_model = self.global_model
        self.rollback_active = False
        self.recovery_counter = 0
        self.recovery_required_windows = 5
        self.current_scenario: str | None = None

    def get_model_for_scenario(self, scenario: str):
        """Get scenario-specific model or fall back to global."""
        if scenario in self.scenario_heads:
            # Create a copy with scenario-specific adjustments
            scenario_model = SGDRegressor()
            scenario_model.coef_ = self.global_model.coef_ + np.array(
                self.scenario_heads[scenario]["coef"]
            )
            scenario_model.intercept_ = (
                self.global_model.intercept_
                + self.scenario_heads[scenario]["intercept"]
            )
            self.current_scenario = scenario
            self.current_model = scenario_model
            return scenario_model
        else:
            self.current_scenario = None
            self.current_model = self.global_model
            return self.global_model

    def rollback(self):
        self.current_model = self.stable_model
        self.rollback_active = True
        self.recovery_counter = 0

    def try_recover(self, system_stable: bool):
        if not self.rollback_active:
            return False

        if system_stable:
            self.recovery_counter += 1
        else:
            self.recovery_counter = 0

        if self.recovery_counter >= self.recovery_required_windows:
            # Restore scenario-specific model if available
            if self.current_scenario and self.current_scenario in self.scenario_heads:
                self.get_model_for_scenario(self.current_scenario)
            else:
                self.current_model = self.global_model
            self.rollback_active = False
            self.recovery_counter = 0
            return True

        return False

    def predict(self, X):
        return self.current_model.predict(X)

    def status(self):
        if self.rollback_active:
            return "rollback_active"
        elif self.current_scenario:
            return f"scenario:{self.current_scenario}"
        return "normal"


model_manager = ScenarioModelManager(
    global_model=fl_model,
    scenario_heads=scenario_heads,
    stable_model=fl_model
)

# =========================================================
# Drift Detectors (HYBRID STRATEGY)
# =========================================================
# Note: MA detector now uses scenario-adaptive thresholds per request
ma_detector = MovingAverageDriftDetector(
    window_size=50,
    scenario=None  # Will be set per-request based on sample
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
    model_manager.global_model,
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
# SHAP Cache (LRU for repeated inputs)
# =========================================================
# Cache SHAP values by input hash to avoid redundant computation
# Typical latency reduction: ~200ms -> ~5ms for cache hits
SHAP_CACHE_MAX_SIZE = 256
_shap_cache: dict = {}


def _compute_input_hash(X: np.ndarray) -> str:
    """Compute hash of input features for cache lookup."""
    return hashlib.sha256(X.tobytes()).hexdigest()[:16]


def _get_cached_shap(X: np.ndarray) -> dict | None:
    """Get cached SHAP values if available."""
    cache_key = _compute_input_hash(X)
    return _shap_cache.get(cache_key)


def _cache_shap(X: np.ndarray, shap_values: dict):
    """Cache SHAP values with LRU eviction."""
    cache_key = _compute_input_hash(X)
    if len(_shap_cache) >= SHAP_CACHE_MAX_SIZE:
        # Remove oldest 10% of cache
        keys_to_remove = list(_shap_cache.keys())[:SHAP_CACHE_MAX_SIZE // 10]
        for key in keys_to_remove:
            del _shap_cache[key]
    _shap_cache[cache_key] = shap_values


def _get_cache_stats() -> dict:
    """Return cache statistics."""
    return {
        "cached_entries": len(_shap_cache),
        "max_size": SHAP_CACHE_MAX_SIZE,
    }


# =========================================================
# Prediction Latency Monitoring (Per-Scenario)
# =========================================================
# Track inference latency per IMT-2030 scenario for SLA compliance.
# Each record stores latency_ms, scenario tag, and timestamp.
LATENCY_WINDOW_SIZE = 1000
_latency_records: deque = deque(maxlen=LATENCY_WINDOW_SIZE)
_current_predict_scenario: dict = {"scenario": None}  # mutable container for middleware


def _record_latency(latency_ms: float):
    """Record prediction latency with scenario tag."""
    _latency_records.append({
        "latency_ms": latency_ms,
        "scenario": _current_predict_scenario.get("scenario"),
        "timestamp": time.time(),
    })


def _filter_latency_records(scenario: str | None = None) -> list[float]:
    """Filter latency records by scenario. None = all."""
    if scenario is None or scenario.lower() == "all":
        return [r["latency_ms"] for r in _latency_records]
    return [r["latency_ms"] for r in _latency_records if r["scenario"] == scenario]


def _get_latency_stats(scenario: str | None = None) -> dict:
    """Return latency percentiles (p50, p95, p99), optionally filtered by scenario."""
    latencies = _filter_latency_records(scenario)
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "samples": 0}

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return {
        "p50": round(sorted_latencies[int(n * 0.50)], 3),
        "p95": round(sorted_latencies[int(n * 0.95)], 3),
        "p99": round(sorted_latencies[min(int(n * 0.99), n - 1)], 3),
        "samples": n,
    }


def _get_adaptive_buckets(sla_target_ms: float) -> list[tuple[float, float | None]]:
    """Generate histogram bucket boundaries adapted to the scenario SLA target."""
    t = sla_target_ms
    if t <= 1.0:       # HRLLC
        return [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, None)]
    elif t <= 5.0:     # ISAC
        return [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, None)]
    elif t <= 10.0:    # IC
        return [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, None)]
    elif t <= 20.0:    # AIAC
        return [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, None)]
    elif t <= 50.0:    # MC
        return [(0, 10), (10, 20), (20, 50), (50, 75), (75, 100), (100, None)]
    else:              # UC
        return [(0, 20), (20, 50), (50, 100), (100, 150), (150, 200), (200, None)]


def _build_histogram(latencies: list[float], sla_target_ms: float) -> list[dict]:
    """Build histogram with adaptive buckets based on SLA target."""
    buckets = _get_adaptive_buckets(sla_target_ms)
    histogram = []

    for start, end in buckets:
        if end is None:
            label = f"{start:.0f}+" if start == int(start) else f"{start}+"
            count = sum(1 for v in latencies if v >= start)
        else:
            label = f"{start:.0f}-{end:.0f}" if start == int(start) and end == int(end) else f"{start}-{end}"
            count = sum(1 for v in latencies if start <= v < end)

        histogram.append({
            "bucket": label,
            "count": count,
            "range_start": start,
            "range_end": end,
        })

    return histogram


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    """Middleware to track prediction endpoint latency."""
    if request.url.path == "/predict":
        start_time = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start_time) * 1000
        _record_latency(latency_ms)
        return response
    return await call_next(request)


# =========================================================
# Webhook Notifications
# =========================================================
async def _send_drift_webhook(event_type: str, details: dict):
    """Send webhook notification for drift/rollback events."""
    if not DRIFT_WEBHOOK_URL:
        return

    payload = {
        "event": event_type,
        "timestamp": datetime.now().isoformat(),
        "model_version": MODEL_VERSION,
        **details,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(DRIFT_WEBHOOK_URL, json=payload)
    except Exception as e:
        # Log but don't fail the request
        print(f"Webhook notification failed: {e}")

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

# =========================================================
# SLA Thresholds per Scenario
# =========================================================
# HRLLC (High-Reliability Low-Latency Communications) requires ultra-low latency
# Thresholds based on IMT-2030 6G requirements
SLA_THRESHOLDS = {
    "HRLLC": {"latency_ms": 1.0, "jitter_ms": 0.1, "packet_loss_rate": 0.00001},  # Ultra-reliable low latency
    "IC": {"latency_ms": 10.0, "jitter_ms": 1.0, "packet_loss_rate": 0.001},       # Immersive communications
    "MC": {"latency_ms": 50.0, "jitter_ms": 5.0, "packet_loss_rate": 0.01},        # Mobile broadband
    "UC": {"latency_ms": 100.0, "jitter_ms": 10.0, "packet_loss_rate": 0.05},      # Ubiquitous connectivity
    "AIAC": {"latency_ms": 20.0, "jitter_ms": 2.0, "packet_loss_rate": 0.005},     # AI-driven autonomous control
    "ISAC": {"latency_ms": 5.0, "jitter_ms": 0.5, "packet_loss_rate": 0.0001},     # Integrated sensing and comms
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
    Uses scenario-adaptive drift detection thresholds.
    """
    # Tag latency record with the current scenario (read by middleware)
    _current_predict_scenario["scenario"] = sample.usage_scenario

    X = build_feature_vector(sample)

    # Apply scaler if available
    if scaler is not None:
        X = scaler.transform(X)

    # Set scenario-specific drift thresholds and load scenario head
    ma_detector.set_scenario(sample.usage_scenario)
    model_manager.get_model_for_scenario(sample.usage_scenario)

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

        # Send webhook notification (fire-and-forget)
        import asyncio
        asyncio.create_task(_send_drift_webhook("rollback", {
            "scenario": sample.usage_scenario,
            "sla_metric": sla_metric,
            "detector": PRIMARY_DETECTOR_NAME,
        }))

    # ---- Auto-recovery ----
    recovered = model_manager.try_recover(system_stable=not ma_drift)

    if recovered:
        drift_logger.log(
            detector_name=PRIMARY_DETECTOR_NAME,
            sla_metric=sla_metric,
            model_version=MODEL_VERSION,
            action="recovered"
        )

        # Send webhook notification for recovery
        import asyncio
        asyncio.create_task(_send_drift_webhook("recovery", {
            "scenario": sample.usage_scenario,
            "sla_metric": sla_metric,
        }))

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
    Uses LRU cache to avoid redundant computation for similar inputs.
    """
    X = build_feature_vector(sample)

    # Apply scaler if available
    if scaler is not None:
        X = scaler.transform(X)

    prediction = model_manager.predict(X)[0]

    # Check cache first
    cached_shap = _get_cached_shap(X)
    if cached_shap is not None:
        return ExplainResponse(
            prediction=float(prediction),
            shap_values=cached_shap,
            note="SHAP values from cache (hit)"
        )

    # Compute SHAP values
    shap_values_full = shap_explainer.shap_values(X)
    shap_values = shap_values_full[0]  # IMT-2030 has 12 features

    shap_dict = {
        name: float(val)
        for name, val in zip(SHAP_FEATURE_NAMES, shap_values)
    }

    # Cache for future requests
    _cache_shap(X, shap_dict)

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
    """Return system health, model version, drift detector status, and cache stats."""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "primary_detector": PRIMARY_DETECTOR_NAME,
        "rollback_active": model_manager.rollback_active,
        "recovery_counter": model_manager.recovery_counter,
        "shap_cache": _get_cache_stats(),
        "latency_stats": _get_latency_stats(),
    }


# =========================================================
# Latency Histogram Endpoint (Per-Scenario)
# =========================================================
@app.get("/latency")
def get_latency(scenario: str | None = None):
    """
    Return prediction latency histogram with percentiles and SLA compliance.

    Query params:
        scenario: IMT-2030 scenario name (IC/HRLLC/MC/UC/AIAC/ISAC) or 'all'.
                  Defaults to 'all' (global view).
    """
    # Determine SLA target based on scenario
    if scenario and scenario.upper() in SLA_THRESHOLDS:
        scenario_key = scenario.upper()
        sla_target_ms = SLA_THRESHOLDS[scenario_key]["latency_ms"]
    else:
        scenario_key = "all"
        sla_target_ms = 10.0  # Default global SLA target (IC-level)

    # Filter latencies by scenario
    filter_scenario = scenario_key if scenario_key != "all" else None
    latencies = _filter_latency_records(filter_scenario)

    # Percentiles
    stats = _get_latency_stats(filter_scenario)

    # Histogram
    histogram = _build_histogram(latencies, sla_target_ms)

    # Mean and SLA compliance
    total_samples = len(latencies)
    mean_ms = round(sum(latencies) / total_samples, 3) if total_samples > 0 else 0
    sla_compliant = sum(1 for v in latencies if v <= sla_target_ms)
    sla_compliance_pct = round((sla_compliant / total_samples) * 100, 1) if total_samples > 0 else 0

    return {
        "scenario": scenario_key,
        "sla_target_ms": sla_target_ms,
        "percentiles": {
            "p50": stats["p50"],
            "p95": stats["p95"],
            "p99": stats["p99"],
        },
        "histogram": histogram,
        "total_samples": total_samples,
        "mean_ms": mean_ms,
        "sla_compliance_pct": sla_compliance_pct,
    }


# =========================================================
# Drift Log Endpoint
# =========================================================
@app.get("/drift-log")
def get_drift_log():
    """Return recent drift events from the log file."""
    log_file = Path("drift_events.csv")
    if not log_file.exists():
        return []

    events = []
    with open(log_file, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                "timestamp": row["timestamp"],
                "detector": row["detector"],
                "sla_metric": float(row["sla_metric"]),
                "model_version": row["model_version"],
                "action": row["action"],
                "type": "drift" if row["action"] == "rollback" else "recovery" if row["action"] == "recovered" else "warning",
            })
    return list(reversed(events))[-100:]  # Last 100 events, reversed for chronological order


# =========================================================
# SLA Alerts Endpoint
# =========================================================
class AlertResponse(BaseModel):
    alerts: list
    predicted_latency_breach: bool
    predicted_jitter_breach: bool
    predicted_packet_loss_breach: bool
    scenario: str
    thresholds: dict


def predict_kpi_from_load(predicted_load: float, scenario: str) -> dict:
    """
    Predict KPI values based on predicted load and scenario.
    Uses a simple linear model scaled by scenario requirements.
    """
    base_latency = SLA_THRESHOLDS[scenario]["latency_ms"]
    base_jitter = SLA_THRESHOLDS[scenario]["jitter_ms"]
    base_packet_loss = SLA_THRESHOLDS[scenario]["packet_loss_rate"]

    # Scale KPIs based on load (normalized to 0-1 range, assuming max load ~100)
    load_factor = min(predicted_load / 100.0, 1.5)

    return {
        "predicted_latency_ms": base_latency * (1 + load_factor * 2),
        "predicted_jitter_ms": base_jitter * (1 + load_factor * 2),
        "predicted_packet_loss_rate": base_packet_loss * (1 + load_factor * 3),
    }


@app.post("/alerts", response_model=AlertResponse)
def check_sla_alerts(sample: TrafficSampleIMT2030):
    """
    Check if predicted network load will breach SLA thresholds for the given scenario.

    Returns alerts when predicted KPIs exceed per-scenario SLA thresholds.
    HRLLC scenario has the strictest thresholds (latency < 1ms, jitter < 0.1ms).
    """
    X = build_feature_vector(sample)
    if scaler is not None:
        X = scaler.transform(X)

    # Get prediction
    predicted_load = model_manager.predict(X)[0]

    # Get scenario thresholds
    scenario = sample.usage_scenario
    thresholds = SLA_THRESHOLDS.get(scenario, SLA_THRESHOLDS["MC"])

    # Predict KPIs from load
    predicted_kpis = predict_kpi_from_load(predicted_load, scenario)

    # Check for breaches
    alerts = []

    latency_breach = predicted_kpis["predicted_latency_ms"] > thresholds["latency_ms"]
    jitter_breach = predicted_kpis["predicted_jitter_ms"] > thresholds["jitter_ms"]
    packet_loss_breach = predicted_kpis["predicted_packet_loss_rate"] > thresholds["packet_loss_rate"]

    if latency_breach:
        alerts.append({
            "severity": "critical" if scenario == "HRLLC" else "warning",
            "type": "latency",
            "message": f"Predicted latency ({predicted_kpis['predicted_latency_ms']:.2f}ms) exceeds SLA threshold ({thresholds['latency_ms']}ms) for {scenario}",
            "predicted_value": predicted_kpis["predicted_latency_ms"],
            "threshold": thresholds["latency_ms"],
        })

    if jitter_breach:
        alerts.append({
            "severity": "critical" if scenario == "HRLLC" else "warning",
            "type": "jitter",
            "message": f"Predicted jitter ({predicted_kpis['predicted_jitter_ms']:.3f}ms) exceeds SLA threshold ({thresholds['jitter_ms']}ms) for {scenario}",
            "predicted_value": predicted_kpis["predicted_jitter_ms"],
            "threshold": thresholds["jitter_ms"],
        })

    if packet_loss_breach:
        alerts.append({
            "severity": "critical" if scenario == "HRLLC" else "warning",
            "type": "packet_loss",
            "message": f"Predicted packet loss ({predicted_kpis['predicted_packet_loss_rate']:.6f}) exceeds SLA threshold ({thresholds['packet_loss_rate']}) for {scenario}",
            "predicted_value": predicted_kpis["predicted_packet_loss_rate"],
            "threshold": thresholds["packet_loss_rate"],
        })

    return AlertResponse(
        alerts=alerts,
        predicted_latency_breach=latency_breach,
        predicted_jitter_breach=jitter_breach,
        predicted_packet_loss_breach=packet_loss_breach,
        scenario=scenario,
        thresholds=thresholds,
    )
