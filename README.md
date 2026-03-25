<p align="center">
  <h1 align="center">🛡️ NETRA 2.0</h1>
  <p align="center"><strong>Trustworthy & Secure NWDAF MLOps Platform</strong></p>
  <p align="center">
    <strong>6G IMT-2030 Compliant</strong> · Federated Learning · Hybrid Drift Detection · SHAP Explainability · Auto-Rollback & Recovery
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.123-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Flower-1.24-FF6F61?logo=data:image/svg+xml;base64,&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" />
</p>

---

## 📌 Overview

**NETRA** (Network Traffic Risk Analyzer) is a trustworthy MLOps platform built around the 3GPP **NWDAF** (Network Data Analytics Function) framework, now upgraded for **6G IMT-2030** compliance. It delivers real-time network traffic load predictions with built-in safeguards for production ML systems:

- **IMT-2030 Compliance** — Aligned with ITU-R M.2160 framework and 6G usage scenarios
- **Federated Learning** — Privacy-preserving model training across distributed network domains using Flower (FedAvg)
- **Hybrid Drift Detection** — Three concurrent detectors (Moving Average, ADWIN, DDM) monitor SLA metrics for concept drift
- **Automatic Rollback & Recovery** — Seamless fallback to a stable model on drift, with auto-recovery after stabilization
- **SHAP Explainability** — Per-prediction feature attribution using SHAP linear explainer for full transparency
- **Interactive Dashboard** — React-based UI for predictions, drift warnings, and SHAP visualizations

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    NETRA 2.0 Platform                    │
├────────────────────────┬─────────────────────────────────┤
│   React Dashboard      │     FastAPI Backend             │
│   (Port 3000)          │     (Port 8000)                 │
│                        │                                 │
│  ┌──────────────┐      │  ┌───────────┐ ┌────────────┐  │
│  │ IMT-2030     │──────┼─▶│ /predict  │ │ /explain   │  │
│  │ Input Form   │      │  │ /health   │ │            │  │
│  │ 6G Scenarios │◀─────┼──│           │ │            │  │
│  │ 12 KPIs      │      │  └─────┬─────┘ └─────┬──────┘  │
│  │ SHAP Panel   │      │        │              │         │
│  │ Status Panel │      │  ┌─────▼──────────────▼──────┐  │
│  └──────────────┘      │  │    Model Manager           │  │
│                        │  │  (rollback / auto-recover) │  │
│                        │  └─────┬─────────────────────┘  │
│                        │        │                        │
│                        │  ┌─────▼─────────────────────┐  │
│                        │  │  Hybrid Drift Detectors    │  │
│                        │  │  MA · ADWIN · DDM          │  │
│                        │  └───────────────────────────┘  │
│                        │                                 │
│                        │  ┌───────────────────────────┐  │
│                        │  │  FL Global Model (SGD)     │  │
│                        │  │  12 IMT-2030 features      │  │
│                        │  │  trained via Flower FedAvg │  │
│                        │  └───────────────────────────┘  │
└────────────────────────┴─────────────────────────────────┘
```

---

## 📂 Project Structure

```
NETRA-2.0/
├── src/                          # Python backend source (IMT-2030)
│   ├── api.py                    # FastAPI app — /predict, /explain, /health
│   ├── data_pipeline.py          # IMT-2030 data generation & validation
│   ├── drift.py                  # Moving Average drift detector
│   ├── drift_adaptive.py         # ADWIN & DDM drift detectors (river)
│   ├── drift_logger.py           # CSV-based drift event auditing
│   ├── rollback.py               # ModelManager — rollback & auto-recovery
│   ├── explain.py                # SHAP explanation utilities
│   ├── shap_logger.py            # SHAP value logging for audit
│   ├── fl_server.py              # Flower FL server (FedAvg + model save)
│   ├── fl_client.py              # Flower FL client (per-domain training)
│   ├── train_centralized.py      # Centralized training baseline
│   ├── train_and_save_models.py  # Model training & serialization
│   └── models.py                 # Model builders (RF, FNN)
│
├── nwdaf-dashboard/              # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx               # Main app with tabbed layout
│   │   ├── api.js                # Backend API client
│   │   └── components/
│   │       ├── InputForm.jsx     # IMT-2030 traffic sample input (12 KPIs)
│   │       ├── PredictionPanel.jsx
│   │       ├── ExplainPanel.jsx  # SHAP visualization (12 features)
│   │       └── StatusPanel.jsx   # System health & drift status
│   └── Dockerfile                # Multi-stage nginx build
│
├── models/                       # Trained model artifacts
│   ├── fl_global_model.pkl       # Federated Learning global model (12 features)
│   ├── model_v1.pkl              # Stable baseline model
│   ├── model_v2_bad.pkl          # Intentionally degraded (for testing)
│   └── ...                       # Additional model checkpoints
│
├── data/                         # Synthetic 6G network traffic datasets
│   ├── traffic_synthetic.csv     # Full IMT-2030 dataset (5000 samples)
│   ├── domain_a.csv              # IC + HRLLC scenarios (1250 samples)
│   ├── domain_b.csv              # MC + UC scenarios (1250 samples)
│   └── domain_c.csv              # AIAC + ISAC scenarios (1250 samples)
│
├── reports/                      # SHAP analysis visualizations
│   ├── shap_summary_fl.png       # Global feature importance
│   └── shap_local_bar_fl.png     # Local explanation bar chart
│
├── logs/                         # Runtime logs (drift, SHAP, FL)
├── notebooks/                    # Jupyter notebooks for exploration
├── Dockerfile.backend            # Backend container (Python 3.10)
├── docker-compose.yml            # Full-stack orchestration
├── requirements-backend.txt      # Backend Python dependencies
├── requirements-dev.txt          # Development dependencies
└── .github/workflows/
    └── docker-ci.yml             # GitHub Actions CI pipeline
```

---

## 🚀 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed & running
- Git
- Python 3.10+ (for local development)
- Node.js 18+ (for frontend development)

### Run with Docker Compose

```bash
# Clone the repository
git clone https://github.com/Hraj24/NETRA-2.0-trustworthy-secure-nwdaf-mlops.git
cd NETRA-2.0-trustworthy-secure-nwdaf-mlops

# Build and start all services
docker compose up --build -d
```

| Service | URL | Description |
|---------|-----|-------------|
| **Backend API** | http://localhost:8000 | FastAPI with Swagger docs at `/docs` |
| **Dashboard** | http://localhost:3000 | React frontend (IMT-2030 UI) |

### Local Development (without Docker)

```bash
# Backend setup
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac
pip install -r requirements-backend.txt

# Start backend
uvicorn src.api:app --reload --port 8000

# Frontend setup
cd nwdaf-dashboard
npm install
npm run dev
```

---

## 🧠 IMT-2030 (6G) Compliance

This platform is aligned with **ITU-R Recommendation M.2160** (IMT-2030 Framework) and supports all 6 usage scenarios:

| Scenario | Description | Evolution From |
|----------|-------------|----------------|
| **IC** | Immersive Communication | 5G eMBB |
| **HRLLC** | Hyper Reliable & Low Latency Communication | 5G URLLC |
| **MC** | Massive Communication | 5G mMTC |
| **UC** | Ubiquitous Connectivity | NEW in 6G |
| **AIAC** | AI and Communication | NEW in 6G |
| **ISAC** | Integrated Sensing and Communication | NEW in 6G |

### 12 IMT-2030 Features

The model uses 12 features for prediction:

1. `time_of_day` — Hour of day (0-23)
2. `usage_scenario` — 6G scenario type (IC/HRLLC/MC/UC/AIAC/ISAC)
3. `throughput_mbps` — User experienced data rate (Mbps)
4. `latency_ms` — User plane latency (ms)
5. `jitter_ms` — Jitter variation (ms)
6. `packet_loss_rate` — Packet loss probability (decimal)
7. `reliability_target` — Reliability target (%)
8. `connection_density_km2` — Devices per km²
9. `mobility_kmph` — User mobility speed (km/h)
10. `area_traffic_capacity_score` — Normalized area traffic capacity (0-1)
11. `ai_load_score` — AI workload demand (0-1)
12. `resilience_score` — Resilience indicator (0-1)

---

## 🔌 API Endpoints

### `POST /predict`

Predict future network traffic load with IMT-2030 KPIs.

**Request Body (IMT-2030 schema):**
```json
{
  "time_of_day": 12,
  "usage_scenario": "IC",
  "throughput_mbps": 400.0,
  "latency_ms": 3.0,
  "jitter_ms": 1.5,
  "packet_loss_rate": 0.0001,
  "reliability_target": 99.999,
  "connection_density_km2": 500000,
  "mobility_kmph": 100,
  "area_traffic_capacity_score": 0.35,
  "ai_load_score": 0.35,
  "resilience_score": 0.75
}
```

**Response:**
```json
{
  "predicted_future_load": 275.4,
  "explanation": "Network conditions appear stable with moderate predicted load.",
  "model_version": "FL-v2.0-IMT2030 (normal)",
  "warning": null
}
```

### `POST /explain`

Get SHAP-based feature attributions for a prediction.

**Response:**
```json
{
  "prediction": 275.4,
  "shap_values": {
    "time_of_day": 0.12,
    "usage_scenario": -0.05,
    "throughput_mbps": 1.85,
    "latency_ms": -0.23,
    ...
  },
  "note": "SHAP values indicate per-feature contribution to the prediction"
}
```

### `GET /health`

System health, model version, drift detector status, and rollback state.

---

## 🧪 Data Generation

Generate IMT-2030 compliant synthetic datasets:

```bash
# Generate full dataset with 5000 samples
python -m src.data_pipeline --generate --samples 5000 --seed 42

# Validate existing dataset
python -m src.data_pipeline --validate --input data/traffic_synthetic.csv

# Generate domain-specific datasets only
python -m src.data_pipeline --domains --samples 2500 --seed 42
```

Output files:
- `data/traffic_synthetic.csv` — Full combined dataset
- `data/domain_a.csv` — IC + HRLLC scenarios
- `data/domain_b.csv` — MC + UC scenarios
- `data/domain_c.csv` — AIAC + ISAC scenarios

---

## 🧪 Federated Learning Training

```bash
# Terminal 1 — Start FL server
python -m src.fl_server

# Terminals 2, 3, 4 — Start FL clients (one per domain)
python -m src.fl_client --domain domain_a
python -m src.fl_client --domain domain_b
python -m src.fl_client --domain domain_c
```

The aggregated global model is saved to `models/fl_global_model.pkl`.

---

## 🔑 Key Components

### Hybrid Drift Detection

Three detectors run in parallel on every prediction:

| Detector | Method | Role |
|----------|--------|------|
| **Moving Average** | Sliding-window mean comparison | Primary — triggers rollback |
| **ADWIN** | Adaptive windowing (statistical) | Early warning system |
| **DDM** | Drift Detection Method (error-rate) | Early warning system |

### Model Rollback & Auto-Recovery

- On **drift detected** (MA) → automatic rollback to the stable model
- After **5 consecutive stable windows** → auto-recovery to the primary model
- All events are audit-logged to `drift_events.csv`

### Federated Learning

- **Framework:** Flower (FedAvg strategy)
- **Clients:** 3 domain-specific clients (domain_a, domain_b, domain_c)
- **Model:** SGDRegressor with 12 IMT-2030 features
- **Privacy:** Raw data never leaves the client

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10, FastAPI, Uvicorn, Pydantic |
| **ML/AI** | scikit-learn, SHAP, NumPy, Pandas |
| **Drift Detection** | River (ADWIN, DDM), Custom Moving Average |
| **Federated Learning** | Flower (flwr) |
| **Frontend** | React 18, Vite, Recharts |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Model Export** | ONNX, joblib |

---

## 📊 Reports

Pre-generated SHAP analysis visualizations are available in the `reports/` directory:

- `shap_summary_fl.png` — Global feature importance (12 IMT-2030 features)
- `shap_local_bar_fl.png` — Local explanation bar chart

---

## 📜 License

This project is developed as part of academic research on trustworthy AI/ML operations in 6G network analytics.

**Standards Compliance:**
- ITU-R Recommendation M.2160 (September 2023)
- ITU-R IMT-2030 Technical Performance Requirements (February 2026)

---

<p align="center">
  Built with ❤️ for trustworthy 6G network intelligence
</p>
