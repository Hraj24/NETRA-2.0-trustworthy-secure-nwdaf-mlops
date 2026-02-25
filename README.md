<p align="center">
  <h1 align="center">🛡️ NETRA 1.0</h1>
  <p align="center"><strong>Trustworthy & Secure NWDAF MLOps Platform</strong></p>
  <p align="center">
    Federated Learning · Hybrid Drift Detection · SHAP Explainability · Auto-Rollback & Recovery
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

**NETRA** (Network Traffic Risk Analyzer) is a trustworthy MLOps platform built around the 3GPP **NWDAF** (Network Data Analytics Function) framework. It delivers real-time network traffic load predictions with built-in safeguards for production ML systems:

- **Federated Learning** — privacy-preserving model training across distributed network domains using Flower (FedAvg)
- **Hybrid Drift Detection** — three concurrent detectors (Moving Average, ADWIN, DDM) monitor SLA metrics for concept drift
- **Automatic Rollback & Recovery** — seamless fallback to a stable model on drift, with auto-recovery after stabilization
- **SHAP Explainability** — per-prediction feature attribution using SHAP linear explainer for full transparency
- **Interactive Dashboard** — React-based UI for predictions, drift warnings, and SHAP visualizations

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    NETRA 1.0 Platform                    │
├────────────────────────┬─────────────────────────────────┤
│   React Dashboard      │     FastAPI Backend             │
│   (Port 3000)          │     (Port 8000)                 │
│                        │                                 │
│  ┌──────────────┐      │  ┌───────────┐ ┌────────────┐  │
│  │ Input Form   │──────┼─▶│ /predict  │ │ /explain   │  │
│  │ Predict View │◀─────┼──│ /health   │ │            │  │
│  │ SHAP Panel   │      │  └─────┬─────┘ └─────┬──────┘  │
│  │ Status Panel │      │        │              │         │
│  └──────────────┘      │  ┌─────▼──────────────▼──────┐  │
│                        │  │    Model Manager           │  │
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
│                        │  │  trained via Flower FedAvg │  │
│                        │  └───────────────────────────┘  │
└────────────────────────┴─────────────────────────────────┘
```

---

## 📂 Project Structure

```
NETRA-1.0/
├── src/                          # Python backend source
│   ├── api.py                    # FastAPI app — /predict, /explain, /health
│   ├── drift.py                  # Moving Average drift detector
│   ├── drift_adaptive.py         # ADWIN & DDM drift detectors (river)
│   ├── drift_logger.py           # CSV-based drift event auditing
│   ├── rollback.py               # ModelManager — rollback & auto-recovery
│   ├── explain.py                # SHAP explanation utilities
│   ├── shap_logger.py            # SHAP value logging for audit
│   ├── fl_server.py              # Flower FL server (FedAvg + model save)
│   ├── fl_client.py              # Flower FL client (per-domain training)
│   ├── data_pipeline.py          # Data loading & preprocessing
│   ├── train_centralized.py      # Centralized training baseline
│   ├── train_and_save_models.py  # Model training & serialization
│   ├── package_model.py          # ONNX model packaging
│   ├── plot_drift.py             # Drift visualization
│   └── plot_drift_comparison.py  # Multi-detector comparison plots
│
├── nwdaf-dashboard/              # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx               # Main app with tabbed layout
│   │   ├── api.js                # Backend API client
│   │   └── components/
│   │       ├── InputForm.jsx     # Traffic sample input form
│   │       ├── PredictionPanel.jsx
│   │       ├── ExplainPanel.jsx  # SHAP visualization
│   │       └── StatusPanel.jsx   # System health & drift status
│   └── Dockerfile                # Multi-stage nginx build
│
├── models/                       # Trained model artifacts
│   ├── fl_global_model.pkl       # Federated Learning global model
│   ├── model_v1.pkl              # Stable baseline model
│   ├── model_v2_bad.pkl          # Intentionally degraded (for testing)
│   ├── nwdaf_rf.onnx             # ONNX-exported model
│   └── nwdaf_rf_meta.json        # Model metadata
│
├── data/                         # Synthetic network traffic datasets
│   ├── traffic_synthetic.csv
│   ├── domain_a.csv
│   ├── domain_b.csv
│   └── domain_c.csv
│
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

### Run with Docker Compose

```bash
# Clone the repository
git clone https://github.com/Hraj24/NETRA-1.0-trustworthy-secure-nwdaf-mlops.git
cd NETRA-1.0-trustworthy-secure-nwdaf-mlops

# Build and start all services
docker compose up --build -d
```

| Service | URL | Description |
|---------|-----|-------------|
| **Backend API** | http://localhost:8000 | FastAPI with Swagger docs at `/docs` |
| **Dashboard** | http://localhost:3000 | React frontend |

### Local Development (without Docker)

```bash
# Backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-backend.txt
uvicorn src.api:app --reload --port 8000

# Frontend
cd nwdaf-dashboard
npm install
npm run dev
```

---

## 🔌 API Endpoints

### `POST /predict`

Predict future network traffic load with drift-aware inference.

```json
{
  "time_of_day": 14,
  "slice_type": "eMBB",
  "jitter": 12.5,
  "packet_loss": 1.2,
  "throughput": 450.0
}
```

**Response:**
```json
{
  "predicted_future_load": 327.8,
  "explanation": "Network conditions appear stable with moderate predicted load.",
  "model_version": "FL-v1.0 (normal)",
  "warning": null
}
```

### `POST /explain`

Get SHAP-based feature attributions for a prediction.

### `GET /health`

System health, model version, drift detector status, and rollback state.

---

## 🧠 Key Components

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
- **Model:** SGDRegressor aggregated across 5 rounds
- **Privacy:** Raw data never leaves the client

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

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10, FastAPI, Uvicorn |
| **ML/AI** | scikit-learn, SHAP, NumPy, Pandas |
| **Drift Detection** | River (ADWIN, DDM), Custom MA |
| **Federated Learning** | Flower (flwr) |
| **Frontend** | React 18, Vite |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Model Export** | ONNX, joblib |

---

## 📊 Reports

Pre-generated SHAP analysis visualizations are available in the `reports/` directory:

- `shap_summary.png` — Global feature importance
- `shap_summary_fl.png` — FL model SHAP summary
- `shap_local_bar_fl.png` — Local explanation bar chart

---

## 📜 License

This project is developed as part of academic research on trustworthy AI/ML operations in 5G network analytics.

---

<p align="center">
  Built with ❤️ for trustworthy 5G network intelligence
</p>
