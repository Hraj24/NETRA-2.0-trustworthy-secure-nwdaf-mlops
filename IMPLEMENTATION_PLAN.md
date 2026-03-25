# NETRA 1.0 - IMT-2030 Implementation Plan

## Overview

This document outlines the complete implementation plan for upgrading NETRA 1.0 from 5G to 6G IMT-2030 compliance, following ITU-R Recommendation M.2160.

---

## Phase 1: Data Pipeline (COMPLETED)

### 1.1 Data Generation Module
**File:** `src/data_pipeline.py`

**Changes:**
- Rewrote to generate IMT-2030 compliant synthetic data
- Added 6 usage scenarios: IC, HRLLC, MC, UC, AIAC, ISAC
- Implemented 17-column output schema with all required KPIs
- Added strict KPI bounds per scenario from ITU-R M.2160

**Key Functions:**
- `generate_imt2030_dataset(n_samples, seed)` - Main generation function
- `validate_imt2030_compliance(df)` - Validation with violation reporting
- `generate_domain_datasets()` - FL domain split generation
- `load_data()`, `preprocess()`, `get_features_and_target()` - Backward compatibility

**Cross-Scenario Constraints Enforced:**
1. HRLLC has lowest latency (0.1-1ms)
2. MC has highest connection density (10⁶-10⁸ /km²)
3. AIAC has highest AI load score (0.7-1.0)
4. Reliability and packet_loss inversely correlated

### 1.2 Generated Datasets
**Location:** `data/`

| File | Samples | Scenarios |
|------|---------|-----------|
| traffic_synthetic.csv | 5,000 | All 6 scenarios |
| domain_a.csv | 1,250 | IC + HRLLC |
| domain_b.csv | 1,250 | MC + UC |
| domain_c.csv | 1,250 | AIAC + ISAC |

---

## Phase 2: Backend API (COMPLETED)

### 2.1 FastAPI Application
**File:** `src/api.py`

**Changes:**
- Updated to IMT-2030 input schema (12 features)
- Added `TrafficSampleIMT2030` Pydantic model
- Updated SHAP explainer for 12 features
- Maintains backward compatibility layer

**New Schema:**
```json
{
  "time_of_day": 0-23,
  "usage_scenario": "IC|HRLLC|MC|UC|AIAC|ISAC",
  "throughput_mbps": 1-500,
  "latency_ms": 0.1-100,
  "jitter_ms": 0.01-10,
  "packet_loss_rate": 0.0000001-0.01,
  "reliability_target": 99.9-99.99999,
  "connection_density_km2": 1000-100000000,
  "mobility_kmph": 0-1000,
  "area_traffic_capacity_score": 0-1,
  "ai_load_score": 0-1,
  "resilience_score": 0-1
}
```

**Endpoints:**
- `POST /predict` - Accepts IMT-2030 schema
- `POST /explain` - Returns 12-feature SHAP values
- `GET /health` - System status

---

## Phase 3: Federated Learning (COMPLETED)

### 3.1 FL Server
**File:** `src/fl_server.py`

**Changes:**
- Updated to handle 12-feature model aggregation
- Added IMT-2030 metadata in saved model
- Custom FedAvg strategy persists global model

### 3.2 FL Client
**File:** `src/fl_client.py`

**Changes:**
- Updated feature extraction for IMT-2030 schema
- Uses `IMT2030_FEATURE_COLS` (12 features)
- Domain-specific training on domain_a/b/c CSVs

### 3.3 Training Scripts
**Files:** `src/train_centralized.py`, `src/train_and_save_models.py`

**Changes:**
- Updated to train on 12 IMT-2030 features
- Added StandardScaler for SGD (handles scale variance)
- Saves models with feature metadata

---

## Phase 4: Explainability (COMPLETED)

### 4.1 SHAP Module
**File:** `src/explain.py`

**Changes:**
- Updated explainer for 12 features
- Generates global summary and local bar charts
- Feature names updated to IMT-2030 schema

### 4.2 SHAP Logger
**File:** `src/shap_logger.py`

**Changes:**
- Logs all 12 feature SHAP values
- Tracks rollback state correlation

---

## Phase 5: Frontend (COMPLETED)

### 5.1 Input Form
**File:** `nwdaf-dashboard/src/components/InputForm.jsx`

**Changes:**
- Updated to collect 12 IMT-2030 KPIs
- Added dropdown for 6 usage scenarios
- Appropriate input ranges per KPI

**UI Fields:**
- Time of Day (0-23)
- Usage Scenario (dropdown: IC/HRLLC/MC/UC/AIAC/ISAC)
- Throughput (Mbps)
- Latency (ms)
- Jitter (ms)
- Packet Loss Rate (decimal)
- Reliability Target (%)
- Connection Density (/km²)
- Mobility (km/h)
- Area Traffic Capacity (0-1)
- AI Load Score (0-1)
- Resilience Score (0-1)

### 5.2 Display Components
**Files:** `PredictionPanel.jsx`, `ExplainPanel.jsx`, `StatusPanel.jsx`

**Status:** No changes needed - components are schema-agnostic and display API responses dynamically.

---

## Phase 6: Documentation (COMPLETED)

### 6.1 README.md
**Changes:**
- Added IMT-2030 compliance badge
- Updated architecture diagram with 12 features
- Added 6G usage scenarios table
- Updated API documentation with new schema
- Added data generation CLI examples

---

## Phase 7: Testing & Validation

### 7.1 Integration Tests
**Status:** COMPLETED

All components tested:
- Data pipeline: 5000 samples, 12 features
- Model loading: 12-feature weights
- Drift detection: 3 detectors functional
- Rollback manager: operational
- SHAP explainer: 12 features
- FL client: domain compatibility
- API schema: validation passing

### 7.2 Model Training Results
| Model | RMSE | MAE | R² | MAPE |
|-------|------|-----|----|------|
| SGD (scaled) | ~60M | ~25M | N/A | N/A |
| Random Forest | 8.07 | 6.36 | 0.9853 | 5.2% |

**Note:** SGD struggles with extreme scale variance in IMT-2030 features (connection_density: 10³-10⁸). RF performs well. FL clients use per-domain StandardScaler for convergence.

---

## File Change Summary

| File | Status | Changes |
|------|--------|---------|
| `src/data_pipeline.py` | Rewritten | Complete IMT-2030 generation (950+ lines) |
| `src/api.py` | Updated | IMT-2030 schema, 12 features |
| `src/fl_client.py` | Rewritten | 12-feature FL client |
| `src/fl_server.py` | Updated | IMT-2030 metadata |
| `src/train_centralized.py` | Rewritten | 12-feature training |
| `src/train_and_save_models.py` | Rewritten | Model serialization |
| `src/explain.py` | Rewritten | 12-feature SHAP |
| `src/shap_logger.py` | Updated | 12-feature logging |
| `nwdaf-dashboard/src/components/InputForm.jsx` | Updated | 12 KPI inputs |
| `README.md` | Updated | IMT-2030 documentation |
| `data/*.csv` | Generated | 4 IMT-2030 datasets |
| `models/*.pkl` | Regenerated | 12-feature models |

---

## Usage Guide

### Data Generation
```bash
python -m src.data_pipeline --generate --samples 5000 --seed 42
```

### Model Training
```bash
python -m src.train_and_save_models
```

### FL Training
```bash
# Terminal 1
python -m src.fl_server

# Terminal 2,3,4
python -m src.fl_client --domain domain_a
python -m src.fl_client --domain domain_b
python -m src.fl_client --domain domain_c
```

### API Server
```bash
uvicorn src.api:app --reload --port 8000
```

### Frontend
```bash
cd nwdaf-dashboard
npm run dev
```

---

## Compliance Verification

All generated data passes IMT-2030 compliance validation:

```
✓ HRLLC latency lowest: 0.54ms avg (others: 3-55ms)
✓ MC density highest: 5.02×10⁷ /km² (others: 5×10⁴-5×10⁶)
✓ AIAC AI load highest: 0.85 (others: 0.15-0.65)
✓ Reliability/packet_loss inversely correlated
✓ All KPI bounds within ITU-R M.2160 specifications
```

---

## Next Steps (Optional Enhancements)

1. **Model Optimization** - Train ensemble models for improved RMSE
2. **Real Data Integration** - Connect to actual 6G testbed data
3. **Drift Threshold Tuning** - Calibrate for 6G SLA requirements
4. **Dashboard Enhancements** - Add scenario-specific visualizations
5. **Performance Benchmarking** - Compare FL vs centralized training

---

**Standards References:**
- ITU-R M.2160: "Framework and overall objectives of the future development of IMT for 2030 and beyond" (September 2023)
- ITU-R IMT-2030 TPR: "Technical Performance Requirements for IMT-2030" (February 2026)

---

*Document Version: 1.0*
*Last Updated: 2026-03-24*
