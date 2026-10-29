# AI-Based Traffic Prediction System with MLOps

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioning-945DD6.svg?logo=dvc&logoColor=white)](https://dvc.org/)
[![Evidently AI](https://img.shields.io/badge/Evidently_AI-Drift_Monitoring-purple.svg)](https://evidentlyai.com/)
[![Native Execution](https://img.shields.io/badge/Deployment-Native_Windows_Python-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-19%20Passed-brightgreen.svg)]()

A production-grade, reproducible Machine Learning Operations (MLOps) system for hourly vehicular traffic volume forecasting across multi-junction road networks. Built with Deep LSTM networks, FastAPI model serving, a Streamlit interactive dashboard, MLflow experiment tracking, DVC pipeline versioning, Evidently AI drift monitoring, and a safe candidate-model retraining workflow.

> **Execution Environment Note**: This project runs **natively on Windows** using Python 3.11, a standard virtual environment (`.venv`), FastAPI, and Streamlit. **Docker is completely removed and is not used in this final configuration.**

---

## 📌 Project Overview

Urban vehicular congestion causes substantial delays, elevated carbon emissions, and fuel wastage. Accurate short-term traffic volume forecasting empowers municipal traffic authorities and Intelligent Transportation Systems (ITS) to dynamically optimize signal timings, anticipate bottlenecks, and route emergency services.

### Key Capabilities (Phase 1 & Phase 2 Complete)
1. **Time-Series Deep Learning**: 2-layer stacked Long Short-Term Memory (LSTM) network with Dropout regularization and Dense output.
2. **Strict Chronological Sequencing**: 24-step sliding window ($T=24$ hours lookback) with zero future data leakage.
3. **RESTful Model Serving**: High-throughput **FastAPI** service with `/health`, `/model-info`, `/junctions`, and `/predict` endpoints, complete with OpenAPI/Swagger documentation.
4. **Interactive Dashboard**: **Streamlit** web interface supporting historical holdout evaluation, recursive multi-step future forecasting, session query history with CSV export, and architecture schemas.
5. **Experiment Tracking**: **MLflow** recording parameters, epoch loss curves, evaluation metrics, and model artifacts in SQLite (`mlflow.db`) and local artifact store.
6. **Data & Pipeline Versioning**: **DVC** tracking raw data and reproducible DAG pipeline stages (`prepare` $\rightarrow$ `train` $\rightarrow$ `evaluate` $\rightarrow$ `monitor`).
7. **Production Data Drift Monitoring**: **Evidently AI** comparing reference training distributions with incoming test horizons and tracking model performance drift.
8. **Safe Candidate Retraining**: Controlled retraining workflow comparing candidate RMSE/$R^2$ against production baseline with automatic promotion gates, atomic backups, and rollback protection.
9. **Automated Unit & Integration Testing**: Comprehensive **pytest** suite validating schemas, endpoints, chronological isolation, and retraining policies (19 passing tests).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data & Pipeline Versioning [DVC & Storage]
        Raw["Raw Data (data/raw/traffic.csv)"] -->|DVC Tracking| Prep["Stage 1: Prepare (src/data_preprocessing.py)"]
        Prep --> Processed["Processed Sequences (data/processed/traffic_sequences.npz)"]
        Prep --> Scalers["MinMax Scalers (models/*_scaler.pkl)"]
        Processed --> Train["Stage 2: Train (src/train_lstm.py)"]
        Train --> Model["Production Model (models/traffic_lstm.keras)"]
        Model --> Eval["Stage 3: Evaluate (src/evaluate.py)"]
        Eval --> Figures["Reports & Figures (reports/figures/)"]
        Model --> Monitor["Stage 4: Monitor (monitoring/drift_report.py)"]
    end

    subgraph Experiment Tracking [MLflow Tracking Server]
        Train -.->|Log Params, Loss, Weights| MLflow["MLflow UI (traffic-prediction-lstm)"]
        Eval -.->|Log Test Metrics & Plots| MLflow
        Retrain["Retraining Pipeline (src/retrain.py)"] -.->|Candidate Comparison & Audit| MLflow
    end

    subgraph Model Serving & Consumption [Native Windows Execution]
        Model --> API["FastAPI Service (api/main.py :8000)"]
        Scalers --> API
        API --> Dashboard["Streamlit Dashboard (dashboard/app.py :8501)"]
        API --> Swagger["Interactive Swagger UI (/docs)"]
    end

    subgraph Monitoring & Governance [Evidently AI]
        Processed --> Monitor
        Monitor --> DriftReport["Drift HTML & Summary JSON (reports/)"]
    end
```

---

## 📊 Dataset & Feature Engineering

- **Dataset**: Kaggle Traffic Prediction Dataset (Hourly Vehicle Counts Across 4 Junctions)
- **Source**: [https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset)
- **Observations**: `48,120` rows $\times$ `4` raw columns (`DateTime`, `Junction`, `Vehicles`, `ID`)
- **Timestamp Coverage**: `2015-11-01 00:00:00` to `2017-06-30 23:00:00`
- **Target Variable**: `Vehicles` (hourly count)

### Input Feature Matrix ($T=24$ Steps $\times 6$ Features)
The six features fed into the LSTM model are ordered as:
| Index | Feature Name | Type | Description |
| :---: | :--- | :--- | :--- |
| 0 | **`Vehicles`** | Float | Scaled vehicle count from preceding observation |
| 1 | **`Hour`** | Integer ($0$–$23$) | Diurnal cycle (morning/evening rush hour identifier) |
| 2 | **`DayOfWeek`** | Integer ($0$–$6$) | Day of week ($0 = \text{Monday}, \dots, 6 = \text{Sunday}$) |
| 3 | **`Month`** | Integer ($1$–$12$) | Seasonal calendar month |
| 4 | **`IsWeekend`** | Binary ($0/1$) | Weekend traffic shift indicator |
| 5 | **`Junction`** | Integer ($1$–$4$) | Specific junction identifier |

### Chronological Holdout Split (Strictly No Random Shuffling)
- **Training Set (70.0%)**: `33,684` samples (`2015-11-01 00:00:00` to `2017-01-31 14:00:00`)
- **Validation Set (15.0%)**: `7,218` samples (`2017-01-31 15:00:00` to `2017-04-16 19:00:00`)
- **Testing Set (15.0%)**: `7,218` samples (`2017-04-16 19:00:00` to `2017-06-30 23:00:00`)

---

## 🧠 LSTM Neural Network Architecture

```text
Input Layer (Shape: [Batch Size, 24 Steps, 6 Features])
       │
       ▼
LSTM Layer 1 (64 Units, return_sequences=True, name="lstm_layer_1")
       │
       ▼
Dropout Layer (Rate = 0.20, name="dropout_1")
       │
       ▼
LSTM Layer 2 (32 Units, return_sequences=False, name="lstm_layer_2")
       │
       ▼
Dense Intermediate (16 Units, Activation="relu", name="dense_intermediate")
       │
       ▼
Dense Output Layer (1 Unit, Activation="linear", name="traffic_prediction_output")
```

---

## 🔍 Metric Verification & Discrepancy Forensic Analysis

During development, two sets of metrics were recorded:
1. **Initial Phase-1 Baseline Report**:
   - `MAE: 6.1383`, `MSE: 67.4930`, `RMSE: 8.2154`, `R²: 0.9137`
2. **Monitoring & Saved Checkpoint Evaluation**:
   - `MAE: 5.6265`, `MSE: 54.0443`, `RMSE: 7.3515`, `R²: 0.9309`

### Root Cause Audit
An audit of `mlflow.db` and the saved model artifacts confirmed:
- **Identical Evaluation Horizon**: Both evaluations evaluated the exact same 15% chronological holdout test set (7,194 sequences of window length 24, from April 16, 2017 to June 30, 2017).
- **Identical Feature Set & Unscaled Units**: Both computed metrics in real vehicle count units (inverting `models/target_scaler.pkl`).
- **Reason for Difference**:
  - Run `f0bd662c` (Sept 28, 2026, 12:09 PM) stopped at early-stopping epoch 19 (best epoch 14) with validation loss `0.002852`, producing the initial Phase-1 baseline (`MAE: 6.14`, `RMSE: 8.22`).
  - Run `0436fa4b` (Sept 29, 2026, 11:31 AM) completed an extended 25-epoch training run reaching lower validation loss `0.002252`. This superior checkpoint was saved to `models/traffic_lstm.keras`.
  - When the saved production model `models/traffic_lstm.keras` is evaluated via `src/evaluate.py` and `monitoring/drift_report.py`, it achieves `MAE: 5.6265` and `RMSE: 7.3515`.
  - **Conclusion**: Both numbers are genuine and valid. The discrepancy reflects convergence gain from extended training of the production checkpoint. The historical baseline (`MAE: 6.14, RMSE: 8.22`) is preserved as the contractual acceptance standard for retraining candidates.

---

## ⚡ Quickstart: Native Windows Setup

### 1. Prerequisites
- Windows 10/11
- Python 3.11 installed (available via `python --version`)
- PowerShell

### 2. Environment Activation & Dependencies
```powershell
# Navigate to project directory
cd "c:\Users\gowri\OneDrive\Desktop\mlops project"

# Activate existing Python 3.11 virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional) Verify or install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the template environment file:
```powershell
Copy-Item .env.example .env
```

---

## 🚀 Running the Services Locally

### Step 1 — Start the FastAPI Serving Backend
In your primary terminal:
```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **ReDoc Schema Documentation**: `http://127.0.0.1:8000/redoc`

#### Available Endpoints
| HTTP Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/health` | Verifies model, scalers, and historical dataset status |
| `GET` | `/model-info` | Returns model architecture, input feature schema, and baseline metrics |
| `GET` | `/junctions` | Returns record counts, date ranges, and volume statistics per junction |
| `POST` | `/predict` | Generates hourly traffic predictions (historical, future, or custom sequence) |

#### Prediction Payload Examples:
**Historical Holdout Evaluation (Zero Leakage)**:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"target_timestamp": "2017-06-30 23:00:00", "junction": 1, "mode": "historical"}' | ConvertTo-Json
```
*Output*:
```json
{
  "target_timestamp": "2017-06-30 23:00:00",
  "junction": 1,
  "predicted_vehicles": 53.57,
  "mode": "historical_evaluation",
  "model_version": "traffic_lstm_v1",
  "actual_vehicles": 78.0,
  "residual_error": -24.43,
  "sequence_start": "2017-06-29 23:00:00",
  "sequence_end": "2017-06-30 22:00:00"
}
```

**Future Recursive Forecast (Multi-Step)**:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"target_timestamp": "2017-07-01 06:00:00", "junction": 1, "mode": "future"}' | ConvertTo-Json
```

---

### Step 2 — Start the Streamlit Dashboard
Open a second PowerShell terminal, activate `.venv`, and run:
```powershell
streamlit run dashboard/app.py
```
- The dashboard opens at: **`http://localhost:8501`**
- It connects directly to the FastAPI service via `API_BASE_URL` (`http://127.0.0.1:8000`).
- **Features**:
  - **Live Prediction Tab**: Select junction (1–4), mode (Historical or Future Forecast), adjust target date/hour, and inspect predicted vs. actual traffic counts.
  - **Baseline Evaluation Tab**: View verified holdout metrics and test horizon charts (`actual_vs_predicted.png`).
  - **Session History Tab**: Track all queries executed during the session with single-click CSV export.
  - **Model & Architecture Tab**: Inspect input feature ordering and the 2-layer LSTM network specifications.

---

### Step 3 — Run Evidently AI Drift Monitoring
```powershell
python monitoring/drift_report.py
```
- Generates interactive HTML report: [`reports/evidently_drift_report.html`](reports/evidently_drift_report.html)
- Generates JSON summary: [`reports/drift_summary.json`](reports/drift_summary.json)
- Evaluates data distribution drift between the 70% training reference partition and the 15% holdout test partition, alongside performance evaluation.

---

### Step 4 — Run Safe Candidate Model Retraining
```powershell
python src/retrain.py --epochs 15 --batch-size 64 --tolerance 0.05
```
- **Retraining Governance**:
  1. Computes current production baseline metrics on the test set.
  2. Trains a candidate model saved separately to `models/candidate_traffic_lstm.keras`.
  3. Executes an inference sanity smoke test (checks non-NaN outputs).
  4. Enforces promotion rules: Candidate must achieve $R^2 \ge 0.85$ and $\text{RMSE} \le \text{baseline} \times 1.05$.
  5. If approved: Backs up the current production model to `models/traffic_lstm_backup.keras` and promotes the candidate.
  6. If rejected: Preserves the production model and records rejection in MLflow.

---

### Step 5 — Run MLflow Experiment Tracking UI
```powershell
python -m mlflow ui --port 5000
```
- Open browser at: **`http://127.0.0.1:5000`**
- Inspect runs, hyperparameters, training loss curves, evaluation metrics, and candidate retraining runs under experiment **`traffic-prediction-lstm`**.

---

### Step 6 — Run the DVC Pipeline
To reproduce the 4-stage pipeline:
```powershell
dvc repro
```
To check stage dependencies and outputs:
```powershell
dvc status
```

---

## 🧪 Automated Testing Suite

Run the full pytest suite:
```powershell
python -m pytest -v
```

### Verified Test Results (19/19 Passed):
| Test Function | Component Verified | Status |
| :--- | :--- | :---: |
| `test_health_endpoint` | Verifies `/health` returns status, model, scalers, records | **PASSED** |
| `test_model_info_endpoint` | Verifies `/model-info` schema, feature order, baseline metrics | **PASSED** |
| `test_predict_historical_valid` | Verifies historical prediction, residual calculation | **PASSED** |
| `test_predict_future_valid` | Verifies multi-step recursive forecasting mode | **PASSED** |
| `test_predict_custom_sequence_valid` | Verifies 24-step client-supplied sequence inference | **PASSED** |
| `test_predict_invalid_junction` | Verifies rejection of invalid junction IDs (e.g. 99) | **PASSED** |
| `test_predict_invalid_sequence_length` | Verifies rejection of sequences $\ne 24$ steps | **PASSED** |
| `test_predict_invalid_timestamp_format` | Verifies rejection of malformed timestamp strings | **PASSED** |
| `test_junctions_endpoint` | Verifies `/junctions` returns data for all 4 junctions | **PASSED** |
| `test_predict_negative_vehicle_count` | Verifies rejection of negative vehicle values | **PASSED** |
| `test_predict_historical_no_future_leakage` | Verifies historical sequence strictly precedes target date | **PASSED** |
| `test_drift_report_generation` | Verifies HTML and JSON drift report creation | **PASSED** |
| `test_feature_engineering_columns` | Verifies calendar feature extraction | **PASSED** |
| `test_sliding_window_sequence_shape` | Verifies sliding window outputs shape $(N-L, L, F)$ | **PASSED** |
| `test_scaler_files_consistency` | Verifies scaler feature counts (6 features, 1 target) | **PASSED** |
| `test_promotion_rule_approval` | Verifies candidate promotion when metrics improve | **PASSED** |
| `test_promotion_rule_rejection_high_rmse` | Verifies candidate rejection when RMSE increases | **PASSED** |
| `test_promotion_rule_rejection_low_r2` | Verifies candidate rejection when $R^2 < 0.85$ | **PASSED** |
| `test_model_preservation_on_candidate_failure` | Verifies production model is preserved on rejection | **PASSED** |

---

## ⚠️ Model Assumptions & Limitations

1. **Continuous 24-Hour Lookback Requirement**: The LSTM model requires exactly 24 preceding continuous hourly observations. It cannot forecast an isolated time point without its historical context.
2. **Recursive Error Compounding**: In multi-step future forecasting, prior predictions are fed back into subsequent inputs. Errors compound over time; forecasts beyond 72 hours are blocked by policy.
3. **Absence of Real-Time IoT Sensors**: The dataset consists of historical recordings across 4 junctions. It does not ingest live telemetry or weather feeds; sudden disruptions (road construction, accidents) cannot be anticipated without dedicated external sensors.
4. **Historical Evaluation vs. Real-Time Telemetry**: Historical predictions evaluate against known holdout ground truth for validation and do not represent live IoT sensor feeds.

---

## 🎓 College Viva & Evaluation Reference

### Q1: What was added in Phase 2 compared to Phase 1?
> **Answer**: Phase 1 established the core machine learning pipeline (EDA, preprocessing, LSTM training, evaluation, initial MLflow, and DVC). Phase 2 operationalized the model by delivering:
> 1. A FastAPI REST serving API (`/health`, `/model-info`, `/junctions`, `/predict`)
> 2. An interactive Streamlit web dashboard with historical evaluation and recursive forecasting
> 3. Evidently AI data drift and performance monitoring with HTML & JSON reporting
> 4. A safe candidate retraining pipeline with promotion gates and rollback backups
> 5. A 19-test automated pytest verification suite
> All running natively on Windows without container dependencies.

### Q2: How does the FastAPI service prevent data leakage during historical evaluation?
> **Answer**: When predicting traffic for timestamp $T$, the API filters observations strictly where $\text{DateTime} < T$ and selects the most recent 24 records. The target timestamp and any future timestamps are completely excluded from the input sequence.

### Q3: Why is candidate model promotion gated in retraining?
> **Answer**: Retraining does not automatically improve accuracy; fresh data can introduce noise or cause overfitting. The retraining workflow (`src/retrain.py`) verifies that candidate models pass a non-NaN smoke test, achieve $R^2 \ge 0.85$, and keep RMSE within 5% of the baseline before promotion. If rejected, the existing production model is preserved.

### Q4: What does Evidently AI measure?
> **Answer**: Evidently AI evaluates covariate drift across all 6 features between the 70% training reference dataset and incoming holdout batches using statistical tests (Kolmogorov-Smirnov test and Wasserstein distance). It also tracks model performance metrics (MAE, RMSE) over time when ground truth is available.
