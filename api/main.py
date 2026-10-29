import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import tensorflow as tf

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_lstm.keras")
FEATURE_SCALER_PATH = os.path.join(BASE_DIR, "models", "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(BASE_DIR, "models", "target_scaler.pkl")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "traffic.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_traffic.csv")

FEATURE_COLUMNS = ['Vehicles', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'Junction']
TARGET_COLUMN = 'Vehicles'
SEQUENCE_LENGTH = 24
VALID_JUNCTIONS = [1, 2, 3, 4]

# Baseline historical metrics from Phase 1 evaluation
BASELINE_METRICS = {
    "evaluation_split": "Test Set (15% chronological holdout, Apr 2017 - Jun 2017)",
    "sequence_length": 24,
    "mae": 6.1383,
    "mse": 67.4930,
    "rmse": 8.2154,
    "r2_score": 0.9137,
    "status": "Verified Phase 1 Baseline"
}

# Initialize FastAPI application
app = FastAPI(
    title="AI-Based Traffic Prediction API",
    description=(
        "Production-grade model serving API for hourly traffic volume forecasting across junctions "
        "using Deep Learning (LSTM networks), scikit-learn scalers, and chronological sequence handling."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory caches for model, scalers, and historical lookup dataframe
model_cache = None
feature_scaler_cache = None
target_scaler_cache = None
dataset_cache = None

def load_artifacts():
    """Load model, scalers, and historical data into memory safely."""
    global model_cache, feature_scaler_cache, target_scaler_cache, dataset_cache
    
    if os.path.exists(MODEL_PATH):
        try:
            model_cache = tf.keras.models.load_model(MODEL_PATH)
            print(f"[API] Loaded LSTM model from {MODEL_PATH}")
        except Exception as e:
            print(f"[API ERROR] Failed to load model: {e}")
            model_cache = None
    else:
        print(f"[API WARNING] Model path not found: {MODEL_PATH}")
        model_cache = None

    if os.path.exists(FEATURE_SCALER_PATH):
        try:
            feature_scaler_cache = joblib.load(FEATURE_SCALER_PATH)
            print(f"[API] Loaded feature scaler from {FEATURE_SCALER_PATH}")
        except Exception as e:
            print(f"[API ERROR] Failed to load feature scaler: {e}")
            feature_scaler_cache = None

    if os.path.exists(TARGET_SCALER_PATH):
        try:
            target_scaler_cache = joblib.load(TARGET_SCALER_PATH)
            print(f"[API] Loaded target scaler from {TARGET_SCALER_PATH}")
        except Exception as e:
            print(f"[API ERROR] Failed to load target scaler: {e}")
            target_scaler_cache = None

    # Load dataset for historical sequence lookups
    csv_path = PROCESSED_DATA_PATH if os.path.exists(PROCESSED_DATA_PATH) else RAW_DATA_PATH
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            # Ensure time features exist
            if 'Hour' not in df.columns:
                df['Hour'] = df['DateTime'].dt.hour
                df['DayOfWeek'] = df['DateTime'].dt.dayofweek
                df['Day'] = df['DateTime'].dt.day
                df['Month'] = df['DateTime'].dt.month
                df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
            # Sort chronologically by Junction and DateTime
            df = df.sort_values(by=['Junction', 'DateTime']).reset_index(drop=True)
            dataset_cache = df
            print(f"[API] Loaded historical dataset from {csv_path} with {len(df)} rows")
        except Exception as e:
            print(f"[API ERROR] Failed to load dataset: {e}")
            dataset_cache = None

# Load artifacts at startup
load_artifacts()


# -------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# -------------------------------------------------------------------------

class HistoricalSequenceStep(BaseModel):
    vehicles: float = Field(..., ge=0.0, description="Observed vehicle count (>= 0)")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: Optional[int] = Field(None, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    month: Optional[int] = Field(None, ge=1, le=12, description="Month (1-12)")
    is_weekend: Optional[int] = Field(None, ge=0, le=1, description="1 if Sat/Sun, else 0")
    junction: Optional[int] = Field(None, ge=1, le=4, description="Junction ID (1-4)")

class PredictRequest(BaseModel):
    target_timestamp: str = Field(
        ...,
        json_schema_extra={"example": "2017-06-30 23:00:00"},
        description="Target timestamp to forecast traffic for (YYYY-MM-DD HH:MM:SS or ISO format)"
    )
    junction: int = Field(
        ...,
        ge=1,
        le=4,
        json_schema_extra={"example": 1},
        description="Junction identifier (1 to 4)"
    )
    mode: Optional[str] = Field(
        "historical",
        description="Inference mode: 'historical' (lookup from dataset), 'future' (recursive rollout), or 'custom_sequence'"
    )
    custom_sequence: Optional[List[Union[float, HistoricalSequenceStep]]] = Field(
        None,
        description="Optional manual 24-step historical sequence (floats or feature dicts). If provided, overrides lookup."
    )

    @field_validator("target_timestamp")
    @classmethod
    def validate_timestamp_format(cls, value: str) -> str:
        try:
            pd.to_datetime(value)
            return value
        except Exception:
            raise ValueError(f"Invalid timestamp format '{value}'. Expected YYYY-MM-DD HH:MM:SS or ISO 8601.")

    @field_validator("custom_sequence")
    @classmethod
    def validate_sequence_length(cls, value):
        if value is not None and len(value) != SEQUENCE_LENGTH:
            raise ValueError(f"Custom sequence must contain exactly {SEQUENCE_LENGTH} observations. Received {len(value)}.")
        return value

class PredictResponse(BaseModel):
    target_timestamp: str
    junction: int
    predicted_vehicles: float
    mode: str
    model_version: str
    actual_vehicles: Optional[float] = None
    residual_error: Optional[float] = None
    note: Optional[str] = None
    sequence_start: Optional[str] = None
    sequence_end: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_scaler_loaded: bool
    target_scaler_loaded: bool
    dataset_records: int
    model_version: str

class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    target_variable: str
    sequence_length: int
    input_feature_names: List[str]
    supported_junctions: List[int]
    available_data_range: dict
    baseline_metrics: dict


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get("/", tags=["Root"])
def root():
    return {
        "service": "AI-Based Traffic Prediction System with MLOps API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "endpoints": ["/health", "/model-info", "/predict", "/junctions"]
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint: verifies service status and loaded model artifacts."""
    model_loaded = model_cache is not None
    f_scaler_loaded = feature_scaler_cache is not None
    t_scaler_loaded = target_scaler_cache is not None
    record_count = len(dataset_cache) if dataset_cache is not None else 0

    if not model_loaded or not f_scaler_loaded or not t_scaler_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "model_loaded": model_loaded,
                "feature_scaler_loaded": f_scaler_loaded,
                "target_scaler_loaded": t_scaler_loaded,
                "message": "Required model or scaler artifacts are missing or could not be loaded."
            }
        )

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        feature_scaler_loaded=f_scaler_loaded,
        target_scaler_loaded=t_scaler_loaded,
        dataset_records=record_count,
        model_version="traffic_lstm_v1"
    )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def model_info():
    """Returns model architecture, input feature schema, sequence window size, and baseline metrics."""
    min_date = str(dataset_cache['DateTime'].min()) if dataset_cache is not None else "Unknown"
    max_date = str(dataset_cache['DateTime'].max()) if dataset_cache is not None else "Unknown"

    return ModelInfoResponse(
        model_name="traffic_lstm",
        architecture="Sequential(Input(24, 6) -> LSTM(64, return_sequences=True) -> Dropout(0.2) -> LSTM(32) -> Dense(16, relu) -> Dense(1, linear))",
        target_variable=TARGET_COLUMN,
        sequence_length=SEQUENCE_LENGTH,
        input_feature_names=FEATURE_COLUMNS,
        supported_junctions=VALID_JUNCTIONS,
        available_data_range={
            "start": min_date,
            "end": max_date,
            "junction_1": "2015-11-01 00:00:00 to 2017-06-30 23:00:00",
            "junction_2": "2015-11-01 00:00:00 to 2017-06-30 23:00:00",
            "junction_3": "2015-11-01 00:00:00 to 2017-06-30 23:00:00",
            "junction_4": "2017-01-01 00:00:00 to 2017-06-30 23:00:00"
        },
        baseline_metrics=BASELINE_METRICS
    )

@app.get("/junctions", tags=["Metadata"])
def get_junctions():
    """Return summary metadata for each junction monitored."""
    if dataset_cache is None:
        return {"junctions": VALID_JUNCTIONS}

    summary = {}
    for j in VALID_JUNCTIONS:
        sub = dataset_cache[dataset_cache['Junction'] == j]
        if len(sub) > 0:
            summary[f"Junction_{j}"] = {
                "records": len(sub),
                "earliest_timestamp": str(sub['DateTime'].min()),
                "latest_timestamp": str(sub['DateTime'].max()),
                "mean_hourly_vehicles": float(sub['Vehicles'].mean()),
                "max_hourly_vehicles": float(sub['Vehicles'].max()),
                "min_hourly_vehicles": float(sub['Vehicles'].min())
            }
    return summary


def engineer_row_features(dt: datetime, vehicles: float, junction: int) -> List[float]:
    """Helper to convert timestamp, vehicles, and junction into model feature array."""
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    is_weekend = 1 if day_of_week in [5, 6] else 0
    # Must match FEATURE_COLUMNS: ['Vehicles', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'Junction']
    return [float(vehicles), float(hour), float(day_of_week), float(month), float(is_weekend), float(junction)]


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Generate traffic volume prediction for a target timestamp and junction.
    
    Modes:
    - 'historical': Retrieves preceding 24 hourly observations from the dataset (guaranteeing no future data leakage).
                    If target exists in dataset, returns actual vehicle count and residual error for evaluation.
    - 'future': Performs multi-step autoregressive rollout from the latest known observations up to target timestamp.
    - 'custom_sequence': Uses user-supplied 24 observations.
    """
    if model_cache is None or feature_scaler_cache is None or target_scaler_cache is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or scalers are not loaded on the server."
        )

    target_dt = pd.to_datetime(request.target_timestamp)
    junction = request.junction

    if junction not in VALID_JUNCTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid junction {junction}. Valid junctions are {VALID_JUNCTIONS}."
        )

    # -------------------------------------------------------------
    # CASE 1: Custom sequence supplied by client
    # -------------------------------------------------------------
    if request.custom_sequence is not None:
        seq_features = []
        for i, step in enumerate(request.custom_sequence):
            if isinstance(step, (int, float)):
                # If scalar vehicles provided, estimate time step backwards from target
                step_dt = target_dt - timedelta(hours=(SEQUENCE_LENGTH - i))
                row = engineer_row_features(step_dt, float(step), junction)
            elif isinstance(step, HistoricalSequenceStep):
                h = step.hour if step.hour is not None else (target_dt - timedelta(hours=SEQUENCE_LENGTH - i)).hour
                dow = step.day_of_week if step.day_of_week is not None else (target_dt - timedelta(hours=SEQUENCE_LENGTH - i)).weekday()
                m = step.month if step.month is not None else (target_dt - timedelta(hours=SEQUENCE_LENGTH - i)).month
                iw = step.is_weekend if step.is_weekend is not None else (1 if dow in [5, 6] else 0)
                jnc = step.junction if step.junction is not None else junction
                row = [float(step.vehicles), float(h), float(dow), float(m), float(iw), float(jnc)]
            elif isinstance(step, dict):
                v = float(step.get('vehicles', 0.0))
                h = int(step.get('hour', (target_dt - timedelta(hours=SEQUENCE_LENGTH - i)).hour))
                dow = int(step.get('day_of_week', (target_dt - timedelta(hours=SEQUENCE_LENGTH - i)).weekday()))
                m = int(step.get('month', (target_dt - timedelta(hours=SEQUENCE_LENGTH - i)).month))
                iw = int(step.get('is_weekend', 1 if dow in [5, 6] else 0))
                jnc = int(step.get('junction', junction))
                row = [v, float(h), float(dow), float(m), float(iw), float(jnc)]
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Unsupported format at sequence index {i}: {type(step)}"
                )
            seq_features.append(row)

        feature_df = pd.DataFrame(seq_features, columns=FEATURE_COLUMNS)
        scaled_seq = feature_scaler_cache.transform(feature_df)
        input_tensor = np.expand_dims(scaled_seq, axis=0)  # Shape (1, 24, 6)

        pred_scaled = model_cache.predict(input_tensor, verbose=0)
        pred_vehicles = float(target_scaler_cache.inverse_transform(pred_scaled)[0, 0])
        pred_vehicles = max(0.0, round(pred_vehicles, 2))

        return PredictResponse(
            target_timestamp=str(target_dt),
            junction=junction,
            predicted_vehicles=pred_vehicles,
            mode="custom_sequence",
            model_version="traffic_lstm_v1",
            note="Prediction computed from user-supplied 24-step historical sequence."
        )

    # -------------------------------------------------------------
    # CASE 2: Dataset lookup / Historical Evaluation Mode
    # -------------------------------------------------------------
    if dataset_cache is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Historical dataset cache is unavailable for lookup. Please supply custom_sequence."
        )

    junc_data = dataset_cache[dataset_cache['Junction'] == junction].sort_values('DateTime').reset_index(drop=True)
    if len(junc_data) < SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Junction {junction} has only {len(junc_data)} records, which is less than SEQUENCE_LENGTH ({SEQUENCE_LENGTH})."
        )

    min_avail_dt = junc_data['DateTime'].min()
    max_avail_dt = junc_data['DateTime'].max()

    # If requested timestamp falls within historical range or exact evaluation:
    if request.mode == "historical" or target_dt <= max_avail_dt:
        # Check if requested timestamp has preceding 24 hours available
        min_required_dt = min_avail_dt + timedelta(hours=SEQUENCE_LENGTH)
        if target_dt < min_required_dt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Timestamp {target_dt} is too early for Junction {junction}. "
                    f"Earliest predictable timestamp with 24 hours of history is {min_required_dt}."
                )
            )
        if target_dt > max_avail_dt + timedelta(hours=1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Timestamp {target_dt} is beyond available historical records for Junction {junction} (Max: {max_avail_dt}). "
                    "Use mode='future' to generate an autoregressive future forecast."
                )
            )

        # Retrieve strictly preceding 24 observations (DateTime < target_dt)
        history_df = junc_data[junc_data['DateTime'] < target_dt].tail(SEQUENCE_LENGTH)
        if len(history_df) != SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient contiguous observations preceding {target_dt}. Found {len(history_df)}, expected {SEQUENCE_LENGTH}."
            )

        seq_matrix = history_df[FEATURE_COLUMNS]
        scaled_seq = feature_scaler_cache.transform(seq_matrix)
        input_tensor = np.expand_dims(scaled_seq, axis=0)

        pred_scaled = model_cache.predict(input_tensor, verbose=0)
        pred_vehicles = float(target_scaler_cache.inverse_transform(pred_scaled)[0, 0])
        pred_vehicles = max(0.0, round(pred_vehicles, 2))

        # Check if actual vehicle ground truth exists at target_dt for evaluation
        actual_row = junc_data[junc_data['DateTime'] == target_dt]
        actual_val = None
        residual_val = None
        if not actual_row.empty:
            actual_val = float(actual_row['Vehicles'].values[0])
            residual_val = round(pred_vehicles - actual_val, 2)

        return PredictResponse(
            target_timestamp=str(target_dt),
            junction=junction,
            predicted_vehicles=pred_vehicles,
            mode="historical_evaluation",
            model_version="traffic_lstm_v1",
            actual_vehicles=actual_val,
            residual_error=residual_val,
            sequence_start=str(history_df['DateTime'].iloc[0]),
            sequence_end=str(history_df['DateTime'].iloc[-1]),
            note=(
                "Evaluated using the exact 24 hourly observations preceding the target timestamp. "
                "No future data was leaked into input features."
            )
        )

    # -------------------------------------------------------------
    # CASE 3: Mode 2 — Future Autoregressive / Recursive Forecast
    # -------------------------------------------------------------
    if request.mode == "future":
        hours_ahead = int((target_dt - max_avail_dt).total_seconds() // 3600)
        if hours_ahead <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Timestamp {target_dt} is not in the future (Max historical is {max_avail_dt}). Use mode='historical'."
            )
        if hours_ahead > 72:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Future forecast requested {hours_ahead} hours ahead. "
                    "Recursive multi-step forecast is capped at 72 hours due to error accumulation."
                )
            )

        # Start from the latest 24 observations
        current_seq_df = junc_data.tail(SEQUENCE_LENGTH).copy()
        current_dt = current_seq_df['DateTime'].iloc[-1]
        
        # Roll forward hour by hour
        last_pred = 0.0
        for step in range(hours_ahead):
            next_dt = current_dt + timedelta(hours=1)
            seq_matrix = current_seq_df[FEATURE_COLUMNS]
            scaled_seq = feature_scaler_cache.transform(seq_matrix)
            input_tensor = np.expand_dims(scaled_seq, axis=0)

            pred_scaled = model_cache.predict(input_tensor, verbose=0)
            pred_val = float(target_scaler_cache.inverse_transform(pred_scaled)[0, 0])
            pred_val = max(0.0, round(pred_val, 2))
            last_pred = pred_val

            # Create new row for next step
            new_row = {
                'DateTime': next_dt,
                'Junction': junction,
                'Vehicles': pred_val,
                'Hour': next_dt.hour,
                'DayOfWeek': next_dt.weekday(),
                'Day': next_dt.day,
                'Month': next_dt.month,
                'IsWeekend': 1 if next_dt.weekday() in [5, 6] else 0
            }
            current_seq_df = pd.concat([current_seq_df.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
            current_dt = next_dt

        return PredictResponse(
            target_timestamp=str(target_dt),
            junction=junction,
            predicted_vehicles=last_pred,
            mode="future_forecast",
            model_version="traffic_lstm_v1",
            actual_vehicles=None,
            residual_error=None,
            note=(
                f"Multi-step recursive forecast ({hours_ahead} hours ahead from last observation {max_avail_dt}). "
                "Caveat: Recursive multi-step forecasts compound uncertainty with each step."
            )
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unknown mode '{request.mode}'. Choose from 'historical', 'future', or provide 'custom_sequence'."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
