import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Traffic Prediction System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F9FAFB 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .badge-healthy {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-error {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Configuration & Session State
# -----------------------------------------------------------------------------
DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
st.sidebar.title("🚦 System Controls")

api_url = st.sidebar.text_input("FastAPI Base URL", value=DEFAULT_API_URL, help="Base address of the FastAPI serving service")

# Health Check Ping
api_status = False
health_data = {}
try:
    resp = requests.get(f"{api_url}/health", timeout=3)
    if resp.status_code == 200:
        api_status = True
        health_data = resp.json()
except Exception:
    api_status = False

if api_status:
    st.sidebar.markdown('<span class="badge-healthy">● API Online (Connected)</span>', unsafe_allow_html=True)
    st.sidebar.caption(f"Model: `{health_data.get('model_version', 'v1')}` | Dataset: `{health_data.get('dataset_records', 0):,}` rows")
else:
    st.sidebar.markdown('<span class="badge-error">● API Offline / Unreachable</span>', unsafe_allow_html=True)
    st.sidebar.warning(f"Could not reach FastAPI at `{api_url}`. Please ensure the API is running.")

st.sidebar.markdown("---")
selected_junction = st.sidebar.selectbox("Select Traffic Junction", [1, 2, 3, 4], index=0)
selected_mode = st.sidebar.radio(
    "Prediction Mode",
    ["Historical Evaluation (Test Set)", "Future Forecast (Recursive)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Project Details")
st.sidebar.caption(
    "**Course**: College MLOps Project (Phase 2)\n\n"
    "**Architecture**: Deep LSTM Network\n\n"
    "**Stack**: FastAPI, Streamlit, MLflow, DVC, Evidently AI (Native Python)"
)

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">AI-Based Traffic Prediction System with MLOps</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Hourly vehicle volume forecasting using Deep LSTM sequence models, '
    'FastAPI model serving, and live MLOps pipeline monitoring.</div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_eval, tab_history, tab_model = st.tabs([
    "🎯 Live Prediction",
    "📊 Baseline Evaluation",
    "🕒 Session History",
    "ℹ️ Model & Architecture"
])

# =============================================================================
# TAB 1: Live Prediction
# =============================================================================
with tab_predict:
    st.markdown("### 🔮 Traffic Volume Inference")
    
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### Input Parameters")

        if selected_mode == "Historical Evaluation (Test Set)":
            st.info(
                "💡 **Mode 1 — Historical Evaluation**: Selects a real timestamp from the dataset, "
                "retrieves the preceding 24-hour sequence, generates the prediction, and compares against "
                "the actual ground-truth count without future data leakage."
            )
            # Default to test partition date
            selected_date = st.date_input("Target Date", value=datetime(2017, 6, 30).date())
            selected_hour = st.slider("Target Hour of Day (00:00 - 23:00)", 0, 23, 18)
            target_dt_str = f"{selected_date} {selected_hour:02d}:00:00"
            mode_api_val = "historical"

        else:
            st.warning(
                "⚠️ **Mode 2 — Future Forecast**: Extends beyond the last available historical observation "
                "(2017-06-30 23:00) using recursive autoregressive rollout. Multi-step forecasts accumulate uncertainty."
            )
            hours_ahead = st.slider("Forecast Horizon (Hours Ahead)", min_value=1, max_value=48, value=6)
            base_dt = datetime(2017, 6, 30, 23, 0, 0)
            target_dt = base_dt + timedelta(hours=hours_ahead)
            target_dt_str = target_dt.strftime("%Y-%m-%d %H:%M:%S")
            mode_api_val = "future"
            st.caption(f"Forecasting for target timestamp: `{target_dt_str}` (+{hours_ahead} hrs)")

        btn_predict = st.button("🚀 Predict Traffic Volume", type="primary", use_container_width=True)

    with col_result:
        st.markdown("#### Prediction Output")

        if btn_predict:
            if not api_status:
                st.error(f"Cannot generate prediction: FastAPI service is unreachable at `{api_url}`. Start it using `python -m uvicorn api.main:app --port 8000`.")
            else:
                with st.spinner("Calling FastAPI inference endpoint..."):
                    payload = {
                        "target_timestamp": target_dt_str,
                        "junction": selected_junction,
                        "mode": mode_api_val
                    }
                    try:
                        resp = requests.post(f"{api_url}/predict", json=payload, timeout=8)
                        if resp.status_code == 200:
                            data = resp.json()
                            pred_val = data["predicted_vehicles"]
                            act_val = data.get("actual_vehicles")
                            res_val = data.get("residual_error")

                            st.success("✅ Prediction Successfully Generated")

                            m_col1, m_col2, m_col3 = st.columns(3)
                            m_col1.metric("Predicted Vehicles", f"{pred_val:.1f}")
                            if act_val is not None:
                                m_col2.metric("Actual Vehicles", f"{act_val:.1f}")
                                m_col3.metric("Error (Pred - Actual)", f"{res_val:+.1f}")
                            else:
                                m_col2.metric("Actual Vehicles", "N/A (Future)")
                                m_col3.metric("Status", "Genuine Forecast")

                            if data.get("note"):
                                st.info(data["note"])

                            # Save to session history
                            st.session_state.prediction_history.append({
                                "Timestamp": target_dt_str,
                                "Junction": selected_junction,
                                "Mode": data["mode"],
                                "Predicted": pred_val,
                                "Actual": act_val if act_val is not None else "N/A",
                                "Residual": res_val if res_val is not None else "N/A",
                                "Generated At": datetime.now().strftime("%H:%M:%S")
                            })

                        else:
                            st.error(f"API Error ({resp.status_code}): {resp.text}")
                    except Exception as err:
                        st.error(f"Network error calling API: {err}")
        else:
            st.info("👈 Select parameters on the left and click **Predict Traffic Volume**.")

# =============================================================================
# TAB 2: Baseline Model Evaluation
# =============================================================================
with tab_eval:
    st.markdown("### 📈 Baseline Model Performance (Phase 1 Holdout Test Set)")
    st.markdown(
        "Performance measured on the **unseen chronological test set** "
        "($N = 7,194$ sequences, April 2017 to June 2017). "
        "Strictly verified without data leakage."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Absolute Error (MAE)", "6.14 vehicles", help="Average absolute difference between actual and predicted count")
    c2.metric("Root Mean Squared Error (RMSE)", "8.22 vehicles", help="Square root of mean squared error")
    c3.metric("Mean Squared Error (MSE)", "67.49")
    c4.metric("R² Score (Variance Explained)", "91.37%", help="Percentage of total variance explained by the LSTM model")

    st.markdown("---")
    st.markdown("#### Actual vs. Predicted Traffic Snapshot (168-Hour / 1-Week Horizon)")

    fig_path = os.path.join(os.path.dirname(__file__), "..", "reports", "figures", "actual_vs_predicted.png")
    if os.path.exists(fig_path):
        st.image(fig_path, caption="LSTM Actual vs. Predicted Traffic on Holdout Test Horizon", use_container_width=True)
    else:
        st.info("Evaluation figure `actual_vs_predicted.png` can be generated by running `python src/evaluate.py`.")

# =============================================================================
# TAB 3: Session Prediction History
# =============================================================================
with tab_history:
    st.markdown("### 🕒 Session Prediction History")
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions made during this browser session yet. Run a prediction from the 'Live Prediction' tab.")
    else:
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(hist_df, use_container_width=True)
        
        # Download as CSV button
        csv_data = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Session Predictions (CSV)",
            data=csv_data,
            file_name="traffic_session_predictions.csv",
            mime="text/csv"
        )

# =============================================================================
# TAB 4: Model Architecture & Schema
# =============================================================================
with tab_model:
    st.markdown("### ℹ️ LSTM Model Architecture & Features Schema")
    
    st.markdown("""
    #### 🧠 Deep LSTM Architecture
    ```text
    Input Layer (Batch, 24 Steps, 6 Features)
           │
           ▼
    LSTM Layer 1 (64 Units, return_sequences=True)
           │
           ▼
    Dropout Layer (Rate = 0.20)
           │
           ▼
    LSTM Layer 2 (32 Units, return_sequences=False)
           │
           ▼
    Dense Intermediate Layer (16 Units, ReLU Activation)
           │
           ▼
    Dense Output Layer (1 Unit, Linear Activation -> Predicted Hourly Vehicle Volume)
    ```

    #### 📋 Input Feature Columns (Exact Chronological Order)
    1. **`Vehicles`**: Scaled vehicle count from preceding observation
    2. **`Hour`**: Hour of day ($0$ to $23$)
    3. **`DayOfWeek`**: Day of week ($0 = \text{Monday}, \dots, 6 = \text{Sunday}$)
    4. **`Month`**: Calendar month ($1$ to $12$)
    5. **`IsWeekend`**: Binary flag ($1$ for Saturday/Sunday, else $0$)
    6. **`Junction`**: Road junction identifier ($1, 2, 3, 4$)

    #### 🔄 Transformation Pipeline
    - Scaler: `MinMaxScaler(feature_range=(0, 1))` fitted strictly on Training split.
    - Inverse Transform: Applied using `models/target_scaler.pkl` to convert model output back to natural vehicle count.
    """)
