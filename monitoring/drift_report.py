import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

# Evidently imports
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# TensorFlow model import for performance evaluation
import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_traffic.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "traffic.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_lstm.keras")
FEATURE_SCALER_PATH = os.path.join(BASE_DIR, "models", "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(BASE_DIR, "models", "target_scaler.pkl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

FEATURE_COLUMNS = ['Vehicles', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'Junction']
TARGET_COLUMN = 'Vehicles'
SEQUENCE_LENGTH = 24

def load_data():
    """Load processed or raw traffic dataset."""
    data_path = PROCESSED_DATA_PATH if os.path.exists(PROCESSED_DATA_PATH) else RAW_DATA_PATH
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Traffic dataset not found at {data_path}. Run preprocessing first.")
    
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    if 'Hour' not in df.columns:
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        df['Day'] = df['DateTime'].dt.day
        df['Month'] = df['DateTime'].dt.month
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    df = df.sort_values(by=['DateTime', 'Junction']).reset_index(drop=True)
    return df

def generate_drift_report(output_html='reports/evidently_drift_report.html',
                          output_json='reports/drift_summary.json'):
    """
    Generate an Evidently Data Drift Report comparing:
    - Reference Dataset: Chronological Training Partition (first 70% of data)
    - Current Dataset: Chronological Test Partition (last 15% of data / holdout batch)
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = load_data()
    n = len(df)
    train_end = int(n * 0.70)
    test_start = int(n * 0.85)

    ref_df = df.iloc[:train_end][FEATURE_COLUMNS].copy()
    curr_df = df.iloc[test_start:][FEATURE_COLUMNS].copy()

    ref_start = df['DateTime'].iloc[0]
    ref_end = df['DateTime'].iloc[train_end - 1]
    curr_start = df['DateTime'].iloc[test_start]
    curr_end = df['DateTime'].iloc[-1]

    print("\n==================================================")
    print("EVIDENTLY AI - DATA DRIFT MONITORING")
    print("==================================================")
    print(f"Monitoring Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Reference Dataset (Train 70%): {len(ref_df)} rows | {ref_start} to {ref_end}")
    print(f"Current Dataset   (Test 15%):  {len(curr_df)} rows | {curr_start} to {curr_end}")
    print(f"Monitored Features: {FEATURE_COLUMNS}")
    print("==================================================")

    # Instantiate Evidently Report with Data Drift & Summary presets
    print("[MONITORING] Running statistical drift tests...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset()
    ])

    snapshot = report.run(reference_data=ref_df, current_data=curr_df)

    # Save HTML interactive report
    snapshot.save_html(output_html)
    print(f"[MONITORING] Saved interactive HTML drift report to: {output_html}")

    # Evaluate Model Performance Monitoring if model and scalers exist
    perf_metrics = {}
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_SCALER_PATH) and os.path.exists(TARGET_SCALER_PATH):
        try:
            print("\n[MONITORING] Evaluating model performance drift against ground-truth labels...")
            model = tf.keras.models.load_model(MODEL_PATH)
            f_scaler = joblib.load(FEATURE_SCALER_PATH)
            t_scaler = joblib.load(TARGET_SCALER_PATH)

            # Construct sequences on the current test holdout
            test_sub = df.iloc[test_start:].copy().reset_index(drop=True)
            if len(test_sub) > SEQUENCE_LENGTH:
                scaled_feat = f_scaler.transform(test_sub[FEATURE_COLUMNS])
                X_mon, y_mon_actual = [], []
                for i in range(len(test_sub) - SEQUENCE_LENGTH):
                    X_mon.append(scaled_feat[i : i + SEQUENCE_LENGTH])
                    y_mon_actual.append(test_sub['Vehicles'].iloc[i + SEQUENCE_LENGTH])
                
                X_mon = np.array(X_mon)
                y_mon_actual = np.array(y_mon_actual)

                preds_scaled = model.predict(X_mon, batch_size=128, verbose=0)
                preds = t_scaler.inverse_transform(preds_scaled).flatten()

                mae = float(np.mean(np.abs(preds - y_mon_actual)))
                mse = float(np.mean((preds - y_mon_actual) ** 2))
                rmse = float(np.sqrt(mse))
                
                perf_metrics = {
                    "evaluation_window_sequences": len(X_mon),
                    "monitoring_mae": round(mae, 4),
                    "monitoring_mse": round(mse, 4),
                    "monitoring_rmse": round(rmse, 4),
                    "baseline_mae": 6.1383,
                    "baseline_rmse": 8.2154,
                    "performance_status": "Healthy (within expected generalization boundary)" if rmse <= 12.0 else "Degraded"
                }
                print(f"[MONITORING] Test Horizon MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        except Exception as e:
            print(f"[MONITORING WARNING] Could not compute model performance metrics: {e}")

    # Generate summary JSON metadata
    summary = {
        "timestamp": datetime.now().isoformat(),
        "reference_dataset": {
            "name": "Phase-1 Training Split (First 70%)",
            "row_count": len(ref_df),
            "date_range": [str(ref_start), str(ref_end)]
        },
        "current_dataset": {
            "name": "Holdout Test Batch (Last 15%)",
            "row_count": len(curr_df),
            "date_range": [str(curr_start), str(curr_end)]
        },
        "features_monitored": FEATURE_COLUMNS,
        "performance_monitoring": perf_metrics,
        "html_report_path": output_html
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[MONITORING] Saved drift summary metadata to: {output_json}\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evidently AI Data & Performance Drift Report")
    parser.add_argument("--output-html", type=str, default="reports/evidently_drift_report.html")
    parser.add_argument("--output-json", type=str, default="reports/drift_summary.json")
    args = parser.parse_args()

    generate_drift_report(output_html=args.output_html, output_json=args.output_json)
