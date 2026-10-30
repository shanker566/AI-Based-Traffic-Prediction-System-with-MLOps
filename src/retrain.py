import os
import sys
import shutil
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import mlflow
import mlflow.keras

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from src.train_lstm import build_lstm_model
except ModuleNotFoundError:
    from train_lstm import build_lstm_model
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffic_sequences.npz")
PRODUCTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_lstm.keras")
CANDIDATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "candidate_traffic_lstm.keras")
BACKUP_MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_lstm_backup.keras")
TARGET_SCALER_PATH = os.path.join(BASE_DIR, "models", "target_scaler.pkl")

EXPERIMENT_NAME = "traffic-prediction-lstm"

def evaluate_model_performance(model, X_test, y_test, target_scaler):
    """Compute MAE, RMSE, MSE, and R2 on unscaled predictions."""
    preds_scaled = model.predict(X_test, batch_size=128, verbose=0)
    y_test_act = target_scaler.inverse_transform(y_test).flatten()
    preds_act = target_scaler.inverse_transform(preds_scaled).flatten()

    mae = float(mean_absolute_error(y_test_act, preds_act))
    mse = float(mean_squared_error(y_test_act, preds_act))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test_act, preds_act))

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }

def run_safe_retraining(epochs=20,
                        batch_size=64,
                        lr=0.001,
                        max_rmse_tolerance=0.05,
                        min_r2_threshold=0.85):
    """
    Safe Retraining Pipeline:
    1. Evaluates current production model to set baseline.
    2. Trains a candidate model saved separately to models/candidate_traffic_lstm.keras.
    3. Evaluates candidate model on unseen test set.
    4. Evaluates promotion rules.
    5. Backs up previous production model and promotes candidate only if approved.
    6. Logs full lifecycle to MLflow.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed sequence file not found at {DATA_PATH}. Run data_preprocessing.py first.")
    if not os.path.exists(TARGET_SCALER_PATH):
        raise FileNotFoundError(f"Target scaler not found at {TARGET_SCALER_PATH}.")

    print("\n==================================================")
    print("SAFE MANUAL RETRAINING & CANDIDATE PROMOTION")
    print("==================================================")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MLflow Experiment: {EXPERIMENT_NAME}")
    print("==================================================")

    data = np.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    # 1. Evaluate baseline production model
    baseline_metrics = None
    if os.path.exists(PRODUCTION_MODEL_PATH):
        try:
            print(f"[RETRAIN] Evaluating current production model at {PRODUCTION_MODEL_PATH}...")
            prod_model = tf.keras.models.load_model(PRODUCTION_MODEL_PATH)
            baseline_metrics = evaluate_model_performance(prod_model, X_test, y_test, target_scaler)
            print(f"[RETRAIN] Baseline Production Metrics: RMSE={baseline_metrics['rmse']:.4f}, MAE={baseline_metrics['mae']:.4f}, R²={baseline_metrics['r2']:.4f}")
        except Exception as e:
            print(f"[RETRAIN WARNING] Failed to evaluate current production model: {e}")
            baseline_metrics = {"rmse": 8.2154, "mae": 6.1383, "r2": 0.9137}
    else:
        print("[RETRAIN] No previous production model found. Using default historical baseline.")
        baseline_metrics = {"rmse": 8.2154, "mae": 6.1383, "r2": 0.9137}

    # 2. Build and train candidate model
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\n[RETRAIN] Building candidate model with input shape {input_shape}...")
    candidate_model = build_lstm_model(input_shape=input_shape, learning_rate=lr)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=CANDIDATE_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"retraining_candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        print(f"[MLFLOW] Retraining Run ID: {run_id}")

        mlflow.log_params({
            "run_type": "retraining_candidate",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_rmse_tolerance": max_rmse_tolerance,
            "min_r2_threshold": min_r2_threshold,
            "baseline_rmse": baseline_metrics["rmse"],
            "baseline_mae": baseline_metrics["mae"],
            "baseline_r2": baseline_metrics["r2"]
        })

        print("\n[RETRAIN] Training candidate model...")
        history = candidate_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        candidate_model.save(CANDIDATE_MODEL_PATH)
        print(f"[RETRAIN] Candidate model saved to {CANDIDATE_MODEL_PATH}")

        # 3. Evaluate candidate model
        print("\n[RETRAIN] Evaluating candidate model on holdout test set...")
        candidate_metrics = evaluate_model_performance(candidate_model, X_test, y_test, target_scaler)
        print(f"[RETRAIN] Candidate Metrics: RMSE={candidate_metrics['rmse']:.4f}, MAE={candidate_metrics['mae']:.4f}, R²={candidate_metrics['r2']:.4f}")

        mlflow.log_metrics({
            "candidate_mae": candidate_metrics["mae"],
            "candidate_mse": candidate_metrics["mse"],
            "candidate_rmse": candidate_metrics["rmse"],
            "candidate_r2": candidate_metrics["r2"]
        })

        # 4. Smoke test candidate model
        smoke_input = X_test[:2]
        smoke_pred = candidate_model.predict(smoke_input, verbose=0)
        smoke_passed = not np.isnan(smoke_pred).any() and len(smoke_pred) == 2

        # 5. Evaluate Promotion Decision Rules
        rmse_diff = candidate_metrics["rmse"] - baseline_metrics["rmse"]
        r2_pass = candidate_metrics["r2"] >= min_r2_threshold
        rmse_pass = candidate_metrics["rmse"] <= (baseline_metrics["rmse"] * (1.0 + max_rmse_tolerance))
        
        promote = smoke_passed and r2_pass and rmse_pass

        print("\n==================================================")
        print("CANDIDATE MODEL VALIDATION & PROMOTION AUDIT")
        print("==================================================")
        print(f"Rule 1 - Smoke Test (Inference & NaN check):  {'PASSED' if smoke_passed else 'FAILED'}")
        print(f"Rule 2 - Minimum R² Threshold (>= {min_r2_threshold}):     {'PASSED' if r2_pass else 'FAILED'} (R² = {candidate_metrics['r2']:.4f})")
        print(f"Rule 3 - RMSE Tolerance (<= {baseline_metrics['rmse'] * (1.0 + max_rmse_tolerance):.4f}): {'PASSED' if rmse_pass else 'FAILED'} (Candidate RMSE = {candidate_metrics['rmse']:.4f})")
        print(f"Delta RMSE vs Baseline: {rmse_diff:+.4f} vehicles")
        print("==================================================")

        if promote:
            print(">>> PROMOTION STATUS: APPROVED")
            print(f"[RETRAIN] Backing up current model to: {BACKUP_MODEL_PATH}")
            if os.path.exists(PRODUCTION_MODEL_PATH):
                shutil.copyfile(PRODUCTION_MODEL_PATH, BACKUP_MODEL_PATH)
            
            print(f"[RETRAIN] Promoting candidate model to: {PRODUCTION_MODEL_PATH}")
            shutil.copyfile(CANDIDATE_MODEL_PATH, PRODUCTION_MODEL_PATH)

            mlflow.log_params({"promotion_decision": "PROMOTED"})
            try:
                mlflow.keras.log_model(candidate_model, artifact_path="promoted_model")
            except Exception as e:
                print(f"[MLFLOW NOTE] Model artifact logging: {e}")
            decision_msg = "Candidate model passed all validation rules and was PROMOTED to production."
        else:
            print(">>> PROMOTION STATUS: REJECTED")
            print("[RETRAIN] Production model kept unchanged. Candidate model remains at models/candidate_traffic_lstm.keras for audit.")
            mlflow.log_params({"promotion_decision": "REJECTED"})
            decision_msg = "Candidate model did not meet promotion criteria. Production model preserved."

        print(f"\n[RETRAIN] Summary: {decision_msg}")
        return {
            "mlflow_run_id": run_id,
            "promoted": promote,
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "decision": decision_msg
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute safe manual retraining and promotion.")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for candidate model training")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Allowed relative RMSE increase (default 0.05)")
    args = parser.parse_args()

    run_safe_retraining(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_rmse_tolerance=args.tolerance
    )
