import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import mlflow

def evaluate_model(data_path='data/processed/traffic_sequences.npz',
                   model_path='models/traffic_lstm.keras',
                   scaler_path='models/target_scaler.pkl',
                   figure_output_path='reports/figures/actual_vs_predicted.png',
                   experiment_name='traffic-prediction-lstm'):
    """Evaluate trained LSTM model on unseen test set."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed sequence file not found at: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Target scaler file not found at: {scaler_path}")
        
    os.makedirs(os.path.dirname(figure_output_path), exist_ok=True)
    
    # 1. Load test sequences & model & scaler
    print(f"[EVALUATE] Loading test data from {data_path}...")
    data = np.load(data_path)
    X_test = data['X_test']
    y_test_scaled = data['y_test']
    
    print(f"[EVALUATE] Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"[EVALUATE] Loading target scaler from {scaler_path}...")
    target_scaler = joblib.load(scaler_path)
    
    # 2. Make predictions
    print(f"[EVALUATE] Generating predictions for {len(X_test)} test sequences...")
    y_pred_scaled = model.predict(X_test, batch_size=128, verbose=0)
    
    # 3. Inverse transform scaled predictions back to actual vehicle counts
    y_test_actual = target_scaler.inverse_transform(y_test_scaled).flatten()
    y_pred_actual = target_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # 4. Calculate actual metrics
    mae = float(mean_absolute_error(y_test_actual, y_pred_actual))
    mse = float(mean_squared_error(y_test_actual, y_pred_actual))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test_actual, y_pred_actual))
    
    print("\n==================================================")
    print("MODEL EVALUATION METRICS (TEST SET)")
    print("==================================================")
    print(f"Mean Absolute Error (MAE):  {mae:.4f} vehicles")
    print(f"Mean Squared Error (MSE):   {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} vehicles")
    print(f"R-squared Score (R²):       {r2:.4f}")
    print("==================================================")
    
    # 5. Display sample predictions table
    sample_size = 15
    print(f"\n--- Sample Predictions Table (First {sample_size} Test Observations) ---")
    print(f"{'Index':<8} | {'Actual Vehicles':<16} | {'Predicted Vehicles':<18} | {'Difference':<12}")
    print("-" * 62)
    for idx in range(sample_size):
        actual = y_test_actual[idx]
        pred = y_pred_actual[idx]
        diff = pred - actual
        print(f"{idx:<8} | {actual:<16.2f} | {pred:<18.2f} | {diff:<+12.2f}")
    print("-" * 62)
    
    # 6. Generate Actual vs Predicted Plot
    # Visualizing a 168-hour (1 week) snapshot to show granular hourly tracking clearly
    plot_window = min(168, len(y_test_actual))
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2.2, 1]})
    
    # Subplot 1: Time series overlay for a 1-week test window
    ax1.plot(range(plot_window), y_test_actual[:plot_window], label='Actual Traffic Volume', color='#1f77b4', linewidth=2.2, marker='o', markersize=3.5)
    ax1.plot(range(plot_window), y_pred_actual[:plot_window], label='Predicted Traffic (LSTM)', color='#d62728', linestyle='--', linewidth=2.0)
    ax1.set_title(f'Actual vs Predicted Traffic Volume (168-Hour / 1-Week Test Snapshot)\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Time Step (Hours in Test Horizon)', fontsize=11)
    ax1.set_ylabel('Vehicle Count', fontsize=11)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Prediction error / residuals
    residuals = y_pred_actual[:plot_window] - y_test_actual[:plot_window]
    ax2.bar(range(plot_window), residuals, color='#7f7f7f', alpha=0.65, width=0.8, label='Residual Error (Pred - Actual)')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Prediction Residuals / Error per Hour', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time Step (Hours)', fontsize=11)
    ax2.set_ylabel('Residual (Vehicles)', fontsize=11)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(figure_output_path, dpi=300)
    plt.close()
    print(f"\n[EVALUATE] Saved comparison plot to {figure_output_path}")
    
    # 7. Log metrics and plot to MLflow
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="lstm_test_evaluation"):
            mlflow.log_metrics({
                "test_mae": mae,
                "test_mse": mse,
                "test_rmse": rmse,
                "test_r2": r2
            })
            mlflow.log_artifact(figure_output_path, artifact_path="evaluation_figures")
            print("[MLFLOW] Logged test metrics and evaluation figure to MLflow.")
    except Exception as e:
        print(f"[MLFLOW] Note on evaluation MLflow logging: {e}")
        
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate LSTM Traffic Prediction Model.")
    parser.add_argument('--data_path', type=str, default='data/processed/traffic_sequences.npz')
    parser.add_argument('--model_path', type=str, default='models/traffic_lstm.keras')
    parser.add_argument('--scaler_path', type=str, default='models/target_scaler.pkl')
    parser.add_argument('--figure_path', type=str, default='reports/figures/actual_vs_predicted.png')
    args = parser.parse_args()
    
    evaluate_model(
        data_path=args.data_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        figure_output_path=args.figure_path
    )
