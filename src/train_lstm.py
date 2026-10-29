import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.keras

def build_lstm_model(input_shape, lstm_units_1=64, lstm_units_2=32, dropout_rate=0.2, dense_units=16, learning_rate=0.001):
    """
    Build LSTM Architecture:
    Input -> LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(16) -> Dense(1)
    """
    model = Sequential([
        Input(shape=input_shape, name="time_series_input"),
        LSTM(lstm_units_1, return_sequences=True, name="lstm_layer_1"),
        Dropout(dropout_rate, name="dropout_1"),
        LSTM(lstm_units_2, return_sequences=False, name="lstm_layer_2"),
        Dense(dense_units, activation="relu", name="dense_intermediate"),
        Dense(1, name="traffic_prediction_output")
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model

def train_model(data_path='data/processed/traffic_sequences.npz',
                model_output_path='models/traffic_lstm.keras',
                lstm_units_1=64,
                lstm_units_2=32,
                dropout_rate=0.2,
                dense_units=16,
                epochs=25,
                batch_size=64,
                learning_rate=0.001,
                experiment_name="traffic-prediction-lstm"):
    """Train LSTM model with MLflow experiment tracking."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed sequence file not found at: {data_path}. Run data_preprocessing.py first.")
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # 1. Load preprocessed sequence arrays
    print(f"[TRAIN] Loading sequence datasets from {data_path}...")
    data = np.load(data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    print(f"[TRAIN] Training shape: X={X_train.shape}, y={y_train.shape}")
    print(f"[TRAIN] Validation shape: X={X_val.shape}, y={y_val.shape}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # 2. Build model
    model = build_lstm_model(
        input_shape=input_shape,
        lstm_units_1=lstm_units_1,
        lstm_units_2=lstm_units_2,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        learning_rate=learning_rate
    )
    model.summary()
    
    # 3. Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath=model_output_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # 4. MLflow Experiment Setup
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="lstm_traffic_run") as run:
        print(f"\n[MLFLOW] Active Run ID: {run.info.run_id}")
        print(f"[MLFLOW] Experiment Name: {experiment_name}")
        
        # Log Parameters
        mlflow.log_params({
            "sequence_length": input_shape[0],
            "n_features": input_shape[1],
            "lstm_units_1": lstm_units_1,
            "lstm_units_2": lstm_units_2,
            "dropout_rate": dropout_rate,
            "dense_units": dense_units,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss_function": "Mean Squared Error"
        })
        
        # Train model
        print("\n[TRAIN] Commencing LSTM model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        # Log final / best epoch metrics
        best_epoch = np.argmin(history.history['val_loss'])
        final_train_loss = float(history.history['loss'][best_epoch])
        final_val_loss = float(history.history['val_loss'][best_epoch])
        final_train_mae = float(history.history['mae'][best_epoch])
        final_val_mae = float(history.history['val_mae'][best_epoch])
        
        mlflow.log_metrics({
            "train_loss_mse": final_train_loss,
            "val_loss_mse": final_val_loss,
            "train_mae": final_train_mae,
            "val_mae": final_val_mae,
            "stopped_epoch": len(history.history['loss']),
            "best_epoch": int(best_epoch + 1)
        })
        
        # Ensure model is saved locally
        model.save(model_output_path)
        print(f"[TRAIN] Model saved locally to {model_output_path}")
        
        # Log model artifact to MLflow
        try:
            mlflow.keras.log_model(model, artifact_path="model")
            print("[MLFLOW] Logged model artifact to MLflow.")
        except Exception as e:
            print(f"[MLFLOW] Note on model logging: {e}")
            
    print(f"[TRAIN] Training completed! Best validation loss: {final_val_loss:.6f}")
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM Traffic Prediction Model.")
    parser.add_argument('--data_path', type=str, default='data/processed/traffic_sequences.npz', help='Path to sequences npz')
    parser.add_argument('--model_path', type=str, default='models/traffic_lstm.keras', help='Path to save model')
    parser.add_argument('--epochs', type=int, default=25, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        model_output_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
