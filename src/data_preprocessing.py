import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SEQUENCE_LENGTH = 24

def load_raw_data(filepath='data/raw/traffic.csv'):
    """Load raw traffic data from csv."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    
    print(f"[PREPROCESS] Loading raw data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def clean_and_engineer_features(df):
    """Clean data and create time-based features."""
    # Convert DateTime
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Sort chronologically
    df = df.sort_values(by=['DateTime', 'Junction']).reset_index(drop=True)
    
    # Check & handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"[PREPROCESS] Found {missing_count} missing values. Imputing...")
        df = df.ffill().bfill()
    else:
        print("[PREPROCESS] Missing values: 0")
        
    # Check & handle duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"[PREPROCESS] Found {dup_count} duplicates. Dropping duplicates...")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("[PREPROCESS] Duplicates: 0")
        
    # Time feature engineering
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Day'] = df['DateTime'].dt.day
    df['Month'] = df['DateTime'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    return df

def create_sequences_for_series(feature_data, target_data, seq_length=24):
    """Create sliding window sequences for LSTM."""
    X, y = [], []
    for i in range(len(feature_data) - seq_length):
        X.append(feature_data[i : i + seq_length])
        y.append(target_data[i + seq_length])
    return np.array(X), np.array(y)

def preprocess_pipeline(raw_path='data/raw/traffic.csv', 
                        processed_csv_path='data/processed/processed_traffic.csv',
                        seq_length=SEQUENCE_LENGTH,
                        train_ratio=0.70,
                        val_ratio=0.15):
    """Full preprocessing and sequence preparation pipeline with 70/15/15 chronological split."""
    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. Load and Clean
    df = load_raw_data(raw_path)
    df = clean_and_engineer_features(df)
    
    # Save clean processed tabular data
    df.to_csv(processed_csv_path, index=False)
    print(f"[PREPROCESS] Saved processed dataset to {processed_csv_path}")
    
    # 2. Chronological Split (70% Train, 15% Validation, 15% Testing)
    total_samples = len(df)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print("\n==================================================")
    print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
    print("==================================================")
    print(f"Training samples:   {len(train_df)} ({len(train_df)/total_samples*100:.1f}%) [From: {train_df['DateTime'].min()} To: {train_df['DateTime'].max()}]")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/total_samples*100:.1f}%) [From: {val_df['DateTime'].min()} To: {val_df['DateTime'].max()}]")
    print(f"Testing samples:    {len(test_df)} ({len(test_df)/total_samples*100:.1f}%) [From: {test_df['DateTime'].min()} To: {test_df['DateTime'].max()}]")
    print("==================================================")
    
    feature_cols = ['Vehicles', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'Junction']
    target_col = 'Vehicles'
    
    print(f"[PREPROCESS] Feature Columns: {feature_cols}")
    print(f"[PREPROCESS] Target Column: {target_col}")
    
    # 3. Fit MinMaxScaler strictly on training data
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    feature_scaler.fit(train_df[feature_cols])
    target_scaler.fit(train_df[[target_col]])
    
    # Save scalers for inference and evaluation
    joblib.dump(feature_scaler, 'models/feature_scaler.pkl')
    joblib.dump(target_scaler, 'models/target_scaler.pkl')
    print("[PREPROCESS] Scalers saved to models/feature_scaler.pkl and models/target_scaler.pkl")
    
    # 4. Transform datasets
    train_feat_scaled = feature_scaler.transform(train_df[feature_cols])
    train_tgt_scaled = target_scaler.transform(train_df[[target_col]])
    
    val_feat_scaled = feature_scaler.transform(val_df[feature_cols])
    val_tgt_scaled = target_scaler.transform(val_df[[target_col]])
    
    test_feat_scaled = feature_scaler.transform(test_df[feature_cols])
    test_tgt_scaled = target_scaler.transform(test_df[[target_col]])
    
    # 5. Create Sequences (Sliding Window)
    # For time-series sequences across junctions in chronological order:
    X_train, y_train = create_sequences_for_series(train_feat_scaled, train_tgt_scaled, seq_length)
    X_val, y_val = create_sequences_for_series(val_feat_scaled, val_tgt_scaled, seq_length)
    X_test, y_test = create_sequences_for_series(test_feat_scaled, test_tgt_scaled, seq_length)
    
    print("\n--- LSTM Input Sequences (Window Size = 24) ---")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    
    # Save preprocessed sequence arrays
    np.savez_compressed(
        'data/processed/traffic_sequences.npz',
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_cols=feature_cols
    )
    print("[PREPROCESS] Preprocessing successfully completed!\n")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess traffic dataset for LSTM model.")
    parser.add_argument('--raw_path', type=str, default='data/raw/traffic.csv', help='Path to raw traffic CSV')
    parser.add_argument('--processed_path', type=str, default='data/processed/processed_traffic.csv', help='Path to output processed CSV')
    parser.add_argument('--seq_length', type=int, default=SEQUENCE_LENGTH, help='Sequence length for LSTM')
    args = parser.parse_args()
    
    preprocess_pipeline(raw_path=args.raw_path, processed_csv_path=args.processed_path, seq_length=args.seq_length)
