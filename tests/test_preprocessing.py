import os
import numpy as np
import pandas as pd
import joblib
import pytest

from src.data_preprocessing import clean_and_engineer_features, create_sequences_for_series, SEQUENCE_LENGTH

def test_feature_engineering_columns():
    """Verify clean_and_engineer_features creates expected temporal columns."""
    sample_df = pd.DataFrame({
        "DateTime": ["2017-01-01 00:00:00", "2017-01-01 01:00:00"],
        "Junction": [1, 1],
        "Vehicles": [10, 15],
        "ID": [1, 2]
    })
    cleaned = clean_and_engineer_features(sample_df)
    for col in ["Hour", "DayOfWeek", "Month", "Day", "IsWeekend"]:
        assert col in cleaned.columns

def test_sliding_window_sequence_shape():
    """Verify sliding window sequences produce exact (N - L, L, F) shape."""
    features = np.random.rand(50, 6)
    target = np.random.rand(50, 1)
    seq_length = 24

    X, y = create_sequences_for_series(features, target, seq_length=seq_length)
    assert X.shape == (26, 24, 6)
    assert y.shape == (26, 1)

def test_scaler_files_consistency():
    """Verify saved scaler artifacts match the required feature dimensions."""
    feature_scaler_path = "models/feature_scaler.pkl"
    target_scaler_path = "models/target_scaler.pkl"

    assert os.path.exists(feature_scaler_path)
    assert os.path.exists(target_scaler_path)

    f_scaler = joblib.load(feature_scaler_path)
    t_scaler = joblib.load(target_scaler_path)

    assert f_scaler.n_features_in_ == 6
    assert t_scaler.n_features_in_ == 1
