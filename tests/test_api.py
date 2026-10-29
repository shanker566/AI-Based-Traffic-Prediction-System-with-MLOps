import pytest

def test_health_endpoint(client):
    """Test GET /health returns 200 and required fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_scaler_loaded"] is True
    assert data["target_scaler_loaded"] is True
    assert data["dataset_records"] > 0
    assert "model_version" in data

def test_model_info_endpoint(client):
    """Test GET /model-info returns architecture, features, and baseline metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["sequence_length"] == 24
    assert data["target_variable"] == "Vehicles"
    assert data["input_feature_names"] == ['Vehicles', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'Junction']
    assert 1 in data["supported_junctions"]
    assert "baseline_metrics" in data
    assert data["baseline_metrics"]["mae"] > 0

def test_predict_historical_valid(client):
    """Test valid historical prediction returns expected structure and values."""
    payload = {
        "target_timestamp": "2017-06-30 23:00:00",
        "junction": 1,
        "mode": "historical"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["junction"] == 1
    assert data["predicted_vehicles"] >= 0.0
    assert isinstance(data["predicted_vehicles"], float)
    assert data["mode"] == "historical_evaluation"
    assert data["actual_vehicles"] is not None
    assert data["residual_error"] is not None

def test_predict_future_valid(client):
    """Test valid future forecast mode."""
    payload = {
        "target_timestamp": "2017-07-01 06:00:00",
        "junction": 1,
        "mode": "future"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_vehicles"] >= 0.0
    assert data["mode"] == "future_forecast"
    assert data["actual_vehicles"] is None
    assert "Recursive" in data["note"] or "recursive" in data["note"]

def test_predict_custom_sequence_valid(client):
    """Test custom 24-step sequence input."""
    custom_seq = [15.0 + (i % 5) for i in range(24)]
    payload = {
        "target_timestamp": "2026-10-30 12:00:00",
        "junction": 2,
        "mode": "custom_sequence",
        "custom_sequence": custom_seq
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_vehicles"] >= 0.0
    assert data["mode"] == "custom_sequence"

def test_predict_invalid_junction(client):
    """Test rejecting invalid junction identifier."""
    payload = {
        "target_timestamp": "2017-06-30 23:00:00",
        "junction": 99,
        "mode": "historical"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]

def test_predict_invalid_sequence_length(client):
    """Test rejecting custom sequence with incorrect length (e.g. 10 instead of 24)."""
    payload = {
        "target_timestamp": "2026-10-30 12:00:00",
        "junction": 1,
        "mode": "custom_sequence",
        "custom_sequence": [10.0, 12.0, 15.0]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_timestamp_format(client):
    """Test rejecting invalid timestamp format string."""
    payload = {
        "target_timestamp": "not-a-valid-date",
        "junction": 1,
        "mode": "historical"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
