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

def test_junctions_endpoint(client):
    """Test GET /junctions returns stats for all monitored junctions."""
    response = client.get("/junctions")
    assert response.status_code == 200
    data = response.json()
    for junc_id in [1, 2, 3, 4]:
        key = f"Junction_{junc_id}"
        assert key in data
        assert "records" in data[key]
        assert data[key]["records"] > 0
        assert "mean_hourly_vehicles" in data[key]

def test_predict_negative_vehicle_count(client):
    """Test rejecting negative vehicle counts in custom sequence."""
    invalid_seq = [10.0] * 23 + [-5.0]
    payload = {
        "target_timestamp": "2026-10-30 12:00:00",
        "junction": 1,
        "mode": "custom_sequence",
        "custom_sequence": invalid_seq
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_historical_no_future_leakage(client):
    """Test that historical mode uses only observations strictly preceding target timestamp."""
    payload = {
        "target_timestamp": "2017-06-30 20:00:00",
        "junction": 1,
        "mode": "historical"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sequence_end"] < "2017-06-30 20:00:00"
    assert data["sequence_end"] == "2017-06-30 19:00:00"
    assert data["sequence_start"] == "2017-06-29 20:00:00"
