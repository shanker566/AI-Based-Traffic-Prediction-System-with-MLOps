import os
import json
import pytest

from monitoring.drift_report import generate_drift_report

def test_drift_report_generation(tmp_path):
    """Verify Evidently drift report generates valid HTML and summary JSON."""
    test_html = str(tmp_path / "test_drift.html")
    test_json = str(tmp_path / "test_summary.json")

    summary = generate_drift_report(output_html=test_html, output_json=test_json)

    assert os.path.exists(test_html)
    assert os.path.getsize(test_html) > 1000
    assert os.path.exists(test_json)

    with open(test_json, "r") as f:
        data = json.load(f)
    assert "reference_dataset" in data
    assert "current_dataset" in data
    assert "performance_monitoring" in data
