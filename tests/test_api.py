from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from drift_detector.api import app


client = TestClient(app)


def write_api_dataset(path: Path) -> Path:
    rows = []
    for index in range(40):
        rows.append({
            "InvoiceNo": str(10000 + index),
            "Quantity": 1 + index % 3,
            "InvoiceDate": "2011-06-01 12:00:00",
            "UnitPrice": 10.0 + index % 4,
            "Country": "United Kingdom",
        })
    for index in range(40):
        rows.append({
            "InvoiceNo": str(20000 + index),
            "Quantity": 2 + index % 3,
            "InvoiceDate": "2011-08-01 12:00:00",
            "UnitPrice": 11.0 + index % 4,
            "Country": "United Kingdom",
        })
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def detect_payload(dataset: Path, output_directory: Path) -> dict:
    return {
        "data_path": str(dataset),
        "split_date": "2011-07-01",
        "min_samples": 5,
        "output_directory": str(output_directory),
    }


def test_health_returns_healthy():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_metadata_returns_expected_fields():
    response = client.get("/metadata")
    body = response.json()

    assert response.status_code == 200
    assert body["project_name"] == "automated-data-drift-detector"
    assert body["api_version"] == "1.0.0"
    assert body["detector_version"] == "1.0.0"
    assert "numerical" in body["supported_feature_types"]
    assert "month" in body["monitoring_features"]


def test_detect_requires_required_fields():
    response = client.post("/detect", json={})

    assert response.status_code == 422


def test_detect_rejects_invalid_parameters():
    base = {
        "data_path": "input.xlsx",
        "split_date": "2011-07-01",
    }

    assert client.post(
        "/detect",
        json={**base, "significance_threshold": 1},
    ).status_code == 422
    assert client.post(
        "/detect",
        json={**base, "min_samples": 0},
    ).status_code == 422
    assert client.post(
        "/detect",
        json={**base, "split_date": "not-a-date"},
    ).status_code == 422


def test_detect_missing_dataset_returns_404(tmp_path):
    response = client.post(
        "/detect",
        json=detect_payload(tmp_path / "missing.xlsx", tmp_path / "reports"),
    )

    assert response.status_code == 404
    assert "traceback" not in response.text.lower()


def test_detect_invalid_dataset_returns_safe_400(tmp_path):
    dataset = tmp_path / "invalid.xlsx"
    pd.DataFrame({"unexpected": [1, 2]}).to_excel(dataset, index=False)

    response = client.post(
        "/detect",
        json=detect_payload(dataset, tmp_path / "reports"),
    )

    assert response.status_code == 400
    assert "Traceback" not in response.text
    assert str(tmp_path) not in response.text


def test_detect_success_returns_summary_and_features(tmp_path):
    dataset = write_api_dataset(tmp_path / "input.xlsx")
    output_directory = tmp_path / "reports"

    response = client.post(
        "/detect",
        json=detect_payload(dataset, output_directory),
    )
    body = response.json()

    assert response.status_code == 200
    assert body["dataset"] == "input.xlsx"
    assert body["reference_rows"] == 40
    assert body["current_rows"] == 40
    assert body["number_of_features"] == 8
    assert body["overall_status"] in {
        "ALERT",
        "INVESTIGATE",
        "MONITOR",
        "NO_DRIFT",
        "INSUFFICIENT_DATA",
    }
    assert set(body) >= {
        "alert_count",
        "investigate_count",
        "monitor_count",
        "no_drift_count",
        "insufficient_data_count",
        "report_paths",
        "features",
    }
    assert len(body["features"]) == 8
    assert body["report_paths"] == {
        "csv": "final_drift_report.csv",
        "json": "final_drift_report.json",
    }