from pathlib import Path

import pandas as pd
import pytest

from drift_detector.api_models import DetectionRequest
import drift_detector.service as service
from drift_detector.service import (
    DatasetNotFoundError,
    InvalidDatasetError,
    run_detection,
)


@pytest.fixture(autouse=True)
def use_temporary_database(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DRIFT_DETECTOR_DATABASE_PATH",
        str(tmp_path / "history.db"),
    )


def write_service_dataset(path: Path) -> Path:
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
            "Country": "France",
        })
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def make_request(dataset: Path, output_directory: Path) -> DetectionRequest:
    return DetectionRequest(
        data_path=str(dataset),
        split_date="2011-07-01",
        significance_threshold=0.05,
        min_samples=5,
        output_directory=str(output_directory),
    )


def test_service_runs_pipeline_and_generates_reports(tmp_path):
    dataset = write_service_dataset(tmp_path / "input.xlsx")
    output_directory = tmp_path / "reports"

    result = run_detection(make_request(dataset, output_directory))

    assert result.dataset == "input.xlsx"
    assert result.reference_rows == 40
    assert result.current_rows == 40
    assert result.number_of_features == 8
    assert result.report_paths.csv == "final_drift_report.csv"
    assert result.report_paths.json_path == "final_drift_report.json"
    assert (output_directory / result.report_paths.csv).exists()
    assert (output_directory / result.report_paths.json_path).exists()
    assert result.run_id.startswith("drift-")
    assert (tmp_path / "history.db").exists()


def test_service_propagates_configuration(tmp_path, monkeypatch):
    dataset = write_service_dataset(tmp_path / "input.xlsx")
    captured = {}
    real_config = service.DriftConfig

    def capture_config(**values):
        captured.update(values)
        return real_config(**values)

    monkeypatch.setattr(service, "DriftConfig", capture_config)
    run_detection(DetectionRequest(
        data_path=str(dataset),
        split_date="2011-07-01",
        significance_threshold=0.1,
        min_samples=7,
        output_directory=str(tmp_path / "reports"),
    ))

    assert captured == {
        "significance_threshold": 0.1,
        "min_samples": 7,
    }


def test_service_rejects_missing_dataset(tmp_path):
    request = make_request(tmp_path / "missing.xlsx", tmp_path / "reports")

    with pytest.raises(DatasetNotFoundError):
        run_detection(request)


def test_service_rejects_unsupported_dataset_type(tmp_path):
    dataset = tmp_path / "input.csv"
    dataset.write_text("not an excel file", encoding="utf-8")

    with pytest.raises(ValueError, match="Only .xlsx"):
        run_detection(make_request(dataset, tmp_path / "reports"))


def test_service_rejects_invalid_raw_dataset(tmp_path):
    dataset = tmp_path / "invalid.xlsx"
    pd.DataFrame({"wrong": [1, 2, 3]}).to_excel(dataset, index=False)

    with pytest.raises(InvalidDatasetError, match="Missing required columns"):
        run_detection(make_request(dataset, tmp_path / "reports"))