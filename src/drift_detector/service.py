"""Application service for running the production drift detector."""

from __future__ import annotations

from pathlib import Path
import time
import sqlite3
import uuid
from datetime import datetime, timezone

import pandas as pd

from drift_detector import __version__
from drift_detector.api_models import (
    DetectionRequest,
    DetectionResponse,
    FeatureResult,
    ReportPaths,
)
from drift_detector.config import DriftConfig
from drift_detector.database import get_database_path
from drift_detector.detector import detect_dataset_drift
from drift_detector.features import create_reference_current_features
from drift_detector.logging_config import get_logger
from drift_detector.repository import DriftRepository
from drift_detector.reporting import (
    calculate_operational_summary,
    export_drift_report,
    prepare_json_records,
)


logger = get_logger("service")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the requested dataset path is not a file."""


class UnsupportedDatasetError(ValueError):
    """Raised when the dataset format is not supported."""


class InvalidDatasetError(ValueError):
    """Raised when the dataset cannot be processed by the engine."""


class HistoricalStorageError(RuntimeError):
    """Raised when a successful run cannot be persisted."""


def persist_detection_result(
    request: DetectionRequest,
    data_path: Path,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    summary: dict,
    records: list[dict],
) -> str:
    """Persist report metadata and feature results as one historical run."""

    run_id = f"drift-{uuid.uuid4()}"
    logger.info("generated run_id=%s dataset=%s", run_id, data_path.name)
    run_metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": data_path.name,
        "split_date": request.split_date.isoformat(),
        **{
            key: summary[key]
            for key in (
                "number_of_features",
                "overall_status",
                "alert_count",
                "investigate_count",
                "monitor_count",
                "no_drift_count",
                "insufficient_data_count",
            )
        },
        "reference_rows": len(reference_data),
        "current_rows": len(current_data),
    }
    try:
        DriftRepository(get_database_path()).save_detection_result(
            run=run_metadata,
            features=records,
        )
        logger.info("historical persistence succeeded run_id=%s", run_id)
    except sqlite3.Error as error:
        logger.exception("historical persistence failed run_id=%s", run_id)
        raise HistoricalStorageError("Historical run storage failed.") from error
    return run_id


def _validate_dataset_path(data_path: str) -> Path:
    """Validate and return a supported local Excel dataset path."""

    path = Path(data_path)
    if not path.is_file():
        raise DatasetNotFoundError("Dataset file was not found.")

    if path.suffix.lower() != ".xlsx":
        raise UnsupportedDatasetError("Only .xlsx datasets are supported.")

    return path


def run_detection(request: DetectionRequest) -> DetectionResponse:
    """Run the existing feature, detector, and reporting pipeline."""

    started_at = time.perf_counter()
    try:
        data_path = _validate_dataset_path(request.data_path)
    except (DatasetNotFoundError, UnsupportedDatasetError) as error:
        logger.error(
            "detection failed dataset=%s error=%s",
            Path(request.data_path).name,
            error,
        )
        raise
    logger.info(
        "detection started dataset=%s split_date=%s",
        data_path.name,
        request.split_date,
    )

    try:
        raw_data = pd.read_excel(data_path)
        reference_data, current_data = (
            create_reference_current_features(
                data=raw_data,
                split_date=request.split_date,
            )
        )
        logger.info(
            "dataset processed dataset=%s reference_rows=%d current_rows=%d",
            data_path.name,
            len(reference_data),
            len(current_data),
        )
        config = DriftConfig(
            significance_threshold=request.significance_threshold,
            min_samples=request.min_samples,
        )
        report = detect_dataset_drift(
            reference_data=reference_data,
            current_data=current_data,
            config=config,
        )
        exported = export_drift_report(
            report=report,
            output_directory=request.output_directory,
            dataset_name=data_path.name,
            reference_rows=len(reference_data),
            current_rows=len(current_data),
        )
        logger.info(
            "reports generated dataset=%s csv=%s json=%s",
            data_path.name,
            Path(exported["csv_path"]).name,
            Path(exported["json_path"]).name,
        )
    except (KeyError, TypeError, ValueError) as error:
        logger.error("detection failed dataset=%s error=%s", data_path.name, error)
        detail = str(error)
        if data_path.as_posix() in detail:
            detail = "Dataset content is invalid."
        raise InvalidDatasetError(detail) from error
    except (OSError, ImportError, RuntimeError) as error:
        logger.exception("detection failed dataset=%s", data_path.name)
        raise InvalidDatasetError("Dataset processing failed.") from error

    summary = calculate_operational_summary(report)
    records = prepare_json_records(report)
    run_id = persist_detection_result(
        request=request,
        data_path=data_path,
        reference_data=reference_data,
        current_data=current_data,
        summary=summary,
        records=records,
    )

    logger.info(
        "detection complete dataset=%s features=%d overall_status=%s duration_ms=%.2f",
        data_path.name,
        summary["number_of_features"],
        summary["overall_status"],
        (time.perf_counter() - started_at) * 1000,
    )
    return DetectionResponse(
        run_id=run_id,
        dataset=data_path.name,
        split_date=request.split_date,
        reference_rows=len(reference_data),
        current_rows=len(current_data),
        number_of_features=summary["number_of_features"],
        overall_status=summary["overall_status"],
        alert_count=summary["alert_count"],
        investigate_count=summary["investigate_count"],
        monitor_count=summary["monitor_count"],
        no_drift_count=summary["no_drift_count"],
        insufficient_data_count=summary["insufficient_data_count"],
        report_paths=ReportPaths(
            csv=Path(exported["csv_path"]).name,
            json_path=Path(exported["json_path"]).name,
        ),
        features=[FeatureResult.model_validate(record) for record in records],
    )