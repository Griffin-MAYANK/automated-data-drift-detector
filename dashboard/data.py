"""Data-access and transformation helpers for the monitoring dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from drift_detector.database import (
    DEFAULT_DATABASE_PATH,
    DATABASE_PATH_ENVIRONMENT_VARIABLE,
)
from drift_detector.repository import DriftRepository


def get_database_path() -> Path:
    """Return the dashboard database path from shared configuration."""

    configured_path = os.getenv(DATABASE_PATH_ENVIRONMENT_VARIABLE)
    return Path(configured_path) if configured_path else DEFAULT_DATABASE_PATH


def get_repository(database_path: str | Path | None = None) -> DriftRepository:
    """Create a repository using the configured or explicit path."""

    return DriftRepository(database_path or get_database_path())


def get_latest_run(repository: DriftRepository) -> dict[str, Any] | None:
    """Return the newest historical run with feature results."""

    runs = repository.list_runs(limit=1)
    return repository.get_run(runs[0]["run_id"]) if runs else None


def get_recent_runs(repository: DriftRepository, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent historical run summaries."""

    return repository.list_runs(limit=limit)


def get_feature_history(repository: DriftRepository, feature: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent results for a selected feature."""

    return repository.get_feature_history(feature=feature, limit=limit)


def run_to_dataframe(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert run dictionaries into a display-friendly DataFrame."""

    columns = [
        "run_id", "created_at", "dataset", "overall_status",
        "alert_count", "investigate_count", "monitor_count",
        "number_of_features",
    ]
    return pd.DataFrame(runs).reindex(columns=columns) if runs else pd.DataFrame(columns=columns)


def features_to_dataframe(features: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert feature results into a concise display DataFrame."""

    columns = [
        "feature", "data_type", "test", "p_value", "adjusted_p_value",
        "magnitude", "normalized_magnitude", "drift_detected", "severity",
        "practical_magnitude", "final_decision", "interpretation",
    ]
    return pd.DataFrame(features).reindex(columns=columns) if features else pd.DataFrame(columns=columns)


def feature_history_to_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert feature history into a chart/table-friendly DataFrame."""

    columns = [
        "run_id", "created_at", "dataset", "feature", "p_value",
        "adjusted_p_value", "magnitude", "normalized_magnitude", "severity",
        "final_decision",
    ]
    return pd.DataFrame(history).reindex(columns=columns) if history else pd.DataFrame(columns=columns)


def build_status_summary(runs: list[dict[str, Any]]) -> dict[str, int]:
    """Count historical outcomes by status."""

    summary = {status: 0 for status in (
        "ALERT", "INVESTIGATE", "MONITOR", "NO_DRIFT", "HEALTHY", "INSUFFICIENT_DATA"
    )}
    for run in runs:
        status = str(run.get("overall_status", ""))
        if status in summary:
            summary[status] += 1
    return summary


def build_status_trend(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Build status counts by run date for a categorical trend chart."""

    if not runs:
        return pd.DataFrame()
    frame = pd.DataFrame(runs)
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    frame = frame.dropna(subset=["created_at"])
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.assign(run_count=1)
        .pivot_table(index="created_at", columns="overall_status", values="run_count", aggfunc="sum", fill_value=0)
        .sort_index()
    )