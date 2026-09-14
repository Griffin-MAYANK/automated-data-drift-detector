"""SQLite database helpers for historical drift detection runs."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


DEFAULT_DATABASE_PATH = Path("reports") / "drift_history.db"
DATABASE_PATH_ENVIRONMENT_VARIABLE = "DRIFT_DETECTOR_DATABASE_PATH"


def get_database_path() -> Path:
    """Return the configured database path or the default report path."""

    configured_path = os.getenv(DATABASE_PATH_ENVIRONMENT_VARIABLE)
    return Path(configured_path) if configured_path else DEFAULT_DATABASE_PATH


def _connect(database_path: str | Path) -> sqlite3.Connection:
    """Open a configured SQLite connection without creating tables."""

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def connection_context(database_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection and close it reliably."""

    connection = _connect(database_path)
    try:
        yield connection
    finally:
        connection.close()


def initialize_database(database_path: str | Path) -> Path:
    """Create the historical database schema if it does not exist."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with connection_context(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS detection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                dataset TEXT NOT NULL,
                split_date TEXT NOT NULL,
                reference_rows INTEGER NOT NULL,
                current_rows INTEGER NOT NULL,
                number_of_features INTEGER NOT NULL,
                overall_status TEXT NOT NULL,
                alert_count INTEGER NOT NULL,
                investigate_count INTEGER NOT NULL,
                monitor_count INTEGER NOT NULL,
                no_drift_count INTEGER NOT NULL,
                insufficient_data_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feature_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                feature TEXT NOT NULL,
                data_type TEXT NOT NULL,
                test TEXT NOT NULL,
                p_value REAL,
                adjusted_p_value REAL,
                magnitude REAL,
                normalized_magnitude REAL,
                drift_detected INTEGER NOT NULL,
                severity TEXT NOT NULL,
                statistical_significance INTEGER NOT NULL,
                fdr_significant INTEGER NOT NULL,
                practical_magnitude TEXT NOT NULL,
                interpretation TEXT NOT NULL,
                final_decision TEXT NOT NULL,
                reference_sample_size INTEGER NOT NULL,
                current_sample_size INTEGER NOT NULL,
                reference_missing_rate REAL NOT NULL,
                current_missing_rate REAL NOT NULL,
                missing_rate_difference REAL NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (run_id)
                    REFERENCES detection_runs(run_id)
                    ON DELETE CASCADE,
                UNIQUE (run_id, feature)
            );

            CREATE INDEX IF NOT EXISTS idx_feature_results_run_id
                ON feature_results(run_id);
            CREATE INDEX IF NOT EXISTS idx_detection_runs_created_at
                ON detection_runs(created_at);
            CREATE INDEX IF NOT EXISTS idx_detection_runs_overall_status
                ON detection_runs(overall_status);
            """
        )
        connection.commit()

    return path