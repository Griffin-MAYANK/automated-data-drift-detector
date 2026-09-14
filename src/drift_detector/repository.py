"""Repository operations for historical drift detection records."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from drift_detector.database import (
    connection_context,
    initialize_database,
)


class DriftRepository:
    """Persist and retrieve drift runs using SQLite."""

    _RUN_COLUMNS = (
        "run_id, created_at, dataset, split_date, reference_rows, "
        "current_rows, number_of_features, overall_status, alert_count, "
        "investigate_count, monitor_count, no_drift_count, "
        "insufficient_data_count"
    )
    _FEATURE_COLUMNS = (
        "run_id, feature, data_type, test, p_value, adjusted_p_value, "
        "magnitude, normalized_magnitude, drift_detected, severity, "
        "statistical_significance, fdr_significant, practical_magnitude, "
        "interpretation, final_decision, reference_sample_size, "
        "current_sample_size, reference_missing_rate, "
        "current_missing_rate, missing_rate_difference, status"
    )

    def __init__(self, database_path: str | Path):
        """Create a repository pointing at a configurable SQLite file."""

        self.database_path = Path(database_path)

    def initialize(self) -> Path:
        """Initialize the database schema without deleting existing data."""

        return initialize_database(self.database_path)

    def save_detection_run(self, run: Mapping[str, Any]) -> None:
        """Save run metadata within the caller's active transaction."""

        values = tuple(run[column] for column in self._RUN_COLUMNS.split(", "))
        placeholders = ", ".join("?" for _ in values)
        with connection_context(self.database_path) as connection:
            connection.execute(
                f"INSERT INTO detection_runs ({self._RUN_COLUMNS}) VALUES ({placeholders})",
                values,
            )
            connection.commit()

    def save_feature_results(
        self,
        run_id: str,
        features: Sequence[Mapping[str, Any]],
    ) -> None:
        """Save feature rows for an existing run."""

        values = [
            tuple(
                {**feature, "run_id": run_id}[column]
                for column in self._FEATURE_COLUMNS.split(", ")
            )
            for feature in features
        ]
        placeholders = ", ".join("?" for _ in self._FEATURE_COLUMNS.split(", "))
        with connection_context(self.database_path) as connection:
            connection.executemany(
                f"INSERT INTO feature_results ({self._FEATURE_COLUMNS}) VALUES ({placeholders})",
                values,
            )
            connection.commit()

    def save_detection_result(
        self,
        run: Mapping[str, Any],
        features: Sequence[Mapping[str, Any]],
    ) -> None:
        """Atomically save one run and all its feature results."""

        run_id = str(run["run_id"])
        self.initialize()
        run_values = tuple(run[column] for column in self._RUN_COLUMNS.split(", "))
        feature_values = [
            tuple(
                {**feature, "run_id": run_id}[column]
                for column in self._FEATURE_COLUMNS.split(", ")
            )
            for feature in features
        ]
        run_placeholders = ", ".join("?" for _ in run_values)
        feature_placeholders = ", ".join("?" for _ in self._FEATURE_COLUMNS.split(", "))

        with connection_context(self.database_path) as connection:
            try:
                connection.execute(
                    f"INSERT INTO detection_runs ({self._RUN_COLUMNS}) VALUES ({run_placeholders})",
                    run_values,
                )
                connection.executemany(
                    f"INSERT INTO feature_results ({self._FEATURE_COLUMNS}) VALUES ({feature_placeholders})",
                    feature_values,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run with its feature results, or ``None``."""

        self.initialize()
        with connection_context(self.database_path) as connection:
            run = connection.execute(
                f"SELECT {self._RUN_COLUMNS} FROM detection_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                return None
            features = connection.execute(
                "SELECT * FROM feature_results WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        result = dict(run)
        result["features"] = [dict(feature) for feature in features]
        return result

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent run metadata, newest first."""

        self.initialize()
        with connection_context(self.database_path) as connection:
            rows = connection.execute(
                f"SELECT {self._RUN_COLUMNS} FROM detection_runs "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_feature_history(
        self,
        feature: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return recent results for one feature, newest first."""

        self.initialize()
        with connection_context(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT feature_results.*, detection_runs.created_at, detection_runs.dataset
                FROM feature_results
                JOIN detection_runs ON detection_runs.run_id = feature_results.run_id
                WHERE feature_results.feature = ?
                ORDER BY detection_runs.created_at DESC, feature_results.id DESC
                LIMIT ?
                """,
                (feature, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_runs(self) -> int:
        """Return the number of stored detection runs."""

        self.initialize()
        with connection_context(self.database_path) as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM detection_runs").fetchone()
        return int(row["count"])

    def delete_run(self, run_id: str) -> bool:
        """Delete one run and its feature rows, returning whether it existed."""

        self.initialize()
        with connection_context(self.database_path) as connection:
            cursor = connection.execute(
                "DELETE FROM detection_runs WHERE run_id = ?",
                (run_id,),
            )
            connection.commit()
        return cursor.rowcount > 0
