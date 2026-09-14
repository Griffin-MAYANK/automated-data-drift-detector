import sqlite3

from drift_detector.database import connection_context, initialize_database


def test_initialize_database_creates_schema_and_indexes(tmp_path):
    database_path = tmp_path / "history.db"

    initialize_database(database_path)

    assert database_path.exists()
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert {"detection_runs", "feature_results"} <= tables
    assert {
        "idx_feature_results_run_id",
        "idx_detection_runs_created_at",
        "idx_detection_runs_overall_status",
    } <= indexes
    assert foreign_keys == 0


def test_initialize_database_is_repeatable_and_preserves_data(tmp_path):
    database_path = tmp_path / "nested" / "history.db"

    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO detection_runs "
            "(run_id, created_at, dataset, split_date, reference_rows, "
            "current_rows, number_of_features, overall_status, alert_count, "
            "investigate_count, monitor_count, no_drift_count, "
            "insufficient_data_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run-1", "2026-01-01", "data.xlsx", "2026-01-01", 1, 1, 0, "HEALTHY", 0, 0, 0, 0, 0),
        )
        connection.commit()

    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM detection_runs").fetchone()[0]

    assert count == 1


def test_foreign_keys_are_enabled_per_repository_connection(tmp_path):
    database_path = tmp_path / "history.db"
    initialize_database(database_path)

    with connection_context(database_path) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1