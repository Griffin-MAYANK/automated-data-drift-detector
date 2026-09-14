import sqlite3

import pytest

from drift_detector.repository import DriftRepository


def run_record(run_id="drift-test-1", created_at="2026-01-01T00:00:00+00:00"):
    return {
        "run_id": run_id,
        "created_at": created_at,
        "dataset": "input.xlsx",
        "split_date": "2026-01-01",
        "reference_rows": 40,
        "current_rows": 40,
        "number_of_features": 1,
        "overall_status": "ALERT",
        "alert_count": 1,
        "investigate_count": 0,
        "monitor_count": 0,
        "no_drift_count": 0,
        "insufficient_data_count": 0,
    }


def feature_record(feature="month"):
    return {
        "feature": feature,
        "data_type": "categorical",
        "test": "Chi-Square Test",
        "p_value": 0.01,
        "adjusted_p_value": 0.02,
        "magnitude": 0.5,
        "normalized_magnitude": None,
        "drift_detected": True,
        "severity": "HIGH",
        "statistical_significance": True,
        "fdr_significant": True,
        "practical_magnitude": "LARGE",
        "interpretation": "HIGH_IMPACT",
        "final_decision": "ALERT",
        "reference_sample_size": 40,
        "current_sample_size": 40,
        "reference_missing_rate": 0.0,
        "current_missing_rate": 0.0,
        "missing_rate_difference": 0.0,
        "status": "OK",
    }


def test_repository_saves_and_retrieves_complete_run(tmp_path):
    repository = DriftRepository(tmp_path / "history.db")
    repository.save_detection_result(run_record(), [feature_record()])

    run = repository.get_run("drift-test-1")

    assert run["dataset"] == "input.xlsx"
    assert run["features"][0]["feature"] == "month"
    assert repository.count_runs() == 1
    assert repository.list_runs(limit=20)[0]["run_id"] == "drift-test-1"
    assert repository.get_feature_history("month")[0]["run_id"] == "drift-test-1"


def test_repository_can_save_run_and_features_separately(tmp_path):
    repository = DriftRepository(tmp_path / "history.db")
    repository.initialize()
    repository.save_detection_run(run_record())
    repository.save_feature_results("drift-test-1", [feature_record()])

    assert repository.get_run("drift-test-1")["features"][0]["feature"] == "month"


def test_repository_limit_and_missing_run(tmp_path):
    repository = DriftRepository(tmp_path / "history.db")
    repository.save_detection_result(run_record(), [feature_record()])
    repository.save_detection_result(
        run_record("drift-test-2", "2026-01-02T00:00:00+00:00"),
        [feature_record()],
    )

    assert len(repository.list_runs(limit=1)) == 1
    assert repository.get_run("missing") is None


def test_repository_rolls_back_run_when_feature_insert_fails(tmp_path):
    repository = DriftRepository(tmp_path / "history.db")
    duplicate_features = [feature_record(), feature_record()]

    with pytest.raises(sqlite3.IntegrityError):
        repository.save_detection_result(run_record(), duplicate_features)

    assert repository.count_runs() == 0


def test_repository_enforces_unique_run_and_feature_constraints(tmp_path):
    repository = DriftRepository(tmp_path / "history.db")
    repository.save_detection_result(run_record(), [feature_record()])

    with pytest.raises(sqlite3.IntegrityError):
        repository.save_detection_result(run_record(), [feature_record()])


def test_repository_delete_cascades_feature_results(tmp_path):
    repository = DriftRepository(tmp_path / "history.db")
    repository.save_detection_result(run_record(), [feature_record()])

    assert repository.delete_run("drift-test-1") is True
    assert repository.get_run("drift-test-1") is None
    assert repository.get_feature_history("month") == []
    assert repository.delete_run("missing") is False