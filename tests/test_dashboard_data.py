from dashboard.data import (
    build_status_summary, build_status_trend, feature_history_to_dataframe,
    features_to_dataframe, get_database_path, run_to_dataframe,
)


def sample_run(run_id="drift-1", status="ALERT"):
    return {
        "run_id": run_id, "created_at": "2026-01-01T00:00:00+00:00",
        "dataset": "input.xlsx", "overall_status": status,
        "alert_count": 1, "investigate_count": 0, "monitor_count": 0,
        "number_of_features": 1,
    }


def sample_feature():
    return {
        "run_id": "drift-1", "feature": "month", "data_type": "categorical",
        "test": "Chi-Square Test", "p_value": 0.01, "adjusted_p_value": 0.02,
        "magnitude": 0.5, "normalized_magnitude": None, "drift_detected": True,
        "severity": "HIGH", "practical_magnitude": "LARGE",
        "final_decision": "ALERT", "interpretation": "HIGH_IMPACT",
    }


def test_database_path_respects_environment(monkeypatch, tmp_path):
    path = tmp_path / "history.db"
    monkeypatch.setenv("DRIFT_DETECTOR_DATABASE_PATH", str(path))
    assert get_database_path() == path


def test_empty_transformations_are_safe():
    assert run_to_dataframe([]).empty
    assert features_to_dataframe([]).empty
    assert feature_history_to_dataframe([]).empty
    assert build_status_trend([]).empty
    assert build_status_summary([])["ALERT"] == 0


def test_transformations_and_status_summary():
    runs = [sample_run(), sample_run("drift-2", "MONITOR")]
    feature_frame = features_to_dataframe([sample_feature()])
    history_frame = feature_history_to_dataframe([{**sample_feature(), "created_at": "2026-01-01T00:00:00+00:00", "dataset": "input.xlsx"}])
    assert list(run_to_dataframe(runs)["run_id"]) == ["drift-1", "drift-2"]
    assert feature_frame.iloc[0]["feature"] == "month"
    assert history_frame.iloc[0]["normalized_magnitude"] is None
    assert build_status_summary(runs)["ALERT"] == 1
    assert build_status_summary(runs)["MONITOR"] == 1
    assert not build_status_trend(runs).empty