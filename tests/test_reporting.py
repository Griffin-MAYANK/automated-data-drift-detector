from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd

from drift_detector.reporting import (
    calculate_operational_summary,
    prepare_json_records,
    export_drift_report,
    load_exported_report,
)


def create_sample_report():
    """
    Create a representative detector-style report
    for testing the reporting layer.
    """

    return pd.DataFrame([
        {
            "feature": "age",
            "feature_type": "numerical",
            "reference_samples": 1000,
            "current_samples": 1200,
            "reference_missing_rate": 0.01,
            "current_missing_rate": 0.02,
            "missing_rate_difference": 0.01,
            "test": "KS",
            "p_value": 0.0001,
            "adjusted_p_value": 0.0003,
            "statistical_drift": True,
            "magnitude": 1.5,
            "normalized_magnitude": 0.15,
            "severity": "MEDIUM",
            "practical_magnitude": "MEDIUM",
            "interpretation": "ACTIONABLE",
            "final_decision": "INVESTIGATE",
            "priority": 2,
        },
        {
            "feature": "income",
            "feature_type": "numerical",
            "reference_samples": 1000,
            "current_samples": 1200,
            "reference_missing_rate": 0.00,
            "current_missing_rate": 0.01,
            "missing_rate_difference": 0.01,
            "test": "KS",
            "p_value": 0.00001,
            "adjusted_p_value": 0.00002,
            "statistical_drift": True,
            "magnitude": 8.0,
            "normalized_magnitude": 0.80,
            "severity": "HIGH",
            "practical_magnitude": "HIGH",
            "interpretation": "HIGH_IMPACT",
            "final_decision": "ALERT",
            "priority": 1,
        },
        {
            "feature": "country",
            "feature_type": "categorical",
            "reference_samples": 1000,
            "current_samples": 1200,
            "reference_missing_rate": 0.00,
            "current_missing_rate": 0.00,
            "missing_rate_difference": 0.00,
            "test": "CHI2",
            "p_value": 0.70,
            "adjusted_p_value": 0.70,
            "statistical_drift": False,
            "magnitude": 0.01,
            "normalized_magnitude": np.nan,
            "severity": "NO_DRIFT",
            "practical_magnitude": "VERY_SMALL",
            "interpretation": "NO_DRIFT",
            "final_decision": "NO_DRIFT",
            "priority": 4,
        },
        {
            "feature": "quantity",
            "feature_type": "numerical",
            "reference_samples": 20,
            "current_samples": 1200,
            "reference_missing_rate": 0.00,
            "current_missing_rate": 0.00,
            "missing_rate_difference": 0.00,
            "test": "KS",
            "p_value": np.nan,
            "adjusted_p_value": np.nan,
            "statistical_drift": False,
            "magnitude": np.nan,
            "normalized_magnitude": np.nan,
            "severity": "INSUFFICIENT_DATA",
            "practical_magnitude": "INSUFFICIENT_DATA",
            "interpretation": "INSUFFICIENT_DATA",
            "final_decision": "INSUFFICIENT_DATA",
            "priority": 5,
        },
    ])


def test_operational_summary():
    report = create_sample_report()

    summary = calculate_operational_summary(report)

    assert isinstance(summary, dict)

    assert summary["number_of_features"] == 4

    assert summary["alert_count"] == 1
    assert summary["investigate_count"] == 1
    assert summary["monitor_count"] == 0
    assert summary["no_drift_count"] == 1
    assert summary["insufficient_data_count"] == 1

    assert summary["overall_status"] == "ALERT"

    print("✓ test_operational_summary")


def test_summary_status_priority():
    """
    ALERT should dominate all lower-priority statuses.
    """

    report = create_sample_report()

    summary = calculate_operational_summary(report)

    assert summary["overall_status"] == "ALERT"

    report_without_alert = report[
        report["final_decision"] != "ALERT"
    ].copy()

    summary_without_alert = calculate_operational_summary(
        report_without_alert
    )

    assert summary_without_alert["overall_status"] == "INVESTIGATE"

    print("✓ test_summary_status_priority")


def test_summary_with_monitor_only():
    report = pd.DataFrame([
        {
            "feature": "age",
            "final_decision": "MONITOR",
        },
        {
            "feature": "income",
            "final_decision": "MONITOR",
        },
    ])

    summary = calculate_operational_summary(report)

    assert summary["number_of_features"] == 2
    assert summary["monitor_count"] == 2
    assert summary["overall_status"] == "MONITOR"

    print("✓ test_summary_with_monitor_only")


def test_summary_with_no_drift_only():
    report = pd.DataFrame([
        {
            "feature": "age",
            "final_decision": "NO_DRIFT",
        },
        {
            "feature": "income",
            "final_decision": "NO_DRIFT",
        },
    ])

    summary = calculate_operational_summary(report)

    assert summary["number_of_features"] == 2
    assert summary["no_drift_count"] == 2
    assert summary["overall_status"] == "HEALTHY"

    print("✓ test_summary_with_no_drift_only")


def test_summary_with_insufficient_data_only():
    report = pd.DataFrame([
        {
            "feature": "age",
            "final_decision": "INSUFFICIENT_DATA",
        },
        {
            "feature": "income",
            "final_decision": "INSUFFICIENT_DATA",
        },
    ])

    summary = calculate_operational_summary(report)

    assert summary["number_of_features"] == 2
    assert summary["insufficient_data_count"] == 2
    assert summary["overall_status"] == "INSUFFICIENT_DATA"

    print("✓ test_summary_with_insufficient_data_only")


def test_empty_report_rejected():
    empty_report = pd.DataFrame()

    try:
        calculate_operational_summary(empty_report)
        assert False, "Empty report should raise an exception"
    except ValueError:
        pass

    print("✓ test_empty_report_rejected")


def test_json_conversion():
    report = create_sample_report()

    records = prepare_json_records(report)

    assert isinstance(records, list)
    assert len(records) == 4

    for record in records:
        assert isinstance(record, dict)

    # Verify NaN values were converted to None.
    for record in records:
        for value in record.values():
            assert not (
                isinstance(value, float)
                and np.isnan(value)
            )

    print("✓ test_json_conversion")


def test_json_records_preserve_values():
    report = create_sample_report()

    records = prepare_json_records(report)

    age_record = next(
        record
        for record in records
        if record["feature"] == "age"
    )

    assert age_record["feature_type"] == "numerical"
    assert age_record["final_decision"] == "INVESTIGATE"
    assert age_record["priority"] == 2

    print("✓ test_json_records_preserve_values")


def test_export_csv_and_json():
    report = create_sample_report()

    temp_dir = Path(tempfile.mkdtemp())

    try:
        result = export_drift_report(
            report,
            output_directory=temp_dir,
            dataset_name="Test Dataset",
            reference_rows=1000,
            current_rows=1200,
        )

        csv_path = temp_dir / "final_drift_report.csv"
        json_path = temp_dir / "final_drift_report.json"

        assert csv_path.exists()
        assert json_path.exists()

        assert result["csv_path"] == csv_path
        assert result["json_path"] == json_path

        assert result["csv_size_bytes"] > 0
        assert result["json_size_bytes"] > 0

        assert result["summary"]["number_of_features"] == 4
        assert result["summary"]["overall_status"] == "ALERT"

        print("✓ test_export_csv_and_json")

    finally:
        shutil.rmtree(temp_dir)


def test_exported_csv_can_be_loaded():
    report = create_sample_report()

    temp_dir = Path(tempfile.mkdtemp())

    try:
        export_drift_report(
            report,
            output_directory=temp_dir,
        )

        csv_path = temp_dir / "final_drift_report.csv"
        json_path = temp_dir / "final_drift_report.json"

        loaded = load_exported_report(
            csv_path,
            json_path,
        )

        loaded_csv = loaded["csv"]
        loaded_json = loaded["json"]

        assert isinstance(loaded_csv, pd.DataFrame)
        assert isinstance(loaded_json, dict)

        assert len(loaded_csv) == 4

        assert "feature" in loaded_csv.columns
        assert "final_decision" in loaded_csv.columns

        print("✓ test_exported_csv_can_be_loaded")

    finally:
        shutil.rmtree(temp_dir)


def test_exported_json_content():
    report = create_sample_report()

    temp_dir = Path(tempfile.mkdtemp())

    try:
        export_drift_report(
            report,
            output_directory=temp_dir,
            dataset_name="Test Dataset",
            reference_rows=1000,
            current_rows=1200,
        )

        csv_path = temp_dir / "final_drift_report.csv"
        json_path = temp_dir / "final_drift_report.json"

        loaded = load_exported_report(
            csv_path,
            json_path,
        )

        json_report = loaded["json"]

        assert isinstance(json_report, dict)

        assert json_report["report_type"] == (
            "Automated Data Drift Detection"
        )

        assert json_report["dataset"] == "Test Dataset"

        assert json_report["reference_rows"] == 1000
        assert json_report["current_rows"] == 1200

        assert json_report["number_of_features"] == 4
        assert json_report["alert_count"] == 1
        assert json_report["investigate_count"] == 1
        assert json_report["monitor_count"] == 0
        assert json_report["no_drift_count"] == 1
        assert json_report["insufficient_data_count"] == 1

        assert json_report["overall_status"] == "ALERT"

        assert isinstance(
            json_report["features"],
            list,
        )

        assert len(json_report["features"]) == 4

        print("✓ test_exported_json_content")

    finally:
        shutil.rmtree(temp_dir)


def test_round_trip_feature_count():
    report = create_sample_report()

    temp_dir = Path(tempfile.mkdtemp())

    try:
        export_drift_report(
            report,
            output_directory=temp_dir,
        )

        csv_path = temp_dir / "final_drift_report.csv"
        json_path = temp_dir / "final_drift_report.json"

        loaded = load_exported_report(
            csv_path,
            json_path,
        )

        loaded_csv = loaded["csv"]
        loaded_json = loaded["json"]

        assert len(loaded_csv) == len(report)
        assert len(loaded_json["features"]) == len(report)

        print("✓ test_round_trip_feature_count")

    finally:
        shutil.rmtree(temp_dir)


def test_export_rejects_empty_report():
    empty_report = pd.DataFrame()

    temp_dir = Path(tempfile.mkdtemp())

    try:
        try:
            export_drift_report(
                empty_report,
                output_directory=temp_dir,
            )
            assert False, "Empty report should raise an exception"
        except ValueError:
            pass

        print("✓ test_export_rejects_empty_report")

    finally:
        shutil.rmtree(temp_dir)


def test_load_missing_files():
    temp_dir = Path(tempfile.mkdtemp())

    try:
        csv_path = temp_dir / "missing.csv"
        json_path = temp_dir / "missing.json"

        try:
            load_exported_report(
                csv_path,
                json_path,
            )
            assert False, "Missing files should raise an exception"
        except FileNotFoundError:
            pass

        print("✓ test_load_missing_files")

    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():

    tests = [
        test_operational_summary,
        test_summary_status_priority,
        test_summary_with_monitor_only,
        test_summary_with_no_drift_only,
        test_summary_with_insufficient_data_only,
        test_empty_report_rejected,
        test_json_conversion,
        test_json_records_preserve_values,
        test_export_csv_and_json,
        test_exported_csv_can_be_loaded,
        test_exported_json_content,
        test_round_trip_feature_count,
        test_export_rejects_empty_report,
        test_load_missing_files,
    ]

    print("=" * 100)
    print("REPORTING LAYER UNIT TESTS")
    print("=" * 100)

    passed = 0

    for test in tests:
        try:
            test()
            passed += 1

        except Exception as error:
            print(f"✗ {test.__name__}")
            print(f"  Error: {error}")

    print()
    print("=" * 100)

    if passed == len(tests):

        print(
            f"ALL {passed} REPORTING TESTS PASSED"
        )

        print("=" * 100)

    else:

        print(
            f"{passed}/{len(tests)} REPORTING TESTS PASSED"
        )

        print("=" * 100)

        raise SystemExit(1)


if __name__ == "__main__":
    run_all_tests()