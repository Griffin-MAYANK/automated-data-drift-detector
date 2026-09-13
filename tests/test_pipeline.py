"""
End-to-end validation of the production
Automated Data Drift Detector.

This test:

1. Loads the real Online Retail dataset.
2. Creates reference/current datasets using
   the production feature module.
3. Creates monitoring features.
4. Runs the production detector.
5. Verifies the expected feature set.
6. Verifies important operational decisions.
7. Exports CSV and JSON reports.
8. Reloads the exported reports.
9. Verifies the exported files.

This test intentionally exercises the actual
production modules rather than duplicating their
implementation.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
from drift_detector.config import DEFAULT_CONFIG
from drift_detector.detector import detect_dataset_drift
from drift_detector.features import (
    MONITORING_FEATURES,
    create_reference_current_features,
)
from drift_detector.reporting import (
    export_drift_report,
    load_exported_report,
)


# ------------------------------------------------
# Dataset paths
# ------------------------------------------------

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "Online Retail.xlsx"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
)


# ------------------------------------------------
# Main end-to-end test
# ------------------------------------------------

def main():

    print("=" * 100)
    print("AUTOMATED DATA DRIFT DETECTOR")
    print("END-TO-END PRODUCTION TEST")
    print("=" * 100)

    # --------------------------------------------
    # 1. Check dataset
    # --------------------------------------------

    print("\n1. CHECKING DATASET")
    print("-" * 100)

    assert DATASET_PATH.exists(), (
        f"Dataset not found: {DATASET_PATH}"
    )

    print("✓ Dataset found:")
    print(DATASET_PATH)

    # --------------------------------------------
    # 2. Load dataset
    # --------------------------------------------

    print("\n2. LOADING DATASET")
    print("-" * 100)

    real_data = pd.read_excel(
        DATASET_PATH
    )

    assert not real_data.empty

    print("✓ Dataset loaded")
    print(
        "Rows:",
        len(real_data),
    )

    print(
        "Columns:",
        len(real_data.columns),
    )

    # --------------------------------------------
    # 3. Create reference/current features
    #    using production feature engineering
    # --------------------------------------------

    print(
        "\n3. CREATING REFERENCE/CURRENT "
        "MONITORING DATA"
    )
    print("-" * 100)

    split_date = pd.Timestamp(
        "2011-07-01"
    )

    (
        reference_monitoring_data,
        current_monitoring_data,
    ) = create_reference_current_features(
        data=real_data,
        split_date=split_date,
    )

    assert not reference_monitoring_data.empty

    assert not current_monitoring_data.empty

    print(
        "✓ Reference rows:",
        len(reference_monitoring_data),
    )

    print(
        "✓ Current rows:",
        len(current_monitoring_data),
    )

    # --------------------------------------------
    # 4. Verify production feature set
    # --------------------------------------------

    print(
        "\n4. VERIFYING MONITORING FEATURES"
    )
    print("-" * 100)

    assert (
        list(reference_monitoring_data.columns)
        == MONITORING_FEATURES
    )

    assert (
        list(current_monitoring_data.columns)
        == MONITORING_FEATURES
    )

    assert (
        reference_monitoring_data.columns.tolist()
        == current_monitoring_data.columns.tolist()
    )

    print("✓ Production feature engineering used")

    print(
        "Features:",
        list(
            reference_monitoring_data.columns
        ),
    )

    # --------------------------------------------
    # 5. Run production detector
    # --------------------------------------------

    print(
        "\n5. RUNNING PRODUCTION DETECTOR"
    )
    print("-" * 100)

    report = detect_dataset_drift(
        reference_data=reference_monitoring_data,
        current_data=current_monitoring_data,
        config=DEFAULT_CONFIG,
    )

    assert not report.empty

    assert len(report) == 8

    print("✓ Drift detection completed")

    print("\nFeature decisions:")

    print(
        report[
            [
                "feature",
                "final_decision",
                "practical_magnitude",
                "interpretation",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------
    # 6. Verify feature coverage
    # --------------------------------------------

    print(
        "\n6. VERIFYING FEATURE COVERAGE"
    )
    print("-" * 100)

    detected_features = set(
        report["feature"]
    )

    expected_features = set(
        MONITORING_FEATURES
    )

    assert (
        detected_features
        == expected_features
    )

    print(
        "✓ All expected features were analyzed"
    )

    # --------------------------------------------
    # 7. Verify month result
    # --------------------------------------------

    print(
        "\n7. VERIFYING MONTH DRIFT"
    )
    print("-" * 100)

    month_result = report[
        report["feature"] == "month"
    ]

    assert len(month_result) == 1

    month_decision = (
        month_result[
            "final_decision"
        ].iloc[0]
    )

    assert month_decision == "ALERT", (
        "Expected month to produce ALERT."
    )

    print(
        "✓ month correctly classified as ALERT"
    )

    # --------------------------------------------
    # 8. Verify cancellation result
    # --------------------------------------------

    print(
        "\n8. VERIFYING CANCELLATION DRIFT"
    )
    print("-" * 100)

    cancellation_result = report[
        report["feature"]
        == "is_cancelled"
    ]

    assert len(cancellation_result) == 1

    cancellation_decision = (
        cancellation_result[
            "final_decision"
        ].iloc[0]
    )

    assert cancellation_decision == "MONITOR", (
        "Expected is_cancelled to produce MONITOR."
    )

    print(
        "✓ is_cancelled correctly classified as MONITOR"
    )

    # --------------------------------------------
    # 9. Export reports
    # --------------------------------------------

    print(
        "\n9. EXPORTING REPORTS"
    )
    print("-" * 100)

    export_result = export_drift_report(
        report=report,
        output_directory=REPORT_DIRECTORY,
        dataset_name="UCI Online Retail",
        reference_rows=len(
            reference_monitoring_data
        ),
        current_rows=len(
            current_monitoring_data
        ),
    )

    csv_path = export_result[
        "csv_path"
    ]

    json_path = export_result[
        "json_path"
    ]

    assert csv_path.exists()

    assert json_path.exists()

    print(
        "✓ CSV created:",
        csv_path,
    )

    print(
        "✓ JSON created:",
        json_path,
    )

    # --------------------------------------------
    # 10. Verify operational summary
    # --------------------------------------------

    print(
        "\n10. VERIFYING OPERATIONAL SUMMARY"
    )
    print("-" * 100)

    summary = export_result[
        "summary"
    ]

    assert (
        summary["number_of_features"]
        == 8
    )

    assert (
        summary["overall_status"]
        == "ALERT"
    )

    assert (
        summary["alert_count"]
        >= 1
    )

    print(
        "✓ Overall status:",
        summary["overall_status"],
    )

    print(
        "✓ Alert count:",
        summary["alert_count"],
    )

    print(
        "✓ Features analyzed:",
        summary["number_of_features"],
    )

    # --------------------------------------------
    # 11. Reload exported reports
    # --------------------------------------------

    print(
        "\n11. RELOADING EXPORTED REPORTS"
    )
    print("-" * 100)

    loaded_reports = load_exported_report(
        csv_path=csv_path,
        json_path=json_path,
    )

    exported_csv = loaded_reports[
        "csv"
    ]

    exported_json = loaded_reports[
        "json"
    ]

    assert isinstance(
        exported_csv,
        pd.DataFrame,
    )

    assert isinstance(
        exported_json,
        dict,
    )

    assert len(
        exported_csv
    ) == len(report)

    assert (
        exported_json["dataset"]
        == "UCI Online Retail"
    )

    assert (
        exported_json["number_of_features"]
        == 8
    )

    assert (
        exported_json["overall_status"]
        == "ALERT"
    )

    assert len(
        exported_json["features"]
    ) == 8

    print(
        "✓ CSV successfully reloaded"
    )

    print(
        "✓ JSON successfully reloaded"
    )

    # --------------------------------------------
    # 12. Final result
    # --------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "ALL END-TO-END TESTS PASSED"
    )

    print(
        "=" * 100
    )

    print(
        "\nProduction pipeline verified:"
    )

    print("✓ Real dataset loading")
    print("✓ Production feature engineering")
    print("✓ Reference/current splitting")
    print("✓ Dataset validation")
    print("✓ Statistical drift detection")
    print("✓ Practical magnitude analysis")
    print("✓ FDR correction")
    print("✓ Operational decisions")
    print("✓ CSV export")
    print("✓ JSON export")
    print("✓ Report reload")

    print(
        "\n" + "=" * 100
    )


if __name__ == "__main__":
    main()
