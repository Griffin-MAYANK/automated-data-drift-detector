"""
Unit tests for the core automated data drift detector.

These tests verify:

1. Numerical drift detection
2. Categorical drift detection
3. Practical magnitude classification
4. Drift interpretation
5. FDR correction
6. Final operational decisions
7. Complete dataset-level detection
8. Feature ordering
9. Configuration handling
10. Invalid input handling
"""

import numpy as np
import pandas as pd

from drift_detector.config import (
    DriftConfig,
    FeatureType,
    resolve_feature_type,
)
from drift_detector.detector import (
    apply_multiple_testing_correction,
    classify_drift_interpretation,
    classify_practical_magnitude,
    detect_dataset_drift,
    determine_final_decision,
)


# =================================================
# TEST 1 — PRACTICAL MAGNITUDE: NO DRIFT
# =================================================

def test_practical_magnitude_no_drift():

    row = pd.Series({
        "status": "OK",
        "drift_detected": False,
        "data_type": "numerical",
        "normalized_magnitude": 0.01,
        "magnitude": 0.01,
    })

    result = classify_practical_magnitude(
        row
    )

    assert result == "NO_DRIFT"


def test_explicit_metadata_overrides_numeric_dtype():
    data = pd.Series([1, 2, 3])

    assert resolve_feature_type(
        "month",
        data,
    ) is FeatureType.CYCLICAL


def test_unknown_numeric_feature_uses_numeric_fallback():
    data = pd.Series([1.0, 2.0, 3.0])

    assert resolve_feature_type(
        "unconfigured_feature",
        data,
    ) is FeatureType.NUMERICAL


def test_unknown_text_feature_uses_categorical_fallback():
    data = pd.Series(["A", "B", "A"])

    assert resolve_feature_type(
        "unconfigured_feature",
        data,
    ) is FeatureType.CATEGORICAL


def test_cyclical_metadata_uses_categorical_detector_path():
    reference = pd.DataFrame({
        "month": [1, 1, 1, 2, 2, 2] * 10,
    })
    current = pd.DataFrame({
        "month": [7, 7, 7, 8, 8, 8] * 10,
    })

    report = detect_dataset_drift(
        reference,
        current,
        config=DriftConfig(min_samples=5),
    )

    assert report.loc[0, "data_type"] == "categorical"
    assert report.loc[0, "test"] == "Chi-Square Test"


# =================================================
# TEST 2 — PRACTICAL MAGNITUDE: NUMERICAL VERY SMALL
# =================================================

def test_practical_magnitude_numerical_very_small():

    row = pd.Series({
        "status": "OK",
        "drift_detected": True,
        "data_type": "numerical",
        "normalized_magnitude": 0.05,
        "magnitude": 1.0,
    })

    result = classify_practical_magnitude(
        row
    )

    assert result == "VERY_SMALL"


# =================================================
# TEST 3 — PRACTICAL MAGNITUDE: NUMERICAL MODERATE
# =================================================

def test_practical_magnitude_numerical_moderate():

    row = pd.Series({
        "status": "OK",
        "drift_detected": True,
        "data_type": "numerical",
        "normalized_magnitude": 0.20,
        "magnitude": 10.0,
    })

    result = classify_practical_magnitude(
        row
    )

    assert result == "MODERATE"


# =================================================
# TEST 4 — PRACTICAL MAGNITUDE: NUMERICAL LARGE
# =================================================

def test_practical_magnitude_numerical_large():

    row = pd.Series({
        "status": "OK",
        "drift_detected": True,
        "data_type": "numerical",
        "normalized_magnitude": 0.80,
        "magnitude": 50.0,
    })

    result = classify_practical_magnitude(
        row
    )

    assert result == "LARGE"


# =================================================
# TEST 5 — PRACTICAL MAGNITUDE: CATEGORICAL VERY SMALL
# =================================================

def test_practical_magnitude_categorical_very_small():

    row = pd.Series({
        "status": "OK",
        "drift_detected": True,
        "data_type": "categorical",
        "normalized_magnitude": np.nan,
        "magnitude": 0.02,
    })

    result = classify_practical_magnitude(
        row
    )

    assert result == "VERY_SMALL"


# =================================================
# TEST 6 — PRACTICAL MAGNITUDE: CATEGORICAL MODERATE
# =================================================

def test_practical_magnitude_categorical_moderate():

    row = pd.Series({
        "status": "OK",
        "drift_detected": True,
        "data_type": "categorical",
        "normalized_magnitude": np.nan,
        "magnitude": 0.10,
    })

    result = classify_practical_magnitude(
        row
    )

    assert result == "MODERATE"


# =================================================
# TEST 7 — PRACTICAL MAGNITUDE: CATEGORICAL LARGE
# =================================================

def test_practical_magnitude_categorical_large():

    row = pd.Series({
        "status": "OK",
        "drift_detected": True,
        "data_type": "categorical",
        "normalized_magnitude": np.nan,
        "magnitude": 0.50,
    })

    result = classify_practical_magnitude(
        row
    )

    assert result == "LARGE"


# =================================================
# TEST 8 — PRACTICAL MAGNITUDE: INSUFFICIENT DATA
# =================================================

def test_practical_magnitude_insufficient_data():

    row = pd.Series({
        "status": "INSUFFICIENT_DATA",
        "drift_detected": False,
        "data_type": "numerical",
        "normalized_magnitude": np.nan,
        "magnitude": np.nan,
    })

    result = classify_practical_magnitude(
        row
    )

    assert result == "INSUFFICIENT_DATA"


# =================================================
# TEST 9 — INTERPRETATION: NO DRIFT
# =================================================

def test_interpretation_no_drift():

    row = pd.Series({
        "status": "OK",
        "drift_detected": False,
        "practical_magnitude": "NO_DRIFT",
    })

    result = classify_drift_interpretation(
        row
    )

    assert result == "NO_DRIFT"


# =================================================
# TEST 10 — INTERPRETATION: STATISTICAL ONLY
# =================================================

def test_interpretation_statistical_only():

    row = pd.Series({
        "status": "OK",
        "drift_detected": True,
        "practical_magnitude": "VERY_SMALL",
    })

    result = classify_drift_interpretation(
        row
    )

    assert result == "STATISTICAL_ONLY"


# =================================================
# TEST 11 — INTERPRETATION: ACTIONABLE
# =================================================

def test_interpretation_actionable():

    row = pd.Series({
        "status": "OK",
        "drift_detected": True,
        "practical_magnitude": "MODERATE",
    })

    result = classify_drift_interpretation(
        row
    )

    assert result == "ACTIONABLE"


# =================================================
# TEST 12 — INTERPRETATION: HIGH IMPACT
# =================================================

def test_interpretation_high_impact():

    row = pd.Series({
        "status": "OK",
        "drift_detected": True,
        "practical_magnitude": "LARGE",
    })

    result = classify_drift_interpretation(
        row
    )

    assert result == "HIGH_IMPACT"


# =================================================
# TEST 13 — FINAL DECISION: NO DRIFT
# =================================================

def test_final_decision_no_drift():

    row = pd.Series({
        "status": "OK",
        "fdr_significant": False,
        "practical_magnitude": "NO_DRIFT",
    })

    result = determine_final_decision(
        row
    )

    assert result == "NO_DRIFT"


# =================================================
# TEST 14 — FINAL DECISION: MONITOR
# =================================================

def test_final_decision_monitor():

    row = pd.Series({
        "status": "OK",
        "fdr_significant": True,
        "practical_magnitude": "VERY_SMALL",
    })

    result = determine_final_decision(
        row
    )

    assert result == "MONITOR"


# =================================================
# TEST 15 — FINAL DECISION: INVESTIGATE
# =================================================

def test_final_decision_investigate():

    row = pd.Series({
        "status": "OK",
        "fdr_significant": True,
        "practical_magnitude": "MODERATE",
    })

    result = determine_final_decision(
        row
    )

    assert result == "INVESTIGATE"


# =================================================
# TEST 16 — FINAL DECISION: ALERT
# =================================================

def test_final_decision_alert():

    row = pd.Series({
        "status": "OK",
        "fdr_significant": True,
        "practical_magnitude": "LARGE",
    })

    result = determine_final_decision(
        row
    )

    assert result == "ALERT"


# =================================================
# TEST 17 — FINAL DECISION: INSUFFICIENT DATA
# =================================================

def test_final_decision_insufficient_data():

    row = pd.Series({
        "status": "INSUFFICIENT_DATA",
        "fdr_significant": False,
        "practical_magnitude": "INSUFFICIENT_DATA",
    })

    result = determine_final_decision(
        row
    )

    assert result == "INSUFFICIENT_DATA"


# =================================================
# TEST 18 — FDR CORRECTION
# =================================================

def test_fdr_correction():

    report = pd.DataFrame({
        "feature": [
            "feature_a",
            "feature_b",
            "feature_c",
            "feature_d",
        ],
        "p_value": [
            0.001,
            0.002,
            0.30,
            0.80,
        ],
    })

    corrected = (
        apply_multiple_testing_correction(
            report
        )
    )

    assert (
        "adjusted_p_value"
        in corrected.columns
    )

    assert (
        "fdr_significant"
        in corrected.columns
    )

    assert (
        corrected[
            "adjusted_p_value"
        ].notna().all()
    )

    assert (
        corrected.loc[
            corrected["feature"] == "feature_a",
            "fdr_significant",
        ].iloc[0]
        == True
    )

    assert (
        corrected.loc[
            corrected["feature"] == "feature_b",
            "fdr_significant",
        ].iloc[0]
        == True
    )

    assert (
        corrected.loc[
            corrected["feature"] == "feature_c",
            "fdr_significant",
        ].iloc[0]
        == False
    )

    assert (
        corrected.loc[
            corrected["feature"] == "feature_d",
            "fdr_significant",
        ].iloc[0]
        == False
    )


# =================================================
# TEST 19 — FDR WITH MISSING P-VALUE
# =================================================

def test_fdr_with_missing_p_value():

    report = pd.DataFrame({
        "feature": [
            "feature_a",
            "feature_b",
            "feature_c",
        ],
        "p_value": [
            0.001,
            np.nan,
            0.80,
        ],
    })

    corrected = (
        apply_multiple_testing_correction(
            report
        )
    )

    assert (
        corrected.loc[
            corrected["feature"] == "feature_b",
            "adjusted_p_value",
        ].isna().iloc[0]
    )

    assert (
        corrected.loc[
            corrected["feature"] == "feature_b",
            "fdr_significant",
        ].iloc[0]
        == False
    )


# =================================================
# TEST 20 — FDR EMPTY REPORT
# =================================================

def test_fdr_empty_report():

    report = pd.DataFrame({
        "feature": [],
        "p_value": [],
    })

    try:

        apply_multiple_testing_correction(
            report
        )

    except ValueError as error:

        assert (
            "empty report"
            in str(error).lower()
        )

    else:

        raise AssertionError(
            "Expected ValueError was not raised."
        )


# =================================================
# TEST 21 — COMPLETE NUMERICAL DRIFT
# =================================================

def test_complete_numerical_drift():

    np.random.seed(42)

    reference = pd.DataFrame({
        "amount": np.random.normal(
            loc=100,
            scale=10,
            size=1000,
        ),
    })

    current = pd.DataFrame({
        "amount": np.random.normal(
            loc=500,
            scale=10,
            size=1000,
        ),
    })

    report = detect_dataset_drift(
        reference,
        current,
    )

    assert len(report) == 1

    assert (
        report.loc[
            0,
            "feature",
        ]
        == "amount"
    )

    assert (
        report.loc[
            0,
            "drift_detected",
        ]
        == True
    )

    assert (
        report.loc[
            0,
            "fdr_significant",
        ]
        == True
    )

    assert (
        report.loc[
            0,
            "practical_magnitude",
        ]
        == "LARGE"
    )

    assert (
        report.loc[
            0,
            "interpretation",
        ]
        == "HIGH_IMPACT"
    )

    assert (
        report.loc[
            0,
            "final_decision",
        ]
        == "ALERT"
    )


# =================================================
# TEST 22 — COMPLETE CATEGORICAL DRIFT
# =================================================

def test_complete_categorical_drift():

    reference = pd.DataFrame({
        "country": (
            ["IN"] * 900
            + ["US"] * 80
            + ["UK"] * 20
        ),
    })

    current = pd.DataFrame({
        "country": (
            ["IN"] * 100
            + ["US"] * 100
            + ["UK"] * 800
        ),
    })

    report = detect_dataset_drift(
        reference,
        current,
    )

    assert len(report) == 1

    assert (
        report.loc[
            0,
            "data_type",
        ]
        == "categorical"
    )

    assert (
        report.loc[
            0,
            "drift_detected",
        ]
        == True
    )

    assert (
        report.loc[
            0,
            "fdr_significant",
        ]
        == True
    )

    assert (
        report.loc[
            0,
            "practical_magnitude",
        ]
        == "LARGE"
    )

    assert (
        report.loc[
            0,
            "interpretation",
        ]
        == "HIGH_IMPACT"
    )

    assert (
        report.loc[
            0,
            "final_decision",
        ]
        == "ALERT"
    )


# =================================================
# TEST 23 — NO DRIFT DATASET
# =================================================

def test_no_drift_dataset():

    np.random.seed(42)

    values = np.random.normal(
        loc=100,
        scale=10,
        size=1000,
    )

    reference = pd.DataFrame({
        "amount": values,
    })

    current = pd.DataFrame({
        "amount": values.copy(),
    })

    report = detect_dataset_drift(
        reference,
        current,
    )

    assert len(report) == 1

    assert (
        report.loc[
            0,
            "drift_detected",
        ]
        == False
    )

    assert (
        report.loc[
            0,
            "fdr_significant",
        ]
        == False
    )

    assert (
        report.loc[
            0,
            "final_decision",
        ]
        == "NO_DRIFT"
    )


# =================================================
# TEST 24 — MIXED DATASET
# =================================================

def test_mixed_dataset():

    np.random.seed(42)

    reference = pd.DataFrame({
        "stable_amount": np.random.normal(
            100,
            10,
            1000,
        ),

        "drifting_amount": np.random.normal(
            100,
            10,
            1000,
        ),

        "country": (
            ["IN"] * 700
            + ["US"] * 200
            + ["UK"] * 100
        ),
    })

    current = pd.DataFrame({
        "stable_amount": reference[
            "stable_amount"
        ].copy(),

        "drifting_amount": np.random.normal(
            500,
            10,
            1000,
        ),

        "country": (
            ["IN"] * 100
            + ["US"] * 100
            + ["UK"] * 800
        ),
    })

    report = detect_dataset_drift(
        reference,
        current,
    )

    assert len(report) == 3

    assert set(
        report["feature"]
    ) == {
        "stable_amount",
        "drifting_amount",
        "country",
    }

    drifting_result = report[
        report["feature"]
        == "drifting_amount"
    ].iloc[0]

    country_result = report[
        report["feature"]
        == "country"
    ].iloc[0]

    stable_result = report[
        report["feature"]
        == "stable_amount"
    ].iloc[0]

    assert (
        drifting_result[
            "final_decision"
        ]
        == "ALERT"
    )

    assert (
        country_result[
            "final_decision"
        ]
        == "ALERT"
    )

    assert (
        stable_result[
            "final_decision"
        ]
        == "NO_DRIFT"
    )


# =================================================
# TEST 25 — INSUFFICIENT DATA IN COMPLETE PIPELINE
# =================================================

def test_insufficient_data_complete_pipeline():

    reference = pd.DataFrame({
        "amount": [10, 20, 30],
    })

    current = pd.DataFrame({
        "amount": [100, 200, 300],
    })

    config = DriftConfig(
        min_samples=30,
    )

    report = detect_dataset_drift(
        reference,
        current,
        config=config,
    )

    assert len(report) == 1

    assert (
        report.loc[
            0,
            "status",
        ]
        == "INSUFFICIENT_DATA"
    )

    assert (
        report.loc[
            0,
            "final_decision",
        ]
        == "INSUFFICIENT_DATA"
    )


# =================================================
# TEST 26 — MISSING COLUMN REJECTION
# =================================================

def test_missing_column_rejection():

    reference = pd.DataFrame({
        "amount": [10, 20, 30],
        "age": [20, 30, 40],
    })

    current = pd.DataFrame({
        "amount": [15, 25, 35],
    })

    try:

        detect_dataset_drift(
            reference,
            current,
        )

    except ValueError as error:

        assert (
            "missing columns"
            in str(error).lower()
        )

    else:

        raise AssertionError(
            "Expected ValueError was not raised."
        )


# =================================================
# TEST 27 — EXTRA COLUMN REJECTION
# =================================================

def test_extra_column_rejection():

    reference = pd.DataFrame({
        "amount": [10, 20, 30],
    })

    current = pd.DataFrame({
        "amount": [15, 25, 35],
        "age": [20, 30, 40],
    })

    try:

        detect_dataset_drift(
            reference,
            current,
        )

    except ValueError as error:

        assert (
            "unexpected columns"
            in str(error).lower()
        )

    else:

        raise AssertionError(
            "Expected ValueError was not raised."
        )


# =================================================
# TEST 28 — DATA TYPE REJECTION
# =================================================

def test_dtype_rejection():

    reference = pd.DataFrame({
        "amount": [10, 20, 30],
    })

    current = pd.DataFrame({
        "amount": [
            "10",
            "20",
            "30",
        ],
    })

    try:

        detect_dataset_drift(
            reference,
            current,
        )

    except TypeError as error:

        assert (
            "data-type mismatch"
            in str(error).lower()
        )

    else:

        raise AssertionError(
            "Expected TypeError was not raised."
        )


# =================================================
# TEST 29 — MULTIPLE FEATURES AND PRIORITY ORDER
# =================================================

def test_priority_order():

    np.random.seed(42)

    reference = pd.DataFrame({
        "large_drift": np.random.normal(
            100,
            10,
            1000,
        ),

        "small_drift": np.random.normal(
            100,
            10,
            1000,
        ),

        "stable": np.random.normal(
            100,
            10,
            1000,
        ),
    })

    current = pd.DataFrame({
        "large_drift": np.random.normal(
            500,
            10,
            1000,
        ),

        "small_drift": (
            reference["small_drift"]
            + 0.001
        ),

        "stable": reference[
            "stable"
        ].copy(),
    })

    report = detect_dataset_drift(
        reference,
        current,
    )

    decisions = list(
        report["final_decision"]
    )

    # ALERT must appear before lower-priority
    # decisions.
    if "ALERT" in decisions:

        alert_index = decisions.index(
            "ALERT"
        )

        for decision in [
            "INVESTIGATE",
            "MONITOR",
            "NO_DRIFT",
            "INSUFFICIENT_DATA",
        ]:

            if decision in decisions:

                assert (
                    alert_index
                    < decisions.index(
                        decision
                    )
                )


# =================================================
# TEST 30 — COMPLETE REPORT STRUCTURE
# =================================================

def test_complete_report_structure():

    reference = pd.DataFrame({
        "amount": np.arange(100),
        "category": (
            ["A", "B", "C", "A", "B"]
            * 20
        ),
    })

    current = pd.DataFrame({
        "amount": np.arange(100),
        "category": (
            ["A", "B", "C", "A", "B"]
            * 20
        ),
    })

    report = detect_dataset_drift(
        reference,
        current,
    )

    required_columns = {
        "feature",
        "data_type",
        "test",
        "p_value",
        "adjusted_p_value",
        "magnitude",
        "normalized_magnitude",
        "drift_detected",
        "severity",
        "statistical_significance",
        "fdr_significant",
        "practical_magnitude",
        "interpretation",
        "final_decision",
        "reference_sample_size",
        "current_sample_size",
        "reference_missing_rate",
        "current_missing_rate",
        "missing_rate_difference",
        "status",
    }

    assert required_columns.issubset(
        report.columns
    )

    assert len(report) == 2


# =================================================
# TEST RUNNER
# =================================================

def main():

    tests = [
        test_practical_magnitude_no_drift,
        test_practical_magnitude_numerical_very_small,
        test_practical_magnitude_numerical_moderate,
        test_practical_magnitude_numerical_large,
        test_practical_magnitude_categorical_very_small,
        test_practical_magnitude_categorical_moderate,
        test_practical_magnitude_categorical_large,
        test_practical_magnitude_insufficient_data,
        test_interpretation_no_drift,
        test_interpretation_statistical_only,
        test_interpretation_actionable,
        test_interpretation_high_impact,
        test_final_decision_no_drift,
        test_final_decision_monitor,
        test_final_decision_investigate,
        test_final_decision_alert,
        test_final_decision_insufficient_data,
        test_fdr_correction,
        test_fdr_with_missing_p_value,
        test_fdr_empty_report,
        test_complete_numerical_drift,
        test_complete_categorical_drift,
        test_no_drift_dataset,
        test_mixed_dataset,
        test_insufficient_data_complete_pipeline,
        test_missing_column_rejection,
        test_extra_column_rejection,
        test_dtype_rejection,
        test_priority_order,
        test_complete_report_structure,
    ]

    print("=" * 100)
    print("CORE DRIFT DETECTOR UNIT TESTS")
    print("=" * 100)

    passed = 0

    for test in tests:

        try:

            test()

            print(
                f"✓ {test.__name__}"
            )

            passed += 1

        except Exception as error:

            print(
                f"✗ {test.__name__}"
            )

            print(
                f"  Error: {error}"
            )

            raise

    print("\n" + "=" * 100)

    print(
        f"ALL {passed} DETECTOR TESTS PASSED"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()