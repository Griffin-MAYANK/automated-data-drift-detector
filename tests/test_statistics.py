"""
Unit tests for statistical drift detection.

These tests verify:

1. Total Variation Distance
2. Numerical drift detection
3. Categorical drift detection
4. Numerical severity classification
5. Categorical severity classification
6. No-drift scenarios
7. Strong-drift scenarios
8. Insufficient-data handling
9. Missing-value handling
10. Result structure
"""

import numpy as np
import pandas as pd

from drift_detector.statistics import (
    calculate_total_variation_distance,
    classify_categorical_severity,
    classify_numerical_severity,
    detect_categorical_drift,
    detect_numerical_drift,
)


# =================================================
# TEST 1 — TVD IDENTICAL DISTRIBUTIONS
# =================================================

def test_tvd_identical_distributions():

    reference = pd.Series([
        "A",
        "A",
        "B",
        "B",
        "C",
        "C",
    ])

    current = pd.Series([
        "A",
        "A",
        "B",
        "B",
        "C",
        "C",
    ])

    tvd = calculate_total_variation_distance(
        reference,
        current,
    )

    assert tvd == 0.0


# =================================================
# TEST 2 — TVD DIFFERENT DISTRIBUTIONS
# =================================================

def test_tvd_different_distributions():

    reference = pd.Series([
        "A",
        "A",
        "A",
        "A",
    ])

    current = pd.Series([
        "B",
        "B",
        "B",
        "B",
    ])

    tvd = calculate_total_variation_distance(
        reference,
        current,
    )

    assert tvd == 1.0


# =================================================
# TEST 3 — NUMERICAL NO DRIFT
# =================================================

def test_numerical_no_drift():

    np.random.seed(42)

    reference = pd.Series(
        np.random.normal(
            loc=100,
            scale=10,
            size=1000,
        )
    )

    current = pd.Series(
        np.random.normal(
            loc=100,
            scale=10,
            size=1000,
        )
    )

    result = detect_numerical_drift(
        reference,
        current,
    )

    assert result["status"] == "OK"

    assert (
        result["reference_sample_size"]
        == 1000
    )

    assert (
        result["current_sample_size"]
        == 1000
    )

    assert isinstance(
        result["drift_detected"],
        bool,
    )


def test_numerical_drift_uses_explicit_asymptotic_ks_method(monkeypatch):
    import drift_detector.statistics as statistics

    captured = {}

    def fake_ks_2samp(reference, current, method):
        captured["method"] = method
        return 0.0, 1.0

    monkeypatch.setattr(statistics, "ks_2samp", fake_ks_2samp)

    result = statistics.detect_numerical_drift(
        pd.Series(np.arange(30)),
        pd.Series(np.arange(30)),
    )

    assert result["status"] == "OK"
    assert captured["method"] == "asymp"


# =================================================
# TEST 4 — NUMERICAL STRONG DRIFT
# =================================================

def test_numerical_strong_drift():

    np.random.seed(42)

    reference = pd.Series(
        np.random.normal(
            loc=100,
            scale=10,
            size=1000,
        )
    )

    current = pd.Series(
        np.random.normal(
            loc=500,
            scale=10,
            size=1000,
        )
    )

    result = detect_numerical_drift(
        reference,
        current,
    )

    assert result["status"] == "OK"

    assert result["drift_detected"] is True

    assert result["p_value"] < 0.05

    assert result["magnitude"] > 0

    assert (
        result["normalized_magnitude"] > 0
    )


# =================================================
# TEST 5 — NUMERICAL MISSING VALUES
# =================================================

def test_numerical_missing_values():

    reference = pd.Series([
        10,
        20,
        30,
        40,
        50,
        np.nan,
        np.nan,
    ])

    current = pd.Series([
        10,
        20,
        30,
        40,
        50,
        60,
        70,
    ])

    result = detect_numerical_drift(
        reference,
        current,
        min_samples=5,
    )

    assert result["status"] == "OK"

    assert (
        result["reference_sample_size"]
        == 5
    )

    assert (
        result["current_sample_size"]
        == 7
    )


# =================================================
# TEST 6 — NUMERICAL INSUFFICIENT DATA
# =================================================

def test_numerical_insufficient_data():

    reference = pd.Series([
        10,
        20,
        30,
    ])

    current = pd.Series([
        10,
        20,
        30,
    ])

    result = detect_numerical_drift(
        reference,
        current,
        min_samples=30,
    )

    assert (
        result["status"]
        == "INSUFFICIENT_DATA"
    )

    assert (
        result["drift_detected"]
        is False
    )

    assert pd.isna(
        result["p_value"]
    )


# =================================================
# TEST 7 — CATEGORICAL NO DRIFT
# =================================================

def test_categorical_no_drift():

    np.random.seed(42)

    categories = [
        "A",
        "B",
        "C",
    ]

    reference = pd.Series(
        np.random.choice(
            categories,
            size=1000,
            p=[
                0.60,
                0.30,
                0.10,
            ],
        )
    )

    current = pd.Series(
        np.random.choice(
            categories,
            size=1000,
            p=[
                0.60,
                0.30,
                0.10,
            ],
        )
    )

    result = detect_categorical_drift(
        reference,
        current,
    )

    assert result["status"] == "OK"

    assert (
        result["reference_sample_size"]
        == 1000
    )

    assert (
        result["current_sample_size"]
        == 1000
    )

    assert isinstance(
        result["drift_detected"],
        bool,
    )


# =================================================
# TEST 8 — CATEGORICAL STRONG DRIFT
# =================================================

def test_categorical_strong_drift():

    reference = pd.Series(
        ["A"] * 900
        + ["B"] * 80
        + ["C"] * 20
    )

    current = pd.Series(
        ["A"] * 100
        + ["B"] * 100
        + ["C"] * 800
    )

    result = detect_categorical_drift(
        reference,
        current,
    )

    assert result["status"] == "OK"

    assert (
        result["drift_detected"]
        is True
    )

    assert (
        result["p_value"] < 0.05
    )

    assert (
        result["magnitude"] > 0.20
    )

    assert (
        result["severity"]
        == "HIGH"
    )


# =================================================
# TEST 9 — CATEGORICAL MISSING VALUES
# =================================================

def test_categorical_missing_values():
    """
    Missing categorical values should be treated
    as a separate category.

    We deliberately use more than the default
    minimum sample size so this test evaluates
    missing-value handling rather than
    insufficient-data handling.
    """

    reference = pd.Series(
        ["A"] * 40
        + ["B"] * 40
        + [np.nan] * 20
    )

    current = pd.Series(
        ["A"] * 40
        + ["B"] * 30
        + ["C"] * 20
        + [np.nan] * 10
    )

    result = detect_categorical_drift(
        reference,
        current,
    )

    assert result["status"] == "OK"

    assert (
        result["reference_sample_size"]
        == 100
    )

    assert (
        result["current_sample_size"]
        == 100
    )

    assert (
        result["magnitude"] > 0
    )


# =================================================
# TEST 10 — CATEGORICAL INSUFFICIENT DATA
# =================================================

def test_categorical_insufficient_data():

    reference = pd.Series([
        "A",
        "B",
        "C",
    ])

    current = pd.Series([
        "A",
        "B",
        "C",
    ])

    result = detect_categorical_drift(
        reference,
        current,
        min_samples=30,
    )

    assert (
        result["status"]
        == "INSUFFICIENT_DATA"
    )

    assert (
        result["drift_detected"]
        is False
    )

    assert pd.isna(
        result["p_value"]
    )


# =================================================
# TEST 11 — NUMERICAL SEVERITY
# =================================================

def test_numerical_severity():

    assert (
        classify_numerical_severity(
            p_value=0.50,
            normalized_magnitude=0.01,
        )
        == "NO_DRIFT"
    )

    assert (
        classify_numerical_severity(
            p_value=0.001,
            normalized_magnitude=0.05,
        )
        == "LOW"
    )

    assert (
        classify_numerical_severity(
            p_value=0.001,
            normalized_magnitude=0.20,
        )
        == "MEDIUM"
    )

    assert (
        classify_numerical_severity(
            p_value=0.001,
            normalized_magnitude=1.00,
        )
        == "HIGH"
    )


# =================================================
# TEST 12 — CATEGORICAL SEVERITY
# =================================================

def test_categorical_severity():

    assert (
        classify_categorical_severity(
            p_value=0.50,
            tvd=0.01,
        )
        == "NO_DRIFT"
    )

    assert (
        classify_categorical_severity(
            p_value=0.001,
            tvd=0.02,
        )
        == "LOW"
    )

    assert (
        classify_categorical_severity(
            p_value=0.001,
            tvd=0.10,
        )
        == "MEDIUM"
    )

    assert (
        classify_categorical_severity(
            p_value=0.001,
            tvd=0.50,
        )
        == "HIGH"
    )


# =================================================
# TEST 13 — TVD MISSING VALUES
# =================================================

def test_tvd_missing_values():

    reference = pd.Series([
        "A",
        "A",
        "B",
        np.nan,
    ])

    current = pd.Series([
        "A",
        "A",
        "B",
        "B",
    ])

    tvd = calculate_total_variation_distance(
        reference,
        current,
    )

    assert tvd > 0

    assert tvd <= 1


# =================================================
# TEST 14 — NUMERICAL RESULT STRUCTURE
# =================================================

def test_numerical_result_structure():

    reference = pd.Series(
        np.arange(100)
    )

    current = pd.Series(
        np.arange(100)
    )

    result = detect_numerical_drift(
        reference,
        current,
    )

    required_keys = {
        "statistic",
        "p_value",
        "magnitude",
        "normalized_magnitude",
        "drift_detected",
        "reference_sample_size",
        "current_sample_size",
        "status",
    }

    assert required_keys.issubset(
        result.keys()
    )


# =================================================
# TEST 15 — CATEGORICAL RESULT STRUCTURE
# =================================================

def test_categorical_result_structure():

    reference = pd.Series(
        ["A", "B", "C"] * 100
    )

    current = pd.Series(
        ["A", "B", "C"] * 100
    )

    result = detect_categorical_drift(
        reference,
        current,
    )

    required_keys = {
        "statistic",
        "p_value",
        "magnitude",
        "drift_detected",
        "severity",
        "reference_sample_size",
        "current_sample_size",
        "status",
    }

    assert required_keys.issubset(
        result.keys()
    )


# =================================================
# TEST RUNNER
# =================================================

def main():

    tests = [
        test_tvd_identical_distributions,
        test_tvd_different_distributions,
        test_numerical_no_drift,
        test_numerical_strong_drift,
        test_numerical_missing_values,
        test_numerical_insufficient_data,
        test_categorical_no_drift,
        test_categorical_strong_drift,
        test_categorical_missing_values,
        test_categorical_insufficient_data,
        test_numerical_severity,
        test_categorical_severity,
        test_tvd_missing_values,
        test_numerical_result_structure,
        test_categorical_result_structure,
    ]

    print("=" * 100)
    print("STATISTICAL UNIT TESTS")
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
        f"ALL {passed} STATISTICAL TESTS PASSED"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()