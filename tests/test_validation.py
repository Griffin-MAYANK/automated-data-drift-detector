"""
Unit tests for dataset validation.

These tests verify:

1. Empty dataset detection
2. Missing-column detection
3. Extra-column detection
4. Data-type mismatch detection
5. Small-sample detection
6. Missing-value detection
7. Constant-feature detection
8. Valid dataset pairs
9. Custom minimum sample sizes
10. Validation result structure
"""

import numpy as np
import pandas as pd

from drift_detector.validation import (
    validate_dataset_pair,
)


# =================================================
# TEST 1 — VALID DATASETS
# =================================================

def test_valid_datasets():
    """
    Identical, healthy datasets should produce
    no structural validation problems.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30, 35, 40],
        "income": [20000, 30000, 40000, 50000, 60000],
        "country": [
            "IN",
            "IN",
            "US",
            "UK",
            "IN",
        ],
    })

    current = reference.copy()

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=5,
    )

    assert result["reference_empty"] is False

    assert result["current_empty"] is False

    assert result["missing_columns"] == []

    assert result["extra_columns"] == []

    assert result["dtype_mismatches"] == {}

    assert result["small_reference_features"] == []

    assert result["small_current_features"] == []

    assert result["reference_missing_values"] == {}

    assert result["current_missing_values"] == {}

    assert result["reference_constant_features"] == []

    assert result["current_constant_features"] == []


# =================================================
# TEST 2 — EMPTY REFERENCE DATASET
# =================================================

def test_empty_reference_dataset():
    """
    Empty reference datasets should be detected.
    """

    reference = pd.DataFrame(
        columns=[
            "age",
            "income",
        ]
    )

    current = pd.DataFrame({
        "age": [20, 25, 30],
        "income": [20000, 30000, 40000],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert result["reference_empty"] is True

    assert result["current_empty"] is False


# =================================================
# TEST 3 — EMPTY CURRENT DATASET
# =================================================

def test_empty_current_dataset():
    """
    Empty current datasets should be detected.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30],
        "income": [20000, 30000, 40000],
    })

    current = pd.DataFrame(
        columns=[
            "age",
            "income",
        ]
    )

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert result["reference_empty"] is False

    assert result["current_empty"] is True


# =================================================
# TEST 4 — BOTH DATASETS EMPTY
# =================================================

def test_both_datasets_empty():
    """
    Both empty datasets should be detected.
    """

    reference = pd.DataFrame(
        columns=[
            "age",
            "income",
        ]
    )

    current = pd.DataFrame(
        columns=[
            "age",
            "income",
        ]
    )

    result = validate_dataset_pair(
        reference,
        current,
    )

    assert result["reference_empty"] is True

    assert result["current_empty"] is True


# =================================================
# TEST 5 — MISSING COLUMN
# =================================================

def test_missing_column():
    """
    A column present in reference but absent from
    current should be reported.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30],
        "income": [20000, 30000, 40000],
        "country": ["IN", "IN", "US"],
    })

    current = pd.DataFrame({
        "age": [21, 26, 31],
        "income": [21000, 31000, 41000],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert (
        result["missing_columns"]
        == ["country"]
    )


# =================================================
# TEST 6 — EXTRA COLUMN
# =================================================

def test_extra_column():
    """
    A column present in current but absent from
    reference should be reported.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30],
        "income": [20000, 30000, 40000],
    })

    current = pd.DataFrame({
        "age": [21, 26, 31],
        "income": [21000, 31000, 41000],
        "country": ["IN", "IN", "US"],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert (
        result["extra_columns"]
        == ["country"]
    )


# =================================================
# TEST 7 — MULTIPLE MISSING COLUMNS
# =================================================

def test_multiple_missing_columns():
    """
    Multiple missing columns should all be reported
    in sorted order.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30],
        "income": [20000, 30000, 40000],
        "country": ["IN", "IN", "US"],
        "city": ["Delhi", "Mumbai", "Pune"],
    })

    current = pd.DataFrame({
        "age": [21, 26, 31],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert result["missing_columns"] == [
        "city",
        "country",
        "income",
    ]


# =================================================
# TEST 8 — DATA TYPE MISMATCH
# =================================================

def test_dtype_mismatch():
    """
    A feature changing from numeric to string should
    be detected as a dtype mismatch.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30],
        "income": [20000, 30000, 40000],
    })

    current = pd.DataFrame({
        "age": ["20", "25", "30"],
        "income": [21000, 31000, 41000],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert "age" in result[
        "dtype_mismatches"
    ]

    assert (
        result["dtype_mismatches"]["age"][
            "reference"
        ]
        == "int64"
    )

    assert (
        result["dtype_mismatches"]["age"][
            "current"
        ]
        == "str"
    )


# =================================================
# TEST 9 — MULTIPLE DATA TYPE MISMATCHES
# =================================================

def test_multiple_dtype_mismatches():
    """
    Multiple dtype changes should be detected.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30],
        "income": [20000.0, 30000.0, 40000.0],
        "active": [1, 0, 1],
    })

    current = pd.DataFrame({
        "age": ["20", "25", "30"],
        "income": ["20000", "30000", "40000"],
        "active": ["yes", "no", "yes"],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert set(
        result["dtype_mismatches"].keys()
    ) == {
        "active",
        "age",
        "income",
    }


# =================================================
# TEST 10 — SMALL REFERENCE FEATURE
# =================================================

def test_small_reference_feature():
    """
    A feature with fewer than min_samples valid
    observations should be reported.
    """

    reference = pd.DataFrame({
        "age": [20, 25, np.nan],
        "income": [20000, 30000, 40000],
    })

    current = pd.DataFrame({
        "age": [21, 26, 31],
        "income": [21000, 31000, 41000],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert (
        "age"
        in result["small_reference_features"]
    )

    assert (
        "income"
        not in result["small_reference_features"]
    )


# =================================================
# TEST 11 — SMALL CURRENT FEATURE
# =================================================

def test_small_current_feature():
    """
    A feature with fewer than min_samples valid
    observations in current data should be reported.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30],
        "income": [20000, 30000, 40000],
    })

    current = pd.DataFrame({
        "age": [21, 26, np.nan],
        "income": [21000, 31000, 41000],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert (
        "age"
        in result["small_current_features"]
    )

    assert (
        "income"
        not in result["small_current_features"]
    )


# =================================================
# TEST 12 — MISSING VALUE DETECTION
# =================================================

def test_missing_value_detection():
    """
    Missing-value counts and rates should be
    calculated correctly.
    """

    reference = pd.DataFrame({
        "age": [
            20,
            25,
            np.nan,
            35,
            np.nan,
        ],
        "income": [
            20000,
            30000,
            40000,
            50000,
            60000,
        ],
    })

    current = pd.DataFrame({
        "age": [
            21,
            np.nan,
            31,
            36,
            41,
        ],
        "income": [
            21000,
            31000,
            np.nan,
            51000,
            61000,
        ],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert (
        result[
            "reference_missing_values"
        ]["age"]["count"]
        == 2
    )

    assert (
        result[
            "reference_missing_values"
        ]["age"]["rate"]
        == 0.4
    )

    assert (
        result[
            "current_missing_values"
        ]["age"]["count"]
        == 1
    )

    assert (
        result[
            "current_missing_values"
        ]["age"]["rate"]
        == 0.2
    )

    assert (
        result[
            "current_missing_values"
        ]["income"]["count"]
        == 1
    )

    assert (
        result[
            "current_missing_values"
        ]["income"]["rate"]
        == 0.2
    )


# =================================================
# TEST 13 — CONSTANT REFERENCE FEATURE
# =================================================

def test_constant_reference_feature():
    """
    A feature containing only one unique non-null
    value should be detected as constant.
    """

    reference = pd.DataFrame({
        "age": [20, 20, 20, 20, 20],
        "income": [
            20000,
            30000,
            40000,
            50000,
            60000,
        ],
    })

    current = pd.DataFrame({
        "age": [20, 21, 22, 23, 24],
        "income": [
            21000,
            31000,
            41000,
            51000,
            61000,
        ],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=5,
    )

    assert (
        result[
            "reference_constant_features"
        ]
        == ["age"]
    )

    assert (
        result[
            "current_constant_features"
        ]
        == []
    )


# =================================================
# TEST 14 — CONSTANT CURRENT FEATURE
# =================================================

def test_constant_current_feature():
    """
    A feature containing only one unique non-null
    value in current data should be detected.
    """

    reference = pd.DataFrame({
        "age": [20, 21, 22, 23, 24],
        "income": [
            20000,
            30000,
            40000,
            50000,
            60000,
        ],
    })

    current = pd.DataFrame({
        "age": [30, 30, 30, 30, 30],
        "income": [
            21000,
            31000,
            41000,
            51000,
            61000,
        ],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=5,
    )

    assert (
        result[
            "reference_constant_features"
        ]
        == []
    )

    assert (
        result[
            "current_constant_features"
        ]
        == ["age"]
    )


# =================================================
# TEST 15 — CONSTANT FEATURE WITH MISSING VALUES
# =================================================

def test_constant_feature_with_missing_values():
    """
    Missing values should not prevent a feature from
    being classified as constant.

    Only non-null unique values are considered.
    """

    reference = pd.DataFrame({
        "age": [
            20,
            20,
            20,
            np.nan,
            np.nan,
        ],
    })

    current = pd.DataFrame({
        "age": [
            20,
            20,
            21,
            22,
            23,
        ],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    assert (
        result[
            "reference_constant_features"
        ]
        == ["age"]
    )

    assert (
        result[
            "current_constant_features"
        ]
        == []
    )


# =================================================
# TEST 16 — CUSTOM MINIMUM SAMPLE SIZE
# =================================================

def test_custom_min_samples():
    """
    Validation should respect a custom minimum
    sample-size threshold.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30, 35, 40],
    })

    current = pd.DataFrame({
        "age": [21, 26, 31, 36, 41],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=10,
    )

    assert (
        "age"
        in result["small_reference_features"]
    )

    assert (
        "age"
        in result["small_current_features"]
    )


# =================================================
# TEST 17 — INVALID MINIMUM SAMPLE SIZE
# =================================================

def test_invalid_min_samples():
    """
    min_samples below 1 should raise ValueError.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30],
    })

    current = pd.DataFrame({
        "age": [21, 26, 31],
    })

    try:

        validate_dataset_pair(
            reference,
            current,
            min_samples=0,
        )

    except ValueError as error:

        assert (
            "min_samples must be at least 1"
            in str(error)
        )

    else:

        raise AssertionError(
            "Expected ValueError was not raised."
        )


# =================================================
# TEST 18 — VALIDATION RESULT STRUCTURE
# =================================================

def test_validation_result_structure():
    """
    Verify that the validation function returns
    all required fields.
    """

    reference = pd.DataFrame({
        "age": [20, 25, 30],
    })

    current = pd.DataFrame({
        "age": [21, 26, 31],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    required_keys = {
        "reference_empty",
        "current_empty",
        "missing_columns",
        "extra_columns",
        "dtype_mismatches",
        "small_reference_features",
        "small_current_features",
        "reference_missing_values",
        "current_missing_values",
        "reference_constant_features",
        "current_constant_features",
    }

    assert required_keys.issubset(
        result.keys()
    )


# =================================================
# TEST 19 — MULTIPLE VALIDATION ISSUES
# =================================================

def test_multiple_validation_issues():
    """
    The validator should be able to detect several
    problems simultaneously.
    """

    reference = pd.DataFrame({
        "age": [
            20,
            20,
            np.nan,
        ],
        "income": [
            30000,
            30000,
            30000,
        ],
        "country": [
            "IN",
            "IN",
            "IN",
        ],
    })

    current = pd.DataFrame({
        "age": [
            "21",
            "22",
            "23",
        ],
        "income": [
            31000,
            32000,
            33000,
        ],
        "city": [
            "Delhi",
            "Mumbai",
            "Pune",
        ],
    })

    result = validate_dataset_pair(
        reference,
        current,
        min_samples=3,
    )

    # country exists only in reference.
    assert (
        result["missing_columns"]
        == ["country"]
    )

    # city exists only in current.
    assert (
        result["extra_columns"]
        == ["city"]
    )

    # age changed from numeric to string.
    assert "age" in result[
        "dtype_mismatches"
    ]

    # Reference age has only two valid values.
    assert (
        "age"
        in result["small_reference_features"]
    )

    # Reference income is constant.
    assert (
        "income"
        in result[
            "reference_constant_features"
        ]
    )

    # Reference country is constant.
    assert (
        "country"
        in result[
            "reference_constant_features"
        ]
    )

    # Reference age contains one missing value.
    assert (
        result[
            "reference_missing_values"
        ]["age"]["count"]
        == 1
    )


# =================================================
# TEST RUNNER
# =================================================

def main():

    tests = [
        test_valid_datasets,
        test_empty_reference_dataset,
        test_empty_current_dataset,
        test_both_datasets_empty,
        test_missing_column,
        test_extra_column,
        test_multiple_missing_columns,
        test_dtype_mismatch,
        test_multiple_dtype_mismatches,
        test_small_reference_feature,
        test_small_current_feature,
        test_missing_value_detection,
        test_constant_reference_feature,
        test_constant_current_feature,
        test_constant_feature_with_missing_values,
        test_custom_min_samples,
        test_invalid_min_samples,
        test_validation_result_structure,
        test_multiple_validation_issues,
    ]

    print("=" * 100)
    print("DATASET VALIDATION UNIT TESTS")
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
        f"ALL {passed} VALIDATION TESTS PASSED"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()