import pandas as pd
import numpy as np

from drift_detector.config import (
    FEATURE_TYPE_METADATA,
    FeatureType,
)
from drift_detector.features import (
    REQUIRED_COLUMNS,
    MONITORING_FEATURES,
    validate_raw_retail_data,
    create_monitoring_features,
    split_reference_current,
    create_reference_current_features,
)


def test_all_monitoring_features_have_semantic_metadata():
    assert set(MONITORING_FEATURES) == set(FEATURE_TYPE_METADATA)


def test_monitoring_feature_semantic_types():
    assert FEATURE_TYPE_METADATA["quantity"] is FeatureType.NUMERICAL
    assert FEATURE_TYPE_METADATA["unit_price"] is FeatureType.NUMERICAL
    assert (
        FEATURE_TYPE_METADATA["transaction_value"]
        is FeatureType.NUMERICAL
    )
    assert (
        FEATURE_TYPE_METADATA["transaction_hour"]
        is FeatureType.CYCLICAL
    )
    assert (
        FEATURE_TYPE_METADATA["day_of_week"]
        is FeatureType.CYCLICAL
    )
    assert FEATURE_TYPE_METADATA["month"] is FeatureType.CYCLICAL
    assert (
        FEATURE_TYPE_METADATA["is_cancelled"]
        is FeatureType.CATEGORICAL
    )
    assert FEATURE_TYPE_METADATA["country"] is FeatureType.CATEGORICAL


def create_sample_raw_data():
    """
    Create a small realistic retail dataset
    for testing feature engineering.
    """

    return pd.DataFrame({
        "InvoiceNo": [
            "10001",
            "10002",
            "C10003",
            "10004",
            "10005",
            "C10006",
        ],

        "StockCode": [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ],

        "Description": [
            "Product A",
            "Product B",
            "Product C",
            "Product D",
            "Product E",
            "Product F",
        ],

        "Quantity": [
            2,
            5,
            -1,
            10,
            3,
            -2,
        ],

        "InvoiceDate": [
            "2011-01-10 08:15:00",
            "2011-02-15 12:30:00",
            "2011-03-20 18:45:00",
            "2011-06-30 23:00:00",
            "2011-07-01 09:30:00",
            "2011-08-10 14:15:00",
        ],

        "UnitPrice": [
            10.0,
            5.0,
            20.0,
            2.5,
            15.0,
            8.0,
        ],

        "CustomerID": [
            1001,
            1002,
            1003,
            1004,
            1005,
            1006,
        ],

        "Country": [
            "United Kingdom",
            "France",
            "Germany",
            "United Kingdom",
            "France",
            "Germany",
        ],
    })


def test_required_columns():

    expected = {
        "InvoiceNo",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "Country",
    }

    assert REQUIRED_COLUMNS == expected

    print("✓ test_required_columns")


def test_monitoring_features_definition():

    expected = [
        "quantity",
        "unit_price",
        "transaction_value",
        "transaction_hour",
        "day_of_week",
        "month",
        "is_cancelled",
        "country",
    ]

    assert MONITORING_FEATURES == expected

    print("✓ test_monitoring_features_definition")


def test_validate_valid_data():

    data = create_sample_raw_data()

    result = validate_raw_retail_data(data)

    assert result is None

    print("✓ test_validate_valid_data")


def test_validate_empty_data():

    data = pd.DataFrame()

    try:
        validate_raw_retail_data(data)
        assert False, "Empty data should be rejected."
    except ValueError:
        pass

    print("✓ test_validate_empty_data")


def test_validate_missing_columns():

    data = create_sample_raw_data()

    data = data.drop(
        columns=["Country"]
    )

    try:
        validate_raw_retail_data(data)
        assert False, "Missing column should be rejected."
    except KeyError:
        pass

    print("✓ test_validate_missing_columns")


def test_validate_multiple_missing_columns():

    data = create_sample_raw_data()

    data = data.drop(
        columns=[
            "Country",
            "InvoiceNo",
            "Quantity",
        ]
    )

    try:
        validate_raw_retail_data(data)
        assert False, "Missing columns should be rejected."
    except KeyError:
        pass

    print("✓ test_validate_multiple_missing_columns")


def test_monitoring_feature_columns():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    assert list(features.columns) == MONITORING_FEATURES

    print("✓ test_monitoring_feature_columns")


def test_monitoring_feature_row_count():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    assert len(features) == len(data)

    print("✓ test_monitoring_feature_row_count")


def test_quantity_feature():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    expected = [
        2,
        5,
        -1,
        10,
        3,
        -2,
    ]

    assert features["quantity"].tolist() == expected

    print("✓ test_quantity_feature")


def test_unit_price_feature():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    expected = [
        10.0,
        5.0,
        20.0,
        2.5,
        15.0,
        8.0,
    ]

    assert features["unit_price"].tolist() == expected

    print("✓ test_unit_price_feature")


def test_transaction_value():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    expected = [
        20.0,
        25.0,
        -20.0,
        25.0,
        45.0,
        -16.0,
    ]

    assert features["transaction_value"].tolist() == expected

    print("✓ test_transaction_value")


def test_transaction_hour():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    expected = [
        8,
        12,
        18,
        23,
        9,
        14,
    ]

    assert features["transaction_hour"].tolist() == expected

    print("✓ test_transaction_hour")


def test_day_of_week():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    expected = (
        pd.to_datetime(
            data["InvoiceDate"]
        )
        .dt.dayofweek
        .tolist()
    )

    assert features["day_of_week"].tolist() == expected

    print("✓ test_day_of_week")


def test_month():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    expected = [
        1,
        2,
        3,
        6,
        7,
        8,
    ]

    assert features["month"].tolist() == expected

    print("✓ test_month")


def test_cancellation_detection():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    expected = [
        0,
        0,
        1,
        0,
        0,
        1,
    ]

    assert features["is_cancelled"].tolist() == expected

    print("✓ test_cancellation_detection")


def test_country_feature():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    expected = [
        "United Kingdom",
        "France",
        "Germany",
        "United Kingdom",
        "France",
        "Germany",
    ]

    assert (
        features["country"]
        .astype(str)
        .tolist()
        == expected
    )

    print("✓ test_country_feature")


def test_feature_dtypes():

    data = create_sample_raw_data()

    features = create_monitoring_features(data)

    numerical_columns = [
        "quantity",
        "unit_price",
        "transaction_value",
        "transaction_hour",
        "day_of_week",
        "month",
        "is_cancelled",
    ]

    for column in numerical_columns:
        assert pd.api.types.is_numeric_dtype(
            features[column]
        )

    assert (
        pd.api.types.is_string_dtype(
            features["country"]
        )
        or
        pd.api.types.is_object_dtype(
            features["country"]
        )
    )

    print("✓ test_feature_dtypes")


def test_split_reference_current():

    data = create_sample_raw_data()

    reference, current = split_reference_current(
        data,
        "2011-07-01",
    )

    assert len(reference) == 4
    assert len(current) == 2

    reference_dates = pd.to_datetime(
        reference["InvoiceDate"]
    )

    current_dates = pd.to_datetime(
        current["InvoiceDate"]
    )

    assert (
        reference_dates < pd.Timestamp("2011-07-01")
    ).all()

    assert (
        current_dates >= pd.Timestamp("2011-07-01")
    ).all()

    print("✓ test_split_reference_current")


def test_split_preserves_all_rows():

    data = create_sample_raw_data()

    reference, current = split_reference_current(
        data,
        "2011-07-01",
    )

    assert (
        len(reference) + len(current)
        == len(data)
    )

    print("✓ test_split_preserves_all_rows")


def test_split_empty_reference():

    data = create_sample_raw_data()

    try:
        split_reference_current(
            data,
            "2000-01-01",
        )

        assert False, (
            "Empty reference dataset should "
            "raise an exception."
        )

    except ValueError:
        pass

    print("✓ test_split_empty_reference")


def test_split_empty_current():

    data = create_sample_raw_data()

    try:
        split_reference_current(
            data,
            "2030-01-01",
        )

        assert False, (
            "Empty current dataset should "
            "raise an exception."
        )

    except ValueError:
        pass

    print("✓ test_split_empty_current")


def test_create_reference_current_features():

    data = create_sample_raw_data()

    reference, current = (
        create_reference_current_features(
            data,
            "2011-07-01",
        )
    )

    assert len(reference) == 4
    assert len(current) == 2

    assert list(reference.columns) == (
        MONITORING_FEATURES
    )

    assert list(current.columns) == (
        MONITORING_FEATURES
    )

    print(
        "✓ test_create_reference_current_features"
    )


def test_feature_engineering_does_not_modify_input():

    data = create_sample_raw_data()

    original_columns = list(data.columns)
    original_shape = data.shape

    create_monitoring_features(data)

    assert list(data.columns) == original_columns
    assert data.shape == original_shape

    print(
        "✓ test_feature_engineering_does_not_modify_input"
    )


def test_transaction_value_with_zero_quantity():

    data = create_sample_raw_data()

    data.loc[0, "Quantity"] = 0

    features = create_monitoring_features(data)

    assert features.loc[0, "transaction_value"] == 0.0

    print(
        "✓ test_transaction_value_with_zero_quantity"
    )


def test_numeric_conversion():

    data = create_sample_raw_data()

    data["Quantity"] = data["Quantity"].astype(str)
    data["UnitPrice"] = data["UnitPrice"].astype(str)

    features = create_monitoring_features(data)

    assert pd.api.types.is_numeric_dtype(
        features["quantity"]
    )

    assert pd.api.types.is_numeric_dtype(
        features["unit_price"]
    )

    assert pd.api.types.is_numeric_dtype(
        features["transaction_value"]
    )

    print("✓ test_numeric_conversion")


def run_all_tests():

    tests = [
        test_required_columns,
        test_monitoring_features_definition,
        test_validate_valid_data,
        test_validate_empty_data,
        test_validate_missing_columns,
        test_validate_multiple_missing_columns,
        test_monitoring_feature_columns,
        test_monitoring_feature_row_count,
        test_quantity_feature,
        test_unit_price_feature,
        test_transaction_value,
        test_transaction_hour,
        test_day_of_week,
        test_month,
        test_cancellation_detection,
        test_country_feature,
        test_feature_dtypes,
        test_split_reference_current,
        test_split_preserves_all_rows,
        test_split_empty_reference,
        test_split_empty_current,
        test_create_reference_current_features,
        test_feature_engineering_does_not_modify_input,
        test_transaction_value_with_zero_quantity,
        test_numeric_conversion,
    ]

    print("=" * 100)
    print("FEATURE ENGINEERING UNIT TESTS")
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
            f"ALL {passed} FEATURE ENGINEERING TESTS PASSED"
        )

        print("=" * 100)

    else:

        print(
            f"{passed}/{len(tests)} "
            "FEATURE ENGINEERING TESTS PASSED"
        )

        print("=" * 100)

        raise SystemExit(1)


if __name__ == "__main__":
    run_all_tests()
