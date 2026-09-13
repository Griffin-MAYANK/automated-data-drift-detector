"""
Dataset validation functions.

This module validates reference and current datasets
before statistical drift detection is performed.

Validation checks include:

1. Empty datasets
2. Missing columns
3. Extra columns
4. Data-type mismatches
5. Small sample sizes
6. Missing values
7. Constant features
"""

import pandas as pd


def validate_dataset_pair(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    min_samples: int = 30,
) -> dict:
    """
    Validate reference and current datasets.

    Parameters
    ----------
    reference_data : pandas.DataFrame
        Baseline dataset used for comparison.

    current_data : pandas.DataFrame
        Current dataset being monitored.

    min_samples : int, default=30
        Minimum number of non-null values required
        for each feature.

    Returns
    -------
    dict
        Validation results containing dataset-level
        and feature-level validation information.
    """

    if min_samples < 1:
        raise ValueError(
            "min_samples must be at least 1."
        )

    validation = {
        "reference_empty": reference_data.empty,
        "current_empty": current_data.empty,
        "missing_columns": [],
        "extra_columns": [],
        "dtype_mismatches": {},
        "small_reference_features": [],
        "small_current_features": [],
        "reference_missing_values": {},
        "current_missing_values": {},
        "reference_constant_features": [],
        "current_constant_features": [],
    }

    reference_columns = set(
        reference_data.columns
    )

    current_columns = set(
        current_data.columns
    )

    validation["missing_columns"] = sorted(
        reference_columns - current_columns
    )

    validation["extra_columns"] = sorted(
        current_columns - reference_columns
    )

    shared_columns = (
        reference_columns & current_columns
    )

    # --------------------------------------------
    # Data-type validation
    # --------------------------------------------

    for column in sorted(shared_columns):

        reference_dtype = (
            reference_data[column].dtype
        )

        current_dtype = (
            current_data[column].dtype
        )

        if reference_dtype != current_dtype:

            validation[
                "dtype_mismatches"
            ][column] = {
                "reference": str(reference_dtype),
                "current": str(current_dtype),
            }

    # --------------------------------------------
    # Missing-value analysis
    # --------------------------------------------

    for column in reference_data.columns:

        missing_count = int(
            reference_data[column].isna().sum()
        )

        if missing_count > 0:

            validation[
                "reference_missing_values"
            ][column] = {
                "count": missing_count,
                "rate": float(
                    missing_count
                    / len(reference_data)
                ),
            }

    for column in current_data.columns:

        missing_count = int(
            current_data[column].isna().sum()
        )

        if missing_count > 0:

            validation[
                "current_missing_values"
            ][column] = {
                "count": missing_count,
                "rate": float(
                    missing_count
                    / len(current_data)
                ),
            }

    # --------------------------------------------
    # Sample-size validation
    # --------------------------------------------

    for column in reference_data.columns:

        non_null_count = int(
            reference_data[column].notna().sum()
        )

        if non_null_count < min_samples:

            validation[
                "small_reference_features"
            ].append(column)

    for column in current_data.columns:

        non_null_count = int(
            current_data[column].notna().sum()
        )

        if non_null_count < min_samples:

            validation[
                "small_current_features"
            ].append(column)

    # --------------------------------------------
    # Constant-feature validation
    # --------------------------------------------

    for column in reference_data.columns:

        unique_count = (
            reference_data[column]
            .nunique(dropna=True)
        )

        if unique_count <= 1:

            validation[
                "reference_constant_features"
            ].append(column)

    for column in current_data.columns:

        unique_count = (
            current_data[column]
            .nunique(dropna=True)
        )

        if unique_count <= 1:

            validation[
                "current_constant_features"
            ].append(column)

    return validation