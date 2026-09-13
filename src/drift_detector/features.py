"""
Feature engineering utilities for the automated
data drift detection system.

This module converts raw transaction data into
features suitable for drift monitoring.
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "InvoiceNo",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "Country",
}


MONITORING_FEATURES = [
    "quantity",
    "unit_price",
    "transaction_value",
    "transaction_hour",
    "day_of_week",
    "month",
    "is_cancelled",
    "country",
]


def validate_raw_retail_data(
    data: pd.DataFrame,
) -> None:
    """
    Validate that the raw retail dataset contains
    the columns required for feature engineering.

    Raises
    ------
    ValueError
        If the dataset is empty.

    KeyError
        If required columns are missing.
    """

    if data.empty:
        raise ValueError(
            "Cannot engineer features from an empty dataset."
        )

    missing_columns = (
        REQUIRED_COLUMNS - set(data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def create_monitoring_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw Online Retail transaction data
    into monitoring features.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw Online Retail transaction data.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the monitoring features.

    Features
    --------
    quantity
        Transaction quantity.

    unit_price
        Unit price of the product.

    transaction_value
        Quantity multiplied by unit price.

    transaction_hour
        Hour extracted from InvoiceDate.

    day_of_week
        Day of week extracted from InvoiceDate.
        Monday = 0, Sunday = 6.

    month
        Month extracted from InvoiceDate.

    is_cancelled
        1 if InvoiceNo begins with "C",
        otherwise 0.

    country
        Customer country.
    """

    validate_raw_retail_data(data)

    result = data.copy()

    # --------------------------------------------
    # Ensure InvoiceDate is datetime
    # --------------------------------------------

    result["InvoiceDate"] = pd.to_datetime(
        result["InvoiceDate"],
        errors="coerce",
    )

    if result["InvoiceDate"].isna().all():
        raise ValueError(
            "InvoiceDate could not be converted "
            "to valid datetime values."
        )

    # --------------------------------------------
    # Numerical features
    # --------------------------------------------

    result["Quantity"] = pd.to_numeric(
        result["Quantity"],
        errors="coerce",
    )

    result["UnitPrice"] = pd.to_numeric(
        result["UnitPrice"],
        errors="coerce",
    )

    result["quantity"] = result["Quantity"]

    result["unit_price"] = result["UnitPrice"]

    result["transaction_value"] = (
        result["Quantity"]
        * result["UnitPrice"]
    )

    # --------------------------------------------
    # Time-based features
    # --------------------------------------------

    result["transaction_hour"] = (
        result["InvoiceDate"].dt.hour
    )

    result["day_of_week"] = (
        result["InvoiceDate"].dt.dayofweek
    )

    result["month"] = (
        result["InvoiceDate"].dt.month
    )

    # --------------------------------------------
    # Cancellation feature
    # --------------------------------------------

    result["is_cancelled"] = (
        result["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
        .astype(int)
    )

    # --------------------------------------------
    # Categorical feature
    # --------------------------------------------

    result["country"] = (
        result["Country"]
        .astype("string")
    )

    # --------------------------------------------
    # Keep only monitoring features
    # --------------------------------------------

    features = result[
        MONITORING_FEATURES
    ].copy()

    return features


def split_reference_current(
    data: pd.DataFrame,
    split_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split raw transaction data into reference
    and current periods.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw transaction dataset.

    split_date : str or pandas.Timestamp
        Transactions before this date become
        reference data.

        Transactions on or after this date become
        current data.

    Returns
    -------
    tuple
        reference_data, current_data
    """

    validate_raw_retail_data(data)

    split_date = pd.Timestamp(split_date)

    dates = pd.to_datetime(
        data["InvoiceDate"],
        errors="coerce",
    )

    if dates.isna().all():
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    reference_mask = dates < split_date

    current_mask = dates >= split_date

    reference_data = data.loc[
        reference_mask
    ].copy()

    current_data = data.loc[
        current_mask
    ].copy()

    if reference_data.empty:
        raise ValueError(
            "Reference dataset is empty after splitting."
        )

    if current_data.empty:
        raise ValueError(
            "Current dataset is empty after splitting."
        )

    return reference_data, current_data


def create_reference_current_features(
    data: pd.DataFrame,
    split_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split raw data into reference/current periods
    and engineer monitoring features for both.

    This is the main convenience function for the
    production pipeline.
    """

    reference_raw, current_raw = (
        split_reference_current(
            data,
            split_date,
        )
    )

    reference_features = (
        create_monitoring_features(
            reference_raw
        )
    )

    current_features = (
        create_monitoring_features(
            current_raw
        )
    )

    return (
        reference_features,
        current_features,
    )

