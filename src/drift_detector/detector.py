"""
Core production engine for automated data drift detection.

This module combines:

1. Dataset validation
2. Numerical drift detection
3. Categorical drift detection
4. Practical magnitude classification
5. Feature-level drift reporting
"""

from typing import Optional

import numpy as np
import pandas as pd

from scipy.stats import false_discovery_control

from drift_detector.config import (
    DEFAULT_CONFIG,
    FeatureType,
    DriftConfig,
    resolve_feature_type,
)
from drift_detector.statistics import (
    classify_categorical_severity,
    classify_numerical_severity,
    detect_categorical_drift,
    detect_numerical_drift,
)
from drift_detector.validation import validate_dataset_pair


def classify_practical_magnitude(
    row: pd.Series,
    config: DriftConfig = DEFAULT_CONFIG,
) -> str:
    """
    Classify the practical size of detected drift.

    Numerical features use normalized Wasserstein distance.

    Categorical features use Total Variation Distance.
    """

    if row["status"] != "OK":
        return "INSUFFICIENT_DATA"

    if not row["drift_detected"]:
        return "NO_DRIFT"

    if row["data_type"] == "numerical":

        magnitude = row["normalized_magnitude"]

        if pd.isna(magnitude):
            return "INSUFFICIENT_DATA"

        if magnitude < config.numerical_low_threshold:
            return "VERY_SMALL"

        if magnitude < config.numerical_medium_threshold:
            return "MODERATE"

        return "LARGE"

    magnitude = row["magnitude"]

    if pd.isna(magnitude):
        return "INSUFFICIENT_DATA"

    if magnitude < config.categorical_low_threshold:
        return "VERY_SMALL"

    if magnitude < config.categorical_medium_threshold:
        return "MODERATE"

    return "LARGE"


def classify_drift_interpretation(
    row: pd.Series,
) -> str:
    """
    Combine statistical significance and practical magnitude.
    """

    if row["status"] != "OK":
        return "INSUFFICIENT_DATA"

    if not row["drift_detected"]:
        return "NO_DRIFT"

    practical_magnitude = row["practical_magnitude"]

    if practical_magnitude == "VERY_SMALL":
        return "STATISTICAL_ONLY"

    if practical_magnitude == "MODERATE":
        return "ACTIONABLE"

    if practical_magnitude == "LARGE":
        return "HIGH_IMPACT"

    return "UNKNOWN"


def determine_final_decision(
    row: pd.Series,
) -> str:
    """
    Convert statistical and practical analysis into
    an operational monitoring decision.

    Decision rules:

        Invalid data
            -> INSUFFICIENT_DATA

        No FDR-significant drift
            -> NO_DRIFT

        Very small drift
            -> MONITOR

        Moderate drift
            -> INVESTIGATE

        Large drift
            -> ALERT
    """

    if row["status"] != "OK":
        return "INSUFFICIENT_DATA"

    if not row["fdr_significant"]:
        return "NO_DRIFT"

    if row["practical_magnitude"] == "VERY_SMALL":
        return "MONITOR"

    if row["practical_magnitude"] == "MODERATE":
        return "INVESTIGATE"

    if row["practical_magnitude"] == "LARGE":
        return "ALERT"

    return "UNKNOWN"


def apply_multiple_testing_correction(
    report: pd.DataFrame,
    method: str = "bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply multiple-testing correction to feature p-values.

    Parameters
    ----------
    report : pandas.DataFrame
        Feature-level drift report.

    method : str, default="bh"
        Multiple-testing correction method supported
        by scipy.stats.false_discovery_control.

    alpha : float, default=0.05
        Significance level after correction.

    Returns
    -------
    pandas.DataFrame
        Report with adjusted p-values and FDR decisions.
    """

    if report.empty:
        raise ValueError(
            "Cannot apply multiple-testing correction "
            "to an empty report."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    corrected_report = report.copy()

    corrected_report["adjusted_p_value"] = np.nan
    corrected_report["fdr_significant"] = False

    valid_mask = corrected_report["p_value"].notna()

    if valid_mask.any():

        valid_p_values = corrected_report.loc[
            valid_mask,
            "p_value",
        ]

        adjusted_p_values = false_discovery_control(
            valid_p_values.to_numpy(),
            method=method,
        )

        corrected_report.loc[
            valid_mask,
            "adjusted_p_value",
        ] = adjusted_p_values

        corrected_report.loc[
            valid_mask,
            "fdr_significant",
        ] = (
            corrected_report.loc[
                valid_mask,
                "adjusted_p_value",
            ]
            < alpha
        )

    return corrected_report


def detect_dataset_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    config: DriftConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Run the complete production drift-detection pipeline.

    Pipeline:

        Validation
            ↓
        Feature processing
            ↓
        Statistical testing
            ↓
        Practical magnitude
            ↓
        FDR correction
            ↓
        Operational decision
            ↓
        Final report
    """

    if not isinstance(reference_data, pd.DataFrame):
        raise TypeError(
            "reference_data must be a pandas DataFrame."
        )

    if not isinstance(current_data, pd.DataFrame):
        raise TypeError(
            "current_data must be a pandas DataFrame."
        )

    validation = validate_dataset_pair(
        reference_data,
        current_data,
        min_samples=config.min_samples,
    )

    if validation["reference_empty"]:
        raise ValueError(
            "Reference dataset is empty."
        )

    if validation["current_empty"]:
        raise ValueError(
            "Current dataset is empty."
        )

    if validation["missing_columns"]:
        raise ValueError(
            "Current dataset is missing columns: "
            f"{validation['missing_columns']}"
        )

    if validation["extra_columns"]:
        raise ValueError(
            "Current dataset contains unexpected columns: "
            f"{validation['extra_columns']}"
        )

    if validation["dtype_mismatches"]:
        raise TypeError(
            "Data-type mismatch detected: "
            f"{validation['dtype_mismatches']}"
        )

    results = []

    for column in reference_data.columns:

        reference = reference_data[column]
        current = current_data[column]

        reference_missing_count = int(
            reference.isna().sum()
        )

        current_missing_count = int(
            current.isna().sum()
        )

        reference_missing_rate = (
            reference_missing_count
            / len(reference)
        )

        current_missing_rate = (
            current_missing_count
            / len(current)
        )

        missing_rate_difference = (
            current_missing_rate
            - reference_missing_rate
        )

        feature_type = resolve_feature_type(
            feature_name=column,
            data=reference,
        )

        if feature_type is FeatureType.NUMERICAL:

            drift_result = detect_numerical_drift(
                reference=reference,
                current=current,
                threshold=config.significance_threshold,
                min_samples=config.min_samples,
            )

            result = {
                "feature": column,
                "data_type": "numerical",
                "test": "KS Test",
                "statistic": drift_result["statistic"],
                "p_value": drift_result["p_value"],
                "magnitude": drift_result["magnitude"],
                "normalized_magnitude": (
                    drift_result["normalized_magnitude"]
                ),
                "drift_detected": (
                    drift_result["drift_detected"]
                ),
                "severity": (
                    classify_numerical_severity(
                        p_value=drift_result["p_value"],
                        normalized_magnitude=(
                            drift_result["normalized_magnitude"]
                        ),
                        threshold=config.significance_threshold,
                    )
                    if drift_result["status"] == "OK"
                    else "INSUFFICIENT_DATA"
                ),
                "reference_sample_size": (
                    drift_result["reference_sample_size"]
                ),
                "current_sample_size": (
                    drift_result["current_sample_size"]
                ),
                "reference_missing_rate": (
                    reference_missing_rate
                ),
                "current_missing_rate": (
                    current_missing_rate
                ),
                "missing_rate_difference": (
                    missing_rate_difference
                ),
                "status": drift_result["status"],
            }

        else:

            drift_result = detect_categorical_drift(
                reference=reference,
                current=current,
                threshold=config.significance_threshold,
                min_samples=config.min_samples,
            )

            result = {
                "feature": column,
                "data_type": "categorical",
                "test": "Chi-Square Test",
                "statistic": drift_result["statistic"],
                "p_value": drift_result["p_value"],
                "magnitude": drift_result["magnitude"],
                "normalized_magnitude": np.nan,
                "drift_detected": (
                    drift_result["drift_detected"]
                ),
                "severity": drift_result["severity"],
                "reference_sample_size": (
                    drift_result["reference_sample_size"]
                ),
                "current_sample_size": (
                    drift_result["current_sample_size"]
                ),
                "reference_missing_rate": (
                    reference_missing_rate
                ),
                "current_missing_rate": (
                    current_missing_rate
                ),
                "missing_rate_difference": (
                    missing_rate_difference
                ),
                "status": drift_result["status"],
            }

        results.append(result)

    report = pd.DataFrame(results)

    report["statistical_significance"] = (
        report["p_value"]
        < config.significance_threshold
    )

    report["practical_magnitude"] = report.apply(
        lambda row: classify_practical_magnitude(
            row,
            config=config,
        ),
        axis=1,
    )

    report["interpretation"] = report.apply(
        classify_drift_interpretation,
        axis=1,
    )

    report = apply_multiple_testing_correction(
        report=report,
        method="bh",
        alpha=config.significance_threshold,
    )

    report["final_decision"] = report.apply(
        determine_final_decision,
        axis=1,
    )

    decision_priority = {
        "ALERT": 1,
        "INVESTIGATE": 2,
        "MONITOR": 3,
        "NO_DRIFT": 4,
        "INSUFFICIENT_DATA": 5,
        "UNKNOWN": 6,
    }

    report["priority"] = (
        report["final_decision"]
        .map(decision_priority)
    )

    report = (
        report
        .sort_values(
            by=[
                "priority",
                "adjusted_p_value",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    final_columns = [
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
    ]

    return report[final_columns]