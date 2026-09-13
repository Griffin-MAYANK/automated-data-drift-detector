"""
Statistical functions for automated data drift detection.

This module contains:

1. Numerical drift detection using:
   - Kolmogorov-Smirnov Test
   - Wasserstein Distance

2. Categorical drift detection using:
   - Chi-Square Test
   - Total Variation Distance

3. Numerical and categorical severity classification
"""

import numpy as np
import pandas as pd

from scipy.stats import (
    chi2_contingency,
    ks_2samp,
    wasserstein_distance,
)


def calculate_total_variation_distance(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """
    Calculate Total Variation Distance between
    two categorical distributions.

    TVD ranges from:

        0 = identical distributions

        1 = completely different distributions

    Missing values are treated as a separate category.
    """

    reference_clean = reference.fillna("__MISSING__")
    current_clean = current.fillna("__MISSING__")

    reference_distribution = (
        reference_clean.value_counts(normalize=True)
    )

    current_distribution = (
        current_clean.value_counts(normalize=True)
    )

    categories = (
        set(reference_distribution.index)
        | set(current_distribution.index)
    )

    reference_distribution = (
        reference_distribution.reindex(
            categories,
            fill_value=0,
        )
    )

    current_distribution = (
        current_distribution.reindex(
            categories,
            fill_value=0,
        )
    )

    tvd = 0.5 * np.sum(
        np.abs(
            reference_distribution
            - current_distribution
        )
    )

    return float(tvd)


def classify_numerical_severity(
    p_value: float,
    normalized_magnitude: float,
    threshold: float = 0.05,
) -> str:
    """
    Classify numerical drift severity.

    Rules:

        p-value >= threshold
            -> NO_DRIFT

        normalized magnitude < 0.10
            -> LOW

        normalized magnitude < 0.50
            -> MEDIUM

        normalized magnitude >= 0.50
            -> HIGH

    These magnitude thresholds are initial project
    thresholds and are not universal standards.
    """

    if pd.isna(p_value):
        return "INSUFFICIENT_DATA"

    if p_value >= threshold:
        return "NO_DRIFT"

    if pd.isna(normalized_magnitude):
        return "INSUFFICIENT_DATA"

    if normalized_magnitude < 0.10:
        return "LOW"

    if normalized_magnitude < 0.50:
        return "MEDIUM"

    return "HIGH"


def classify_categorical_severity(
    p_value: float,
    tvd: float,
    threshold: float = 0.05,
) -> str:
    """
    Classify categorical drift severity.

    Rules:

        p-value >= threshold
            -> NO_DRIFT

        TVD < 0.05
            -> LOW

        TVD < 0.20
            -> MEDIUM

        TVD >= 0.20
            -> HIGH

    These TVD thresholds are initial project
    thresholds and are not universal standards.
    """

    if pd.isna(p_value):
        return "INSUFFICIENT_DATA"

    if p_value >= threshold:
        return "NO_DRIFT"

    if pd.isna(tvd):
        return "INSUFFICIENT_DATA"

    if tvd < 0.05:
        return "LOW"

    if tvd < 0.20:
        return "MEDIUM"

    return "HIGH"


def detect_numerical_drift(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = 0.05,
    min_samples: int = 30,
) -> dict:
    """
    Detect drift in a numerical feature.

    Statistical test:
        Kolmogorov-Smirnov two-sample test.

    Practical magnitude:
        Wasserstein distance.

    Normalized magnitude:
        Wasserstein distance divided by the
        reference standard deviation.

    Missing values are removed before testing.
    """

    reference_clean = reference.dropna()
    current_clean = current.dropna()

    reference_sample_size = len(reference_clean)
    current_sample_size = len(current_clean)

    if (
        reference_sample_size < min_samples
        or current_sample_size < min_samples
    ):
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "magnitude": np.nan,
            "normalized_magnitude": np.nan,
            "drift_detected": False,
            "reference_sample_size": reference_sample_size,
            "current_sample_size": current_sample_size,
            "status": "INSUFFICIENT_DATA",
        }

    # Asymptotic p-values avoid SciPy's exact-method fallback warnings and
    # scale predictably for production-sized monitoring samples.
    ks_statistic, p_value = ks_2samp(
        reference_clean,
        current_clean,
        method="asymp",
    )

    wasserstein = wasserstein_distance(
        reference_clean,
        current_clean,
    )

    reference_std = reference_clean.std()

    if reference_std > 0:
        normalized_magnitude = (
            wasserstein / reference_std
        )
    else:
        reference_median = reference_clean.median()
        current_median = current_clean.median()

        normalized_magnitude = abs(
            current_median - reference_median
        )

    drift_detected = p_value < threshold

    return {
        "statistic": float(ks_statistic),
        "p_value": float(p_value),
        "magnitude": float(wasserstein),
        "normalized_magnitude": float(
            normalized_magnitude
        ),
        "drift_detected": bool(drift_detected),
        "reference_sample_size": reference_sample_size,
        "current_sample_size": current_sample_size,
        "status": "OK",
    }


def detect_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = 0.05,
    min_samples: int = 30,
) -> dict:
    """
    Detect drift in a categorical feature.

    Statistical test:
        Chi-Square test.

    Practical magnitude:
        Total Variation Distance.

    Missing values are treated as a separate category.
    """

    reference_clean = reference.fillna("__MISSING__")
    current_clean = current.fillna("__MISSING__")

    reference_sample_size = len(reference_clean)
    current_sample_size = len(current_clean)

    if (
        reference_sample_size < min_samples
        or current_sample_size < min_samples
    ):
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "magnitude": np.nan,
            "drift_detected": False,
            "severity": "INSUFFICIENT_DATA",
            "reference_sample_size": reference_sample_size,
            "current_sample_size": current_sample_size,
            "status": "INSUFFICIENT_DATA",
        }

    reference_counts = reference_clean.value_counts()
    current_counts = current_clean.value_counts()

    categories = (
        set(reference_counts.index)
        | set(current_counts.index)
    )

    reference_counts = reference_counts.reindex(
        categories,
        fill_value=0,
    )

    current_counts = current_counts.reindex(
        categories,
        fill_value=0,
    )

    contingency_table = pd.DataFrame({
        "reference": reference_counts,
        "current": current_counts,
    })

    contingency_table = contingency_table.to_numpy(dtype=float)

    chi2, p_value, _, _ = chi2_contingency(
         contingency_table
    )

    reference_distribution = (
        reference_counts / reference_counts.sum()
    )

    current_distribution = (
        current_counts / current_counts.sum()
    )

    tvd = 0.5 * np.sum(
        np.abs(
            reference_distribution
            - current_distribution
        )
    )

    drift_detected = p_value < threshold

    severity = classify_categorical_severity(
        p_value=p_value,
        tvd=tvd,
        threshold=threshold,
    )

    return {
        "statistic": float(chi2),
        "p_value": float(p_value),
        "magnitude": float(tvd),
        "drift_detected": bool(drift_detected),
        "severity": severity,
        "reference_sample_size": reference_sample_size,
        "current_sample_size": current_sample_size,
        "status": "OK",
    }
