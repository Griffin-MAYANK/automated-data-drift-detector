"""
Central configuration for the automated data drift detector.

Keeping configuration in one file makes the system easier to
maintain, test, and modify without changing the core logic.
"""

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class FeatureType(Enum):
    """Semantic type used to select a feature's drift method."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    CYCLICAL = "cyclical"


FEATURE_TYPE_METADATA = {
    "quantity": FeatureType.NUMERICAL,
    "unit_price": FeatureType.NUMERICAL,
    "transaction_value": FeatureType.NUMERICAL,
    "transaction_hour": FeatureType.CYCLICAL,
    "day_of_week": FeatureType.CYCLICAL,
    "month": FeatureType.CYCLICAL,
    "is_cancelled": FeatureType.CATEGORICAL,
    "country": FeatureType.CATEGORICAL,
}


def resolve_feature_type(
    feature_name: str,
    data: pd.Series,
) -> FeatureType:
    """Resolve semantic type, preferring metadata over pandas dtype.

    Features without explicit metadata retain the previous dtype-based
    fallback. Cyclical features currently use categorical drift statistics,
    while retaining their semantic type for future circular methods.
    """

    if feature_name in FEATURE_TYPE_METADATA:
        return FEATURE_TYPE_METADATA[feature_name]

    if pd.api.types.is_numeric_dtype(data):
        return FeatureType.NUMERICAL

    return FeatureType.CATEGORICAL


@dataclass(frozen=True)
class DriftConfig:
    """
    Configuration settings for drift detection.

    Attributes
    ----------
    significance_threshold : float
        Statistical significance threshold used by tests.

    min_samples : int
        Minimum valid sample size required for each feature.

    numerical_low_threshold : float
        Upper limit for LOW numerical drift magnitude.

    numerical_medium_threshold : float
        Upper limit for MEDIUM numerical drift magnitude.

    categorical_low_threshold : float
        Upper limit for LOW categorical drift magnitude.

    categorical_medium_threshold : float
        Upper limit for MEDIUM categorical drift magnitude.
    """

    significance_threshold: float = 0.05

    min_samples: int = 30

    numerical_low_threshold: float = 0.10

    numerical_medium_threshold: float = 0.50

    categorical_low_threshold: float = 0.05

    categorical_medium_threshold: float = 0.20

    def __post_init__(self):
        """
        Validate configuration values after initialization.
        """

        if not 0 < self.significance_threshold < 1:
            raise ValueError(
                "significance_threshold must be between 0 and 1."
            )

        if self.min_samples < 1:
            raise ValueError(
                "min_samples must be at least 1."
            )

        if self.numerical_low_threshold < 0:
            raise ValueError(
                "numerical_low_threshold cannot be negative."
            )

        if (
            self.numerical_medium_threshold
            <= self.numerical_low_threshold
        ):
            raise ValueError(
                "numerical_medium_threshold must be greater "
                "than numerical_low_threshold."
            )

        if self.categorical_low_threshold < 0:
            raise ValueError(
                "categorical_low_threshold cannot be negative."
            )

        if (
            self.categorical_medium_threshold
            <= self.categorical_low_threshold
        ):
            raise ValueError(
                "categorical_medium_threshold must be greater "
                "than categorical_low_threshold."
            )


DEFAULT_CONFIG = DriftConfig()