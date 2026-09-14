"""Pydantic request and response models for the drift detection API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DetectionRequest(BaseModel):
    """Validated inputs for one reference/current drift analysis."""

    data_path: str = Field(..., min_length=1)
    split_date: date
    significance_threshold: float = Field(default=0.05, gt=0, lt=1)
    min_samples: int = Field(default=30, ge=1)
    output_directory: str = Field(default="reports", min_length=1)

    @field_validator("data_path", "output_directory")
    @classmethod
    def reject_blank_paths(cls, value: str) -> str:
        """Reject whitespace-only path values."""

        value = value.strip()
        if not value:
            raise ValueError("path must not be empty")
        return value


class ErrorResponse(BaseModel):
    """Stable, safe error response returned by the API."""

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str
    request_id: str


class HealthResponse(BaseModel):
    """Health endpoint response."""

    model_config = ConfigDict(extra="forbid")

    status: str


class MetadataResponse(BaseModel):
    """Supported detector metadata exposed by the API."""

    model_config = ConfigDict(extra="forbid")

    project_name: str
    api_version: str
    detector_version: str
    supported_feature_types: list[str]
    monitoring_features: list[str]


class ReportPaths(BaseModel):
    """Safe report names returned by the API."""

    model_config = ConfigDict(populate_by_name=True)

    csv: str
    json_path: str = Field(alias="json")


class FeatureResult(BaseModel):
    """One feature result using the existing report column schema."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    data_type: str
    test: str
    p_value: float | None
    adjusted_p_value: float | None
    magnitude: float | None
    normalized_magnitude: float | None
    drift_detected: bool
    severity: str
    statistical_significance: bool
    fdr_significant: bool
    practical_magnitude: str
    interpretation: str
    final_decision: str
    reference_sample_size: int
    current_sample_size: int
    reference_missing_rate: float
    current_missing_rate: float
    missing_rate_difference: float
    status: str


class DetectionResponse(BaseModel):
    """Complete API response for a drift detection run."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    split_date: date
    reference_rows: int
    current_rows: int
    number_of_features: int
    overall_status: str
    alert_count: int
    investigate_count: int
    monitor_count: int
    no_drift_count: int
    insufficient_data_count: int
    report_paths: ReportPaths
    features: list[FeatureResult]