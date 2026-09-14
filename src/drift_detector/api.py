"""FastAPI application for the automated data drift detector."""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Sequence

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from drift_detector import __version__
from drift_detector.api_models import (
    DetectionRequest,
    DetectionResponse,
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
)
from drift_detector.config import FeatureType
from drift_detector.features import MONITORING_FEATURES
from drift_detector.logging_config import (
    configure_logging,
    request_id_is_safe,
)
from drift_detector.service import (
    DatasetNotFoundError,
    InvalidDatasetError,
    UnsupportedDatasetError,
    run_detection,
)


API_VERSION = "1.0.0"
logger = configure_logging()

app = FastAPI(
    title="Automated Data Drift Detector API",
    version=API_VERSION,
    description=(
        "REST API for running the packaged reference/current data drift "
        "detection pipeline."
    ),
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log request lifecycle details and attach a request correlation ID."""

    request_id = request.headers.get("X-Request-ID")
    if not request_id_is_safe(request_id):
        request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.error(
            "request failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": "Internal server error.",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )

    response.headers["X-Request-ID"] = request_id
    duration_ms = (time.perf_counter() - started_at) * 1000
    log_method = logger.info
    if response.status_code >= 500:
        log_method = logger.error
    elif response.status_code >= 400:
        log_method = logger.warning
    log_method(
        "request complete request_id=%s method=%s path=%s status_code=%d duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(DatasetNotFoundError)
async def dataset_not_found_handler(
    request: Request,
    error: DatasetNotFoundError,
) -> JSONResponse:
    """Return a safe 404 response for missing datasets."""

    return _error_response(
        status_code=404,
        error_code="dataset_not_found",
        detail=str(error),
        request=request,
    )


@app.exception_handler(UnsupportedDatasetError)
async def invalid_dataset_handler(
    request: Request,
    error: UnsupportedDatasetError,
) -> JSONResponse:
    """Return a safe 400 response for invalid datasets."""

    return _error_response(
        status_code=400,
        error_code="unsupported_dataset",
        detail=str(error),
        request=request,
    )


@app.exception_handler(InvalidDatasetError)
async def invalid_dataset_content_handler(
    request: Request,
    error: InvalidDatasetError,
) -> JSONResponse:
    """Return a safe 400 response for invalid dataset content."""

    return _error_response(
        status_code=400,
        error_code="invalid_dataset",
        detail=str(error),
        request=request,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    """Return a stable 422 response without exposing validation internals."""

    return _error_response(
        status_code=422,
        error_code="invalid_request",
        detail="Request validation failed.",
        request=request,
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Avoid exposing tracebacks or filesystem details to API clients."""

    return _error_response(
        status_code=500,
        error_code="internal_server_error",
        detail="Internal server error.",
        request=request,
    )


def _error_response(
    status_code: int,
    error_code: str,
    detail: str,
    request: Request,
) -> JSONResponse:
    """Build a structured error response with the active request ID."""

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    body = ErrorResponse(
        error=error_code,
        detail=detail,
        request_id=request_id,
    ).model_dump()
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={"X-Request-ID": request_id},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    description="Return the liveness status of the drift detector API.",
    responses={200: {"description": "API is healthy."}},
)
def health() -> HealthResponse:
    """Return the service health status."""

    return HealthResponse(status="healthy")


@app.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Get detector metadata",
    description="Return API, package, and supported monitoring feature metadata.",
    responses={200: {"description": "Metadata returned successfully."}},
)
def metadata() -> MetadataResponse:
    """Return package, API, and supported feature metadata."""

    return MetadataResponse(
        project_name="automated-data-drift-detector",
        api_version=API_VERSION,
        detector_version=__version__,
        supported_feature_types=[feature_type.value for feature_type in FeatureType],
        monitoring_features=list(MONITORING_FEATURES),
    )


@app.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Run drift detection",
    description="Run the existing batch reference/current drift pipeline.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid dataset."},
        404: {"model": ErrorResponse, "description": "Dataset not found."},
        422: {"model": ErrorResponse, "description": "Invalid request."},
        500: {
            "model": ErrorResponse,
            "description": "Internal server error.",
        },
    },
)
def detect(request: DetectionRequest) -> DetectionResponse:
    """Run drift detection using the validated request."""

    return run_detection(request)


def build_parser() -> argparse.ArgumentParser:
    """Build the API server command-line parser."""

    parser = argparse.ArgumentParser(
        prog="drift-detector-api",
        description="Serve the automated data drift detector REST API.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Start the API server on the configured host and port."""

    args = build_parser().parse_args(argv)

    uvicorn.run(
        "drift_detector.api:app",
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    sys.exit(main())