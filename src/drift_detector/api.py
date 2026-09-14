"""FastAPI application for the automated data drift detector."""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Sequence

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from drift_detector import __version__
from drift_detector.api_models import (
    DetectionRequest,
    DetectionResponse,
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
        raise

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

    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.exception_handler(UnsupportedDatasetError)
async def invalid_dataset_handler(
    request: Request,
    error: UnsupportedDatasetError,
) -> JSONResponse:
    """Return a safe 400 response for invalid datasets."""

    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.exception_handler(InvalidDatasetError)
async def invalid_dataset_content_handler(
    request: Request,
    error: InvalidDatasetError,
) -> JSONResponse:
    """Return a safe 400 response for invalid dataset content."""

    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Avoid exposing tracebacks or filesystem details to API clients."""

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
)
def health() -> HealthResponse:
    """Return the service health status."""

    return HealthResponse(status="healthy")


@app.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Get detector metadata",
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