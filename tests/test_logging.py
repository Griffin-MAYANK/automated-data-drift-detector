import io
import logging
import uuid

from fastapi.testclient import TestClient

from drift_detector.api import app
from drift_detector.logging_config import (
    configure_logging,
    get_logger,
)


def test_configure_logging_emits_formatted_records():
    stream = io.StringIO()
    logger = configure_logging(level=logging.INFO, stream=stream)

    logger.info("test event")

    output = stream.getvalue()
    assert "INFO" in output
    assert "drift_detector" in output
    assert "test event" in output
    assert output[:4].count("-") == 0


def test_configure_logging_does_not_duplicate_handlers():
    logger = configure_logging(stream=io.StringIO())
    handler_count = len(logger.handlers)

    configure_logging(stream=io.StringIO())

    assert len(logger.handlers) == handler_count


def test_request_id_is_generated_and_logged():
    stream = io.StringIO()
    configure_logging(stream=stream)

    response = TestClient(app).get("/health")

    request_id = response.headers["X-Request-ID"]
    uuid.UUID(request_id)
    assert response.status_code == 200
    assert f"request_id={request_id}" in stream.getvalue()
    assert "status_code=200" in stream.getvalue()


def test_request_id_is_preserved():
    request_id = "trace-123"

    response = TestClient(app).get(
        "/health",
        headers={"X-Request-ID": request_id},
    )

    assert response.headers["X-Request-ID"] == request_id


def test_invalid_request_is_logged_as_warning():
    stream = io.StringIO()
    configure_logging(stream=stream)

    response = TestClient(app).post("/detect", json={})

    assert response.status_code == 422
    assert "WARNING" in stream.getvalue()
    assert "status_code=422" in stream.getvalue()


def test_application_logger_names_are_namespaced():
    assert get_logger("test").name == "drift_detector.test"