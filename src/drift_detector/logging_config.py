"""Centralized logging configuration for the drift detector application."""

from __future__ import annotations

import logging
import sys
from typing import TextIO


APPLICATION_LOGGER_NAME = "drift_detector"
_DEFAULT_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s %(message)s"
)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return an application logger in the drift detector namespace."""

    if name is None:
        name = APPLICATION_LOGGER_NAME
    elif not name.startswith(f"{APPLICATION_LOGGER_NAME}."):
        name = f"{APPLICATION_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def configure_logging(
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure application logging once and return the root app logger.

    Logs are emitted to stderr by default. Repeated calls update the existing
    application handler instead of adding duplicate handlers.
    """

    logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    handler = next(
        (
            candidate
            for candidate in logger.handlers
            if getattr(candidate, "_drift_detector_handler", False)
        ),
        None,
    )

    if handler is None:
        handler = logging.StreamHandler(stream or sys.stderr)
        setattr(handler, "_drift_detector_handler", True)
        logger.addHandler(handler)
    elif stream is not None:
        handler.setStream(stream)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    return logger


def request_id_is_safe(request_id: str | None) -> bool:
    """Return whether an incoming request ID is safe to preserve."""

    if not request_id or len(request_id) > 128:
        return False
    return all(character.isalnum() or character in "-_." for character in request_id)