"""Observability - OpenTelemetry hooks with no-op fallback."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any, TypeVar, cast

import structlog

_T = TypeVar("_T")


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog JSON logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)


