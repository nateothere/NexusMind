"""Shared errors and small text/id helpers."""

from __future__ import annotations

import re
import time
import uuid
from typing import Iterable

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


class NexusMindError(Exception):
    """Base class for all NexusMind errors."""


class ConfigError(NexusMindError):
    """Invalid configuration."""


class ModelUnavailableError(NexusMindError):
    """The LLM or embedding endpoint is not reachable."""


class IngestionError(NexusMindError):
    """A document could not be parsed or indexed."""


class RetrievalError(NexusMindError):
    """Retrieval failed."""


class ToolError(NexusMindError):
    """A tool raised while executing."""


def new_id(prefix: str) -> str:
    """Return a short random id with a readable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return _TOKEN_RE.findall(text.lower())


def token_count(text: str) -> int:
    """Rough token estimate (whitespace + punctuation aware)."""
    return len(_TOKEN_RE.findall(text))


def now_ms() -> int:
    """Current epoch milliseconds."""
    return int(time.time() * 1000)


def unique(iterable: Iterable[str]) -> list[str]:
    """Return items in order, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out