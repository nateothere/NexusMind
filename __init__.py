"""NexusMind - local-first Agentic RAG engine."""

from __future__ import annotations

from .config import AgentsConfig, IngestionConfig, MemoryConfig, RetrieverConfig, Settings
from .engine import NexusMind, Answer, Citation

__version__ = "0.3.0"
__all__ = [
    "NexusMind",
    "Answer",
    "Citation",
    "Settings",
    "RetrieverConfig",
    "AgentsConfig",
    "IngestionConfig",
    "MemoryConfig",
    "__version__",
]