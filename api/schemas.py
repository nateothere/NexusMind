"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """POST /v1/ask body."""

    query: str = Field(min_length=1, max_length=8192)
    stream: bool = True
    top_k: int | None = None
    rerank: bool | None = None
    graph_expansion: int | None = None
    collection: str = "default"


class IngestRequest(BaseModel):
    """POST /v1/ingest body."""

    source: str
    recursive: bool = False
    collection: str = "default"


class ToolRegisterRequest(BaseModel):
    """POST /v1/tools body."""

    name: str
    description: str = ""
    schema: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """GET /v1/health body."""

    status: str
    llm_configured: bool
    collections: list[str]
    version: str