"""FastAPI application - streaming /v1/ask, ingestion, health, MCP."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .. import __version__
from ..engine import NexusMind
from ..utils import NexusMindError
from .schemas import AskRequest, HealthResponse, IngestRequest, ToolRegisterRequest


async def sse_events(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for event in events:
        event_type = event.pop("event", "message")
        yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"


def require_key(settings_key: str = ""):
    """Dependency factory: enforce NEXUSMIND_API_KEY for non-loopback."""

    def _dep(request: Request, authorization: str | None = Header(default=None)) -> None:
        if not settings_key:
            return
        host = request.client.host if request.client else ""
        if host in ("127.0.0.1", "::1", "localhost"):
            return
        if authorization != f"Bearer {settings_key}":
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    return _dep


def create_app(engine: NexusMind) -> FastAPI:
    """Build the FastAPI app around an engine instance."""
    app = FastAPI(title="NexusMind", version=__version__)
    auth = require_key(engine.settings.api_key)

    @app.get("/v1/health", response_model=HealthResponse, dependencies=[Depends(auth)])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            llm_configured=True,
            collections=engine.collections(),
            version=__version__,
        )

    @app.post("/v1/ask", dependencies=[Depends(auth)])
    async def ask(body: AskRequest) -> Any:
        try:
            if body.stream:
                events = engine.ask_stream(
                    body.query,
                    top_k=body.top_k,
                    rerank=body.rerank,
                    graph_expansion=body.graph_expansion,
                    collection=body.collection,
                )
                return StreamingResponse(sse_events(events), media_type="text/event-stream")
            answer = await engine.ask(
                body.query,
                top_k=body.top_k,
                rerank=body.rerank,
                graph_expansion=body.graph_expansion,
                collection=body.collection,
            )
            return answer.model_dump()
        except NexusMindError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/ingest", dependencies=[Depends(auth)])
    async def ingest(body: IngestRequest) -> dict[str, object]:
        chunks = engine.ingest(body.source, recursive=body.recursive, collection=body.collection)
        return {"ingested": len(chunks), "collection": body.collection}

    @app.get("/v1/collections", dependencies=[Depends(auth)])
    async def collections() -> dict[str, object]:
        return {"collections": engine.collections()}

    @app.get("/v1/documents/{doc_id}", dependencies=[Depends(auth)])
    async def document(doc_id: str) -> dict[str, object]:
        doc = engine.document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        return doc

    @app.delete("/v1/documents/{doc_id}", dependencies=[Depends(auth)])
    async def delete_document(doc_id: str) -> dict[str, object]:
        engine.delete_document(doc_id)
        return {"deleted": doc_id}

    @app.post("/v1/tools", dependencies=[Depends(auth)])
    async def register_tool(body: ToolRegisterRequest) -> dict[str, object]:
        engine.register_tool_by_schema(body.name, body.description, body.schema)
        return {"registered": body.name}

    return app