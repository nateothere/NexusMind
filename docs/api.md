# API

NexusMind exposes a FastAPI service. Run `nexusmind serve` or the Docker
stack, then open http://localhost:8000/docs.

## Authentication

Production requires `NEXUSMIND_API_KEY`. Send it as a Bearer token:

```bash
curl -H "Authorization: Bearer $NEXUSMIND_API_KEY" http://localhost:8000/v1/health
```

## Endpoints

### POST /v1/ask

Ask a question. `stream: true` returns SSE events (`token`, `citation`,
`done`); `stream: false` returns a JSON `Answer`.

```bash
curl -N -X POST http://localhost:8000/v1/ask \
  -H "Authorization: Bearer $NEXUSMIND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What changed between v2 and v3?", "stream": true}'
```

### POST /v1/ingest

Queue an ingestion job for a file, URL or folder.

```json
{"source": "./knowledge-base", "recursive": true, "collection": "default"}
```

### GET /v1/collections

List collections with chunk counts.

### GET /v1/documents/{id} · DELETE /v1/documents/{id}

Document metadata + chunk map; removal also deletes graph entities.

### POST /v1/tools

Register a runtime tool (name, description, JSON schema).

### GET /v1/health

Liveness + model status (LLM, embeddings reachable?).

### GET /mcp

MCP streamable-HTTP endpoint for MCP clients.

## SSE event shape

```json
{"event": "token",    "data": {"delta": "The", "answer_id": "a1"}}
{"event": "citation", "data": {"index": 0, "score": 0.93, "source": "docs/plan.md", "page": 3}}
{"event": "done",     "data": {"answer_id": "a1", "latency_ms": 4120, "tokens": 312}}
```
## Error handling

All endpoints return the standard FastAPI error envelope (`{"detail": ...}`). `422` means validation or pipeline failure (for example, the LLM endpoint is unreachable); `401` means the API key is missing or wrong. When the worker is down, `POST /v1/ingest` still accepts the job and the API responds with `202` semantics on the next poll of the collection.
