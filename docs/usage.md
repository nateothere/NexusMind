# CLI usage

NexusMind ships one binary: `nexusmind`.

## Commands

| Command                              | Purpose                                   |
| ------------------------------------ | ----------------------------------------- |
| `nexusmind ingest <path> [--recursive] [--collection NAME]` | Parse → chunk → embed → graph |
| `nexusmind ask "question" [--citations] [--top-k N] [--rerank]` | Ask once, stream answer |
| `nexusmind chat`                     | Interactive TUI (Rich), persistent conversation |
| `nexusmind serve --host 0.0.0.0 --port 8000` | FastAPI server + UI            |
| `nexusmind eval run --suite rag --dataset evals/golden.jsonl` | Eval harness   |
| `nexusmind mcp serve [--http]`       | MCP server (stdio or streamable HTTP)     |
| `nexusmind tools connect --server URL` | Attach external MCP tools               |

## Examples

```bash
# Ingest a folder, then ask a multi-hop question with citations
nexusmind ingest ./knowledge-base --recursive
nexusmind ask "Which API changes break contract Y?" --citations

# Stream from the API instead
curl -N -X POST http://localhost:8000/v1/ask \
  -H "Authorization: Bearer $NEXUSMIND_API_KEY" \
  -d '{"query": "Summarize the Q3 migration plan.", "stream": true}'

# Run the eval gate before a retrieval PR
nexusmind eval run --suite rag --dataset ./evals/golden.jsonl --output report.html
```

## Environment

All settings come from `.env` / environment variables with the
`NEXUSMIND_` prefix — see config.example.yaml for the full table.
## Streaming from the API

When you want token-by-token output without the CLI, hit `/v1/ask` with `stream: true` and consume the SSE events:

```bash
curl -N -X POST http://localhost:8000/v1/ask \
  -H "Authorization: Bearer $NEXUSMIND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the migration plan.", "stream": true}'
```

Events arrive as `token`, `citation` and `done` frames; the `done` frame carries `latency_ms` and total token count.
