# NexusMind
**The local-first Agentic RAG engine.**

GraphRAG · Hybrid Retrieval · Multi-Agent Orchestration · MCP-Native · 100% Local Inference

```text
                      ┌─────────────────────────────────────────────┐
                      │            INGESTION (local only)           │
                      │  PDF · DOCX · MD · URL · Audio · Code       │
                      │        │  parser / chunker / enricher       │
                      │        ▼                                    │
                      │  ┌─────────┐  ┌──────────┐  ┌──────────┐   │
                      │  │ Qdrant  │  │  Kuzu    │  │ Tantivy  │   │
                      │  │ dense   │  │ graph    │  │ BM25     │   │
                      │  └─────────┘  └──────────┘  └──────────┘   │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                      ┌─────────────────────────────────────────────┐
                      │            QUERY (100% on your box)         │
                      │  Question ──► Hybrid Retriever ──► RRF      │
                      │      │              └──► rerank ─► 1-hop    │
                      │      ▼                                      │
                      │  Planner ─► Researcher ─► Critic ─► Writer  │
                      │                              └──► SSE stream│
                      └─────────────────────────────────────────────┘
```

## What it is

NexusMind runs the entire RAG pipeline — parsing, embedding, retrieval,
graph expansion, agent reasoning, generation — on your hardware. Zero data
egress. It pairs a hybrid retriever (BM25 + dense + Reciprocal Rank Fusion
+ cross-encoder rerank) with a knowledge graph (entity/relation extraction,
1-hop expansion) and an async multi-agent runtime (Planner → Researcher →
Critic → Writer) with typed tools and token budgets.

## Highlights

- GraphRAG: entities and relations extracted into an embedded graph;
  query-time expansion across entity relationships for multi-hop questions
- Hybrid retrieval: pure-Python BM25 + Ollama embeddings (bge-m3), fused
  with RRF, optional bge-reranker-v2-m3 cross-encoder pass
- Multi-agent runtime: async Planner / Researcher / Critic / Writer with
  per-agent token budgets and structured outputs
- MCP-native: serve your knowledge base as an MCP server, or attach
  external MCP tools to the agents
- Streaming: token-level SSE with inline citations, CLI TUI via Rich
- Eval harness: RAGAS-style metrics (faithfulness, relevancy, context
  precision/recall) gating retrieval changes
- Deploy anywhere: Docker Compose (api + ui + worker + qdrant + redis +
  ollama), GPU profiles, `make dev`

## Quickstart

```bash
make setup          # venv + deps + pre-commit + ollama models
make dev            # api :8000 + ui :8501 + worker
nexusmind ingest ./knowledge-base --recursive
nexusmind ask "What changed between v2 and v3 of the API?" --citations
```

Docker: `docker compose up -d` boots the full stack.

## Docs

- [Architecture](ARCHITECTURE.md)
- [Config](config.example.yaml)
- [CLI](docs/usage.md)
- [API](docs/api.md)
- [FAQ](FAQ.md)
- [Roadmap](ROADMAP.md)

## License

MIT — see LICENSE. Model weights carry their own licenses (NOTICE).