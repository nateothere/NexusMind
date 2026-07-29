# Changelog

All notable changes to NexusMind are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Late chunking prototype behind `retriever.late_chunking` flag
- ColBERT-style late-interaction scoring in the eval harness

### Changed
- CLI `ask` now emits ANSI-coded citations by default

## [0.3.0] - 2025-11-18

### Added
- MCP server over stdio (`nexusmind mcp serve`) and streamable HTTP (`--http`)
- `nexusmind tools connect` to attach external MCP tools as agent tools
- Episodic memory: long-term answers written back into the vector index
- Structured outputs: JSON-schema constrained decoding for every agent step
- `GET /v1/health` model status and `GET /mcp` streamable-HTTP endpoint
- OpenTelemetry spans for parse → retrieve → rerank → agent steps
- Optional Langfuse tracing via `NEXUSMIND_LANGFUSE_KEY`

### Fixed
- RRF fusion now normalizes per-index scores before merging (was skewing
  dense candidates with very high similarities)
- Semantic chunker no longer drops single-paragraph documents

## [0.2.0] - 2025-06-30

### Added
- GraphRAG v1: LLM entity/relation extraction into an embedded Kuzu graph
- 1-hop entity expansion at query time (`retriever.graph_expansion`)
- Multi-agent runtime: Planner / Researcher / Critic / Writer with
  per-agent token budgets
- `@tool` decorator with automatic JSON-schema generation (Pydantic v2)
- RAGAS-style eval harness (`nexusmind eval run`) with golden datasets
- Docker Compose stack: api + ui + worker + qdrant + redis + ollama

### Changed
- Hybrid retrieval now fuses BM25 + dense via RRF before reranking

## [0.1.0] - 2024-12-20

### Added
- Ingestion pipeline: Markdown/text parsing, recursive + semantic chunking
- Hybrid retriever: pure-Python BM25 + Ollama embeddings (bge-m3)
- Reciprocal Rank Fusion and optional cross-encoder reranking
- FastAPI server with SSE streaming for `/v1/ask`
- Typer CLI: `ingest`, `ask`, `chat`, `serve`, `eval`
- Pydantic Settings + YAML configuration

[Unreleased]: https://github.com/nexusmind-ai/nexusmind/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/nexusmind-ai/nexusmind/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nexusmind-ai/nexusmind/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nexusmind-ai/nexusmind/releases/tag/v0.1.0