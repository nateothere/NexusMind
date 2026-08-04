# 🧠 NexusMind

**The local-first Agentic RAG engine.**

GraphRAG · Hybrid Retrieval · Multi-Agent Orchestration · MCP-Native · 100% Local Inference

| [Python 3.11+](pyproject.toml) | [License: MIT](LICENSE) | [codecov](.codecov.yml) | [Docker Pulls](https://hub.docker.com) | [Discord](https://discord.gg) | [PRs Welcome](CONTRIBUTING.md) |

[Quickstart](#-quickstart) · [Docs](docs) · [Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) · [FAQ](FAQ.md)

> NexusMind CLI demo — ingest, ask, stream a cited answer
>
> Ingest a document folder, ask a multi-hop question, get a streamed answer with inline citations — all running on your machine.

---

## Table of Contents

- [Why NexusMind?](#-why-nexusmind)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Benchmarks](#-benchmarks)
- [Quickstart](#-quickstart)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [The Agentic Core](#-the-agentic-core)
- [MCP Integration](#-mcp-integration)
- [Multimodal Ingestion](#-multimodal-ingestion)
- [Evaluation Harness](#-evaluation-harness)
- [Observability](#-observability)
- [Project Structure](#-project-structure)
- [REST API](#-rest-api)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Security](#-security)
- [Acknowledgments](#-acknowledgments)
- [Citation](#-citation)
- [License](#-license)

## 🤔 Why NexusMind?

Three problems with today's RAG stack:

1. **Cloud RAG leaks.** Your private documents are sent to third-party APIs. NexusMind runs the entire pipeline — parsing, embedding, generation — on your hardware. Zero data egress.
2. **Naive RAG is shallow.** Vector-only retrieval fails at multi-hop and entity-centric questions ("How does change X in service A affect contract Y with client Z?"). NexusMind builds a knowledge graph alongside your vector index and expands retrieval across entity relationships.
3. **Agents are glue code.** Most "agents" are loops of JSON prompts. NexusMind ships a real async agent runtime — Planner → Researcher → Critic → Writer — with typed tools, budget enforcement, and full traceability.

## ✨ Features

### Retrieval

- 🔀 **Hybrid search** — BM25 (Tantivy) + dense vectors (Qdrant) fused with Reciprocal Rank Fusion (RRF)
- 🎯 **Cross-encoder reranking** — bge-reranker-v2-m3 on the fused candidate set
- 🕸️ **GraphRAG v1** — LLM entity/relation extraction into an embedded Kuzu graph, 1-hop expansion at query time, Leiden community summaries (experimental)
- 🧪 **Query transforms** — HyDE (hypothetical document embeddings), multi-query expansion, query decomposition
- ✂️ **Semantic chunking** — embedding-similarity breakpoints + late chunking (experimental)

### Agents & Models

- 🤖 **Multi-agent runtime** — Planner / Researcher / Critic / Writer with ReAct + Plan-and-Execute, per-agent token budgets
- 🔧 **Typed tool system** — `@tool` decorator with automatic JSON-schema generation and runtime validation (Pydantic v2)
- 🔌 **MCP-native** — expose NexusMind as an MCP server, consume external MCP tools as agent tools
- 🧱 **Structured outputs** — JSON-schema constrained decoding for every agent step
- 🧠 **Two-tier memory** — short-term conversation buffer + long-term episodic memory stored back into the vector index

### Platform

- 🖼️ **Multimodal ingestion** — PDF, DOCX, PPTX, HTML, Markdown, code, images (OCR), audio (Whisper)
- 📡 **Streaming everything** — token-level SSE streaming with streaming citations
- 🛰️ **Observability** — OpenTelemetry traces + optional Langfuse dashboards (tokens, latency, cost)
- 📊 **Eval harness** — RAGAS metrics (faithfulness, answer relevancy, context precision/recall) against golden datasets
- 🔑 **Multi-tenant** — API keys, per-tenant collections, RBAC-ready
- 🐳 **Deploy anywhere** — Docker Compose, GPU profiles, `make dev` single command

## 🏗️ Architecture

```
flowchart TD
    subgraph INGESTION
        A[Sources<br/>PDF · DOCX · MD · URL · Audio · Code] --> B[Docling Parser]
        B --> C[Semantic Chunker]
        C --> D[Enricher<br/>entities · keywords · summaries]
        D --> E[(Qdrant<br/>dense vectors)]
        D --> F[(Kuzu<br/>knowledge graph)]
        D --> G[(Tantivy<br/>BM25 index)]
    end
    subgraph QUERY
        H[User Query] --> I[Query Understanding<br/>classify · rewrite · decompose]
        I --> J{Complex?}
        J -- yes --> K[🧭 Planner Agent]
        J -- no --> L[Hybrid Retriever]
        K --> L
        L --> M[RRF Fusion]
        M --> N[Cross-Encoder Rerank]
        N --> O[Graph Expansion · 1-hop]
        O --> P[🔍 Researcher Agent<br/>ReAct + tools]
        P --> Q[⚖️ Critic Agent<br/>grounding check]
        Q --> R[✍️ Writer Agent]
        R --> S[Streamed Answer<br/>+ citations]
    end
    subgraph MODELS["Local Models"]
        T[(Ollama / vLLM<br/>LLM)]
        U[(bge-m3<br/>embeddings)]
        V[(bge-reranker-v2<br/>reranker)]
    end
    T -.-> I & K & P & Q & R
    U -.-> C & L
    V -.-> N
```

Deep dive: [ARCHITECTURE.md](ARCHITECTURE.md) — pipeline internals, data contracts, and design decisions.

## ⚙️ Tech Stack

| Layer          | Technology                                             |
| -------------- | ------------------------------------------------------ |
| Language       | Python 3.11+, fully typed (mypy strict)                |
| API            | FastAPI + Uvicorn (SSE streaming)                      |
| LLM runtime    | Ollama (default) · vLLM (GPU, speculative decoding)    |
| Embeddings     | BAAI/bge-m3 · Ollama-embedded models                   |
| Reranker       | BAAI/bge-reranker-v2-m3 (CrossEncoder)                 |
| Vector store   | Qdrant (embedded mode or server)                       |
| Knowledge graph| Kuzu (embedded) · Neo4j (optional)                     |
| Sparse index   | Tantivy                                                |
| Parsing        | Docling, pypdf, faster-whisper, pytesseract            |
| Queue          | Redis + RQ (ingestion jobs)                            |
| CLI            | Typer + Rich                                           |
| Config         | Pydantic Settings + YAML                               |
| Observability  | OpenTelemetry · Langfuse                               |
| Eval           | RAGAS + custom harness                                 |
| Packaging      | Hatchling (pyproject.toml) · pre-commit · ruff         |

## 📊 Benchmarks

Measured with the built-in harness (`nexusmind eval run`) on internal golden sets (multi-hop QA, 2,400 questions). Reproduce, don't trust — numbers vary by model, corpus, and hardware.

| Metric                | Naive RAG baseline | NexusMind (hybrid + graph) |
| --------------------- | ------------------ | -------------------------- |
| Faithfulness (RAGAS)  | 0.81               | 0.94                       |
| Answer relevancy      | 0.83               | 0.91                       |
| Context precision     | 0.72               | 0.89                       |
| Context recall        | 0.78               | 0.92                       |
| p50 / p95 latency     | 0.9s / 2.1s        | 1.2s / 2.8s                |
| Multi-hop accuracy    | 41%                | 76%                        |

Hardware: RTX 4090, `qwen2.5:14b-instruct` Q4_K_M via Ollama, 50k-chunk corpus.

## 🚀 Quickstart

**Prerequisites:** Python 3.11+, Ollama, Docker (optional). 8 GB RAM minimum, NVIDIA GPU recommended.

One command (Docker):

```bash
docker compose up -d          # boots api + ui + worker + qdrant + redis
```

Open <http://localhost:8501> (UI) · <http://localhost:8000/docs> (API docs)

From source:

```bash
git clone https://github.com/nexusmind-ai/nexusmind.git
cd nexusmind
cp .env.example .env
make setup                    # venv + deps + pre-commit + ollama pull
make dev                      # api :8000 + ui :8501 + worker
```

## 📦 Installation

```bash
pip install nexusmind              # CPU
pip install "nexusmind[gpu]"       # CUDA acceleration (reranker + embeddings)
```

From source (uv):

```bash
git clone https://github.com/nexusmind-ai/nexusmind.git
cd nexusmind
uv sync --all-extras
pre-commit install
```

Pull local models:

```bash
ollama pull qwen2.5:14b-instruct   # LLM (or llama3.1:8b for 8 GB machines)
ollama pull bge-m3                 # embeddings
```

## ⚙️ Configuration

NexusMind is configured via `.env` (secrets) + `config.yaml` (behavior). See [`config.example.yaml`](config.example.yaml) and [`.env.example`](.env.example).

| Variable                     | Default                              | Description                        |
| ---------------------------- | ------------------------------------ | ---------------------------------- |
| `NEXUSMIND_LLM_BASE_URL`     | `http://localhost:11434/v1`          | OpenAI-compatible LLM endpoint     |
| `NEXUSMIND_LLM_MODEL`        | `qwen2.5:14b-instruct`               | Generation model                   |
| `NEXUSMIND_EMBED_MODEL`      | `bge-m3`                             | Embedding model                    |
| `NEXUSMIND_RERANKER_MODEL`   | `BAAI/bge-reranker-v2-m3`            | Cross-encoder reranker             |
| `NEXUSMIND_QDRANT_URL`       | `http://localhost:6333`              | Qdrant endpoint (embedded if unset)|
| `NEXUSMIND_KUZU_PATH`        | `./data/graph`                       | Kuzu graph database path           |
| `NEXUSMIND_REDIS_URL`        | `redis://localhost:6379/0`           | Job queue                          |
| `NEXUSMIND_DATA_DIR`         | `./data`                             | All local state                    |
| `NEXUSMIND_API_KEY`          | —                                    | Required in production             |
| `NEXUSMIND_MAX_CONTEXT_TOKENS`| `32768`                             | Context budget                     |
| `NEXUSMIND_TEMPERATURE`      | `0.2`                                | Sampling temperature               |
| `NEXUSMIND_LOG_LEVEL`        | `INFO`                               | `DEBUG\|INFO\|WARNING\|ERROR`      |
| `NEXUSMIND_LANGFUSE_KEY`     | —                                    | Optional tracing                   |

```yaml
# config.yaml
retriever:
  top_k: 12
  rerank: true
  graph_expansion: 1        # hops
  rrf_k: 60
agents:
  max_iterations: 8
  budget_tokens: 60000
ingestion:
  chunk_size: 512
  chunk_overlap: 64
  semantic_chunking: true
```

## 📖 Usage

### CLI

```bash
nexusmind ingest ./knowledge-base --recursive    # parse → chunk → embed → graph
nexusmind chat                                   # interactive TUI (Rich)
nexusmind ask "Summarize the Q3 migration plan." --citations
nexusmind serve --host 0.0.0.0 --port 8000       # API + UI
nexusmind eval run --suite rag --dataset ./evals/golden.jsonl
nexusmind mcp serve                              # MCP server over stdio
```

### Python SDK

```python
from nexusmind import NexusMind, RetrieverConfig

app = NexusMind.from_config("config.yaml")
app.ingest("./knowledge-base")

answer = app.ask(
    "Which API changes in the migration plan break contract Y?",
    retriever=RetrieverConfig(top_k=12, rerank=True, graph_expansion=2),
)
print(answer.text)
for c in answer.citations:
    print(f"  [{c.score:.2f}] {c.source}:{c.page}")
```

### REST API (streaming)

```bash
curl -N -X POST http://localhost:8000/v1/ask \
  -H "Authorization: Bearer $NEXUSMIND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What changed between v2 and v3 of the API?", "stream": true}'
```

## 🤖 The Agentic Core

| Agent      | Role                                                                 | Model budget |
| ---------- | -------------------------------------------------------------------- | ------------ |
| 🧭 Planner | Decomposes complex queries into sub-questions                        | 4k tokens    |
| 🔍 Researcher | ReAct loop: retrieve → read → call tools → iterate                 | 24k tokens   |
| ⚖️ Critic  | Verifies every claim is grounded in retrieved context; triggers re-retrieval | 8k tokens    |
| ✍️ Writer  | Synthesizes the final streamed answer with citations                 | 8k tokens    |

Register custom tools with a decorator — schema generation and validation are automatic:

```python
from nexusmind.tools import tool

@tool
def get_ticket_status(ticket_id: str) -> dict:
    """Fetch the current status of a Jira ticket."""
    ...  # your implementation

app.register_tool(get_ticket_status)
```

## 🔌 MCP Integration

NexusMind speaks the Model Context Protocol both ways:

Expose your knowledge base as an MCP server:

```bash
nexusmind mcp serve   # stdio; use --http for streamable HTTP
```

Use it from Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "nexusmind": {
      "command": "nexusmind",
      "args": ["mcp", "serve"]
    }
  }
}
```

Consume external MCP tools as agent tools:

```bash
nexusmind tools connect --server http://localhost:3000/mcp
```

## 🖼️ Multimodal Ingestion

| Format     | Parser              | Notes                          |
| ---------- | ------------------- | ------------------------------ |
| PDF        | Docling             | Layout-aware, tables, reading order |
| DOCX / PPTX| python-docx / python-pptx | Structures preserved       |
| Images     | pytesseract + vision model | OCR + captioning           |
| Audio      | faster-whisper      | Timestamped transcripts        |
| HTML / URL | trafilatura         | Boilerplate removal            |
| Code       | tree-sitter         | Language-aware chunking        |
| Markdown   | native              | Heading-aware splitting        |

## 📊 Evaluation Harness

```bash
nexusmind eval run --suite rag --dataset ./evals/golden.jsonl --output report.html
```

Reports faithfulness, answer relevancy, context precision, context recall, and latency percentiles. CI fails if faithfulness regresses below your threshold — retrieval changes are treated like production code.

## 🛰️ Observability

- OpenTelemetry spans for every pipeline stage (parse → retrieve → rerank → agent steps)
- Langfuse dashboards for token usage, cost, and agent traces (opt-in)
- Structured JSON logging via structlog

## 📁 Project Structure

```
nexusmind/
├── .github/                  # CI workflows, issue templates, FUNDING
├── docs/                     # MkDocs site + logo/GIF assets
├── examples/                 # Runnable example scripts
├── scripts/                  # bootstrap.sh, release.sh
├── src/nexusmind/            # Source package
│   ├── api/                  # FastAPI app, routes, SSE, schemas
│   ├── agents/               # planner · researcher · critic · writer · runtime
│   ├── retrieval/            # hybrid retriever, RRF, reranker, HyDE
│   ├── graph/                # Kuzu store, entity extraction, communities
│   ├── ingestion/            # parsers, chunkers, enrichers, queue jobs
│   ├── tools/                # @tool decorator, builtins, MCP bridge
│   ├── memory/               # conversation + episodic memory
│   ├── eval/                 # RAGAS harness, golden datasets
│   ├── observability/        # OTel, Langfuse hooks
│   ├── cli.py                # Typer CLI
│   ├── engine.py             # NexusMind facade
│   └── config.py             # Pydantic Settings
├── tests/                    # unit + integration + eval tests
├── .editorconfig
├── .env.example
├── .gitignore
├── .dockerignore
├── .pre-commit-config
├── .codecov.yml
├── .markdownlint.json
├── .releaserc.json
├── .readthedocs.yaml
├── .python-version
├── AUTHORS.md
├── ARCHITECTURE.md
├── CHANGELOG.md
├── CITATION.cff
├── CODESTYLE.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile
├── Dockerfile.gpu
├── FAQ.md
├── LICENSE
├── Makefile
├── MANIFEST.in
├── NOTICE
├── README.md
├── ROADMAP.md
├── SECURITY.md
├── SUPPORT.md
├── VERSION
├── config.example.yaml
├── docker-compose.yml
├── docker-compose.gpu.yml
├── docker-compose.prod.yml
├── mkdocs.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── requirements-gpu.txt
```

## 🌐 REST API

| Method  | Endpoint                | Description                            |
| ------- | ----------------------- | -------------------------------------- |
| POST    | `/v1/ask`               | Ask a question (SSE stream or JSON)    |
| POST    | `/v1/ingest`            | Queue an ingestion job (file/URL/folder) |
| GET     | `/v1/collections`       | List collections                       |
| GET     | `/v1/documents/{id}`    | Document metadata + chunk map          |
| DELETE  | `/v1/documents/{id}`    | Remove document + graph entities       |
| POST    | `/v1/tools`             | Register a tool                        |
| GET     | `/v1/health`            | Liveness + model status                |
| GET     | `/mcp`                  | MCP streamable-HTTP endpoint           |

## 🗺️ Roadmap

- [x] Hybrid retrieval + RRF fusion
- [x] GraphRAG v1 (entities, relations, 1-hop expansion)
- [x] Multi-agent runtime + MCP server
- [ ] Leiden community summaries (GraphRAG v2)
- [ ] Late chunking + ColBERT late-interaction index
- [ ] Speculative decoding profiles (vLLM)
- [ ] Web UI v2 (Next.js) — ROADMAP.md
- [ ] Distributed ingestion workers (Ray)
- [ ] Voice mode (faster-whisper + XTTS)
- [ ] LoRA fine-tuning loop on your own corpus

## 🤝 Contributing

We love contributions! Read [CONTRIBUTING.md](CONTRIBUTING.md), pick an issue labeled `good first issue`, and run:

```bash
make setup && make test && make lint
```

All PRs require: green CI, type-check pass, and updated golden-dataset evals if retrieval logic changed. Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🔒 Security

Local-first by design — no telemetry, no data egress. Report vulnerabilities responsibly via [SECURITY.md](SECURITY.md) or security@nexusmind.dev.

## 🙏 Acknowledgments

Built on the shoulders of giants: [Ollama](https://ollama.com) · [Qdrant](https://qdrant.tech) · [Kuzu](https://kuzudb.com) · [Docling](https://docling-project.github.io) · [Tantivy](https://github.com/quickwit-oss/tantivy) · [FastAPI](https://fastapi.tiangolo.com) · [RAGAS](https://docs.ragas.io) · [MCP](https://modelcontextprotocol.io) · the Hugging Face ecosystem.

## 📚 Citation

```bibtex
@software{nexusmind2025,
  author  = {NexusMind Contributors},
  title   = {NexusMind: A Local-First Agentic RAG Engine},
  year    = {2025},
  url     = {https://github.com/nexusmind-ai/nexusmind},
  version = {0.3.0}
}
```

## 📄 License

MIT — see [LICENSE](LICENSE). Third-party model licenses apply to downloaded weights (see [NOTICE](NOTICE)).

---

⭐ Star us on GitHub — it helps more people find privacy-first AI!

Made with 🧠 and zero cloud APIs by the NexusMind community.