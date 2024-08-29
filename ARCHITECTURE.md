# Architecture

NexusMind is a local-first Agentic RAG engine. This document describes the
pipeline internals, data contracts, and the design decisions behind them.

## 1. Pipeline overview

Two pipelines share one data model: **ingestion** (documents in, indexes
out) and **query** (question in, streamed cited answer out).

### 1.1 Ingestion DAG

```
Sources (PDF · DOCX · MD · URL · Audio · Code)
   │  parser (Docling native fallback: Markdown/Text)
   ▼
Document (id, collection, metadata, content)
   │  chunker (recursive → semantic → late)
   ▼
Chunk[] (id, doc_id, text, page, heading)
   │  enricher (entities · keywords · summaries)
   ▼
EnrichedChunk[]
   ├─► DenseIndex (Qdrant / numpy in-memory)      embeddings via Ollama bge-m3
   ├─► BM25Index (Tantivy / pure-Python fallback) tokenized field index
   └─► GraphStore (Kuzu / SQLite fallback)         entity+relation rows
```

Ingestion jobs run on the worker (Redis + RQ when available; an in-process
queue otherwise). Each stage emits an OpenTelemetry span.

### 1.2 Query flow

```
User Query
   │  query understanding (classify · rewrite · decompose)
   ▼
simple? ──yes──► HybridRetriever
   │                BM25 hits ⊕ dense hits
   │                ▼ RRF (formula below)
   │                ▼ cross-encoder rerank (optional)
   │                ▼ graph 1-hop expansion (optional)
   ▼ no
PlannerAgent → sub-questions ──► HybridRetriever per sub-question
   ▼
ResearcherAgent (ReAct loop, tools, token budget)
   ▼
CriticAgent (grounding check → re-retrieval if ungrounded)
   ▼
WriterAgent (streamed answer + citations)
```

### 1.3 RRF math

Reciprocal Rank Fusion combines ranked lists without score calibration:

```
RRF(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))        k = 60 (default)
```

`k` is exposed as `retriever.rrf_k`. Scores are normalized per index
before fusion so dense similarity and BM25 weights are comparable.

## 2. Data contracts

All inter-stage data crosses typed Pydantic models (see `api/schemas.py`
and `src/nexusmind/*`):

| Model            | Fields                                                        |
| ---------------- | ------------------------------------------------------------- |
| `Document`       | id, collection, source, title, content, metadata, ingested_at |
| `Chunk`          | id, doc_id, text, page, heading, tokens                       |
| `Entity`         | id, name, kind, collection, attributes                        |
| `Relation`       | id, src, dst, predicate, weight                               |
| `RetrievedChunk` | chunk, score, rank, index (`bm25`/`dense`/`rrf`)              |
| `Answer`         | text, citations[], trace[], cost, latency_ms                  |

## 3. Agent state machine

Each agent is an async generator of `AgentEvent`s:

```
RUNNING → (think | tool_call | tool_result | retrieval)* → DONE
                                    │
                                    └─ over budget → HALT (tokens refunded)
```

Budgets are enforced by `AgentRuntime`: every LLM call and tool result is
metered against the agent's `budget_tokens`; `HALT` events carry a partial
answer so the Writer can still produce output.

## 4. Graph store

`GraphStore` exposes an embedded, filesystem-backed graph with two
backends behind one interface:

- **Kuzu** (default when installed) — native Cypher-like queries
- **SQLite adjacency fallback** — same CRUD + `expand()` semantics,
  zero native dependencies (used in CI)

Entity extraction runs in batches with the LLM; relations are stored
with predicates normalized to snake_case.

## 5. Design decisions

- **Local-first, fallback-first.** Every heavy dependency (Kuzu, Qdrant,
  Tantivy, Docling, sentence-transformers) is optional at import time.
  The defaults are pure-Python and testable offline. You get the native
  backend when you install the extras; you never get a broken core.
- **Streams everywhere.** `/v1/ask` and the Writer both stream tokens; the
  CLI renders them incrementally.
- **Structured outputs are constraints, not prompts.** Agent steps use
  JSON-schema decoding (`response_format`) so downstream parsing cannot
  fail silently.
- **Eval is a gate.** Retrieval changes must pass the golden dataset
  before merge (see CONTRIBUTING.md).

## 6. Observability

Every stage emits OTel spans (`parse`, `chunk`, `embed`, `index`,
`retrieve`, `rerank`, `graph_expand`, `agent.step`). With
`NEXUSMIND_LANGFUSE_KEY` set, spans are exported to Langfuse for token /
cost / latency dashboards. Structured logs go through structlog as JSON
lines.