# FAQ

## Models & hardware

**How much GPU memory do I need?**
`qwen2.5:14b-instruct` Q4_K_M uses ~9-10 GB VRAM. On 8 GB cards use
`llama3.1:8b` (~6 GB). CPU-only works with `qwen2.5:7b` but expect
1-3 s/token. Embeddings (`bge-m3`) and the reranker run fine on CPU.

**Ollama or vLLM?**
Ollama is the default: one binary, no container needed, good quantized
quality. vLLM wins for throughput and speculative decoding on big GPUs —
set `NEXUSMIND_LLM_BASE_URL` to its OpenAI-compatible endpoint.

**Which embedding model is used?**
`bge-m3` by default (multilingual, 8192 context, dense + sparse).
Any Ollama-embedded model works — set `NEXUSMIND_EMBED_MODEL`.

## Retrieval

**Why Kuzu instead of Neo4j?**
Kuzu is embedded (no server), fast for read-heavy graph walks, and
installs from a single wheel. Neo4j remains an optional backend for
teams that already run it. See ARCHITECTURE.md.

**How do I tune chunking?**
Start with `chunk_size: 512, chunk_overlap: 64`. Smaller chunks (256)
help exact-answer questions; larger (768-1024) help summarization.
Enable `semantic_chunking` when your documents have topic shifts that
recursive splitting ignores.

**What does `graph_expansion` do?**
After hybrid retrieval, top chunks' entities are linked in the graph and
their 1-hop neighbors are pulled in as additional context. 1 hop is a
good default; 2 hops raise recall but dilute precision.

**Why is my RRF fusion skewed?**
You likely have `rrf_k` far from 60, or you're mixing score scales. The
hybrid retriever normalizes per index before fusion — if you bypass it
and fuse manually, normalize first.

## Errors & fixes

**`connection refused` on 11434**
Ollama isn't running or is on another port. Check `ollama serve` and
`NEXUSMIND_LLM_BASE_URL`.

**`model not found: bge-m3`**
Run `ollama pull bge-m3` and `ollama pull qwen2.5:14b-instruct`.

**Ingestion hangs on PDFs**
Docling is optional and heavy. If you only need Markdown/Text, set
`ingestion.parser: native`. For PDFs, install `nexusmind[gpu]` extras or
`pip install docling`.

**`NEXUSMIND_API_KEY` required in production**
Set it in `.env`. The API refuses non-loopback requests without it —
that is by design (SECURITY.md).

**Windows: `pytesseract` can't find tesseract**
Install Tesseract and add its install dir to `PATH`, or skip image
ingestion for now.

## Project

**Does NexusMind send my data anywhere?**
No. There is no telemetry, no analytics, no cloud API calls. All model
traffic goes to your local Ollama/vLLM endpoint. Network calls happen
only when you explicitly ingest a URL.

**Can I contribute with a small machine?**
Yes — the unit suite is offline and runs in seconds. The integration
suite is optional and clearly marked.

**License?**
MIT for the codebase; downloaded model weights carry their own licenses
(NOTICE).