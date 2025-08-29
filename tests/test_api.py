from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nexusmind.api.app import create_app
from nexusmind.engine import NexusMind
from nexusmind.eval.harness import context_precision, context_recall, faithfulness_simple, latency_percentiles
from nexusmind.retrieval.hybrid import BM25Index, DenseIndex, HybridRetriever


class _StubLLM:
    async def complete(self, messages, temperature=None, json_schema=None) -> str:  # type: ignore[no-untyped-def]
        import json

        return json.dumps({"type": "finish", "result": "a grounded answer"})

    async def stream(self, messages):  # type: ignore[no-untyped-def]
        yield "stub"

    async def close(self) -> None:
        pass

    @staticmethod
    def system(content: str) -> object:
        return {"role": "system", "content": content}

    @staticmethod
    def user(content: str) -> object:
        return {"role": "user", "content": content}


def _engine() -> NexusMind:
    engine = NexusMind()
    engine._bm25 = BM25Index()  # noqa: SLF001
    engine._bm25.add("c1", "Qdrant stores dense vectors for semantic search")  # noqa: SLF001
    engine._bm25.add("c2", "Kuzu stores the knowledge graph of entities")  # noqa: SLF001
    engine._retriever = HybridRetriever(engine._bm25, DenseIndex(), rerank=False, top_k=2)  # noqa: SLF001
    engine._collections.add("default")  # noqa: SLF001
    engine._pipeline._embeddings = None  # noqa: SLF001
    engine._dense._embeddings = None  # noqa: SLF001
    engine._llm = _StubLLM()  # type: ignore[assignment]
    return engine


def test_health_endpoint() -> None:
    client = TestClient(create_app(_engine()))
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["collections"] == ["default"]


def test_ask_json_and_stream() -> None:
    client = TestClient(create_app(_engine()))
    json_resp = client.post("/v1/ask", json={"query": "What stores dense vectors?", "stream": False})
    assert json_resp.status_code == 200
    assert "text" in json_resp.json()
    stream_resp = client.post("/v1/ask", json={"query": "What stores dense vectors?", "stream": True})
    assert stream_resp.status_code == 200
    assert stream_resp.headers["content-type"].startswith("text/event-stream")
    assert "event:" in stream_resp.text


def test_ingest_documents_tools(tmp_path) -> None:
    source = tmp_path / "doc.md"
    source.write_text("# Doc\n\nSome content here.\n", encoding="utf-8")
    client = TestClient(create_app(_engine()))
    assert client.post("/v1/ingest", json={"source": str(source), "collection": "docs"}).status_code == 200
    assert client.get("/v1/collections").json()["collections"] == ["default", "docs"]
    assert client.get("/v1/documents/unknown").status_code == 404
    assert client.delete("/v1/documents/doc-doc").status_code == 200
    response = client.post(
        "/v1/tools",
        json={"name": "hello", "description": "say hi", "schema": {"type": "object"}},
    )
    assert response.status_code == 200


def test_api_key_enforced() -> None:
    engine = _engine()
    engine.settings.api_key = "secret"
    client = TestClient(create_app(engine))
    assert client.get("/v1/health").status_code == 401
    assert client.get("/v1/health", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_ask_empty_query_rejected() -> None:
    client = TestClient(create_app(_engine()))
    assert client.post("/v1/ask", json={"query": ""}).status_code == 422


def test_engine_retrieve_memory_stream() -> None:
    engine = _engine()
    hits = engine.retrieve("dense vectors", top_k=2)
    assert len(hits) >= 1
    assert hits[0].chunk_id == "c1"
    assert hits[0].score >= 0
    assert engine.memory_stats() == {"buffer_turns": 0, "episodes": 0}
    events: list[dict[str, object]] = []

    import asyncio

    async def collect() -> None:
        async for event in engine.ask_stream("What stores dense vectors?"):
            events.append(event)

    asyncio.run(collect())
    kinds = [e["event"] for e in events]
    assert "done" in kinds
    assert any(e["event"] == "token" for e in events)


def test_engine_tool_registration_and_config(tmp_path) -> None:
    engine = _engine()

    from nexusmind.tools.registry import tool

    @tool
    def ping(message: str) -> dict:
        """Echo a message."""
        return {"echo": message}

    engine.register_tool(ping)
    assert any(t["name"] == "ping" for t in engine.tool_schemas())
    engine.register_tool_by_schema("raw", "raw tool", {"type": "object"})
    assert "raw" in [t["name"] for t in engine.tool_schemas()]
    config = tmp_path / "config.yaml"
    config.write_text("retriever:\n  top_k: 5\n", encoding="utf-8")
    assert NexusMind.from_config(config).retriever_cfg.top_k == 5


def test_eval_percentiles() -> None:
    result = latency_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["p50"] == 3.0
    assert result["p95"] == 5.0
    assert context_precision(["a", "b", "c"], {"c"}) == pytest.approx(1 / 9)
    assert context_recall([], {"a"}) == 0.0
    assert faithfulness_simple("anything", "") == 0.0


def test_cli_importable() -> None:
    from nexusmind import cli

    assert cli.app.info.name == "nexusmind"