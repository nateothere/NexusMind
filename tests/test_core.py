from __future__ import annotations

import numpy as np
import pytest

from nexusmind.config import IngestionConfig, RetrieverConfig, Settings
from nexusmind.graph.store import GraphStore
from nexusmind.ingestion.pipeline import IngestionPipeline, parse_markdown, split_recursive, split_semantic
from nexusmind.retrieval.hybrid import BM25Index, CrossEncoderReranker, DenseIndex, HybridRetriever, rrf_fuse
from nexusmind.utils import new_id, token_count, tokenize


def test_config_defaults() -> None:
    settings = Settings()
    assert settings.llm_model == "qwen2.5:14b-instruct"
    assert RetrieverConfig().top_k == 12
    assert IngestionConfig().chunk_overlap == 64


def test_config_rejects_bad_temperature() -> None:
    with pytest.raises(ValueError):
        Settings(temperature=2.0)


def _bm25() -> BM25Index:
    bm25 = BM25Index()
    bm25.add("d1", "The quick brown fox jumps over the lazy dog")
    bm25.add("d2", "Graph databases store entities and relations between entities")
    bm25.add("d3", "The lazy cat sleeps near the quick brown dog")
    return bm25


def test_bm25_ranks_relevant_first() -> None:
    assert _bm25().search("graph entities relations", top_k=2)[0][0] == "d2"


def test_bm25_edge_cases() -> None:
    assert _bm25().search("zyxwvutsrqponmlkj") == []
    assert BM25Index().search("anything") == []
    assert len(BM25Index()) == 0
    bm25 = BM25Index()
    bm25.add_many({"a": "alpha beta gamma", "b": "delta epsilon"})
    assert len(bm25) == 2
    assert bm25.text("a") == "alpha beta gamma"


def test_bm25_idf_prefers_rare_terms() -> None:
    bm25 = _bm25()
    assert bm25.search("entities", top_k=1)[0][0] == "d2"
    assert bm25.search("GRAPH", top_k=1)[0][0] == "d2"


def test_rrf_fuses_and_orders() -> None:
    fused = rrf_fuse([[("x", 0.9), ("y", 0.8)], [("z", 0.7), ("x", 0.6)]], k=60, top_k=10)
    assert [d for d, _ in fused][0] == "x"
    assert len(rrf_fuse([], top_k=5)) == 0
    assert len(rrf_fuse([[("a", 1.0), ("b", 0.5)]], top_k=1)) == 1


class _FakeEmbeddings:
    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> np.ndarray:
        return self._vectors.get(text, np.zeros(4, dtype=np.float32))


def _hybrid() -> HybridRetriever:
    bm25 = BM25Index()
    bm25.add("c1", "Qdrant stores dense vectors for semantic search")
    bm25.add("c2", "Kuzu stores the knowledge graph of entities")
    bm25.add("c3", "BM25 is a sparse retrieval method over tokens")
    dense = DenseIndex()
    dense.add("c1", "Qdrant stores dense vectors", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    dense.add("c2", "Kuzu graph", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    dense.add("c3", "BM25 sparse", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    embeddings = _FakeEmbeddings(
        {"dense vectors semantic": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    )
    return HybridRetriever(bm25, dense, embeddings=embeddings, rerank=False, top_k=3)


def test_hybrid_fuses_sparse_and_dense() -> None:
    assert _hybrid().retrieve("dense vectors semantic", top_k=3)[0][0] == "c1"
    empty = HybridRetriever(BM25Index(), DenseIndex())
    assert empty.retrieve("anything") == []
    assert _hybrid().chunk_text("c1").startswith("Qdrant")


def test_hybrid_rerank_passthrough() -> None:
    bm25 = BM25Index()
    bm25.add("a", "one two three")
    bm25.add("b", "four five six")
    hits = HybridRetriever(bm25, DenseIndex(), rerank=True, top_k=2).retrieve("one")
    assert hits[0][0] == "a"
    assert len(hits) == 1


def test_reranker_noop_fallback() -> None:
    reranker = CrossEncoderReranker("nonexistent-model")
    result = reranker.rerank("q", [("a", "text", 0.9), ("b", "text", 0.5)], top_k=2)
    assert result[0] == ("a", 0.9)
    assert result[1] == ("b", 0.5)
    assert reranker.rerank("q", []) == []


def test_graph_store() -> None:
    store = GraphStore(":memory:")
    first = store.upsert_entity("Qdrant", "database", "default")
    assert store.upsert_entity("Qdrant", "database", "default").id == first.id
    assert store.entity_count("default") == 1
    b = store.upsert_entity("Kuzu", "database", "default")
    store.add_relation(first.id, b.id, "pairs_with", "default")
    expanded = store.expand([first.id], hops=1)
    assert len(expanded) == 1
    assert expanded[0]["entity"]["name"] == "Kuzu"
    assert store.expand([], hops=1) == []
    assert store.expand([first.id], hops=0) == []
    store.remove_collection("default")
    assert store.entity_count("default") == 0
    store.close()


def test_graph_two_hop_expansion() -> None:
    store = GraphStore(":memory:")
    a = store.upsert_entity("A", "x", "default")
    b = store.upsert_entity("B", "x", "default")
    c = store.upsert_entity("C", "x", "default")
    store.add_relation(a.id, b.id, "linked_to", "default")
    store.add_relation(b.id, c.id, "linked_to", "default")
    assert len(store.expand([a.id], hops=1)) == 1
    assert len(store.expand([a.id], hops=2)) == 2
    assert [e.name for e in store.list_entities("default")] == ["A", "B", "C"]
    store.close()


def test_split_recursive() -> None:
    assert split_recursive("Short document.") == ["Short document."]
    chunks = split_recursive("word " * 600, chunk_size=128, overlap=16)
    assert len(chunks) > 1
    chunks = split_recursive("word " * 300, chunk_size=100, overlap=20)
    assert len(chunks) == 4
    heading_text = "# Intro\nshort intro\n\n## Details\n" + "paragraph " * 120
    chunks = split_recursive(heading_text, chunk_size=64, overlap=8)
    assert len(chunks) >= 2
    assert any(c.startswith("# Intro") for c in chunks)


def test_split_semantic() -> None:
    assert split_semantic("short text here.", embed=None) == ["short text here."]

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "topic" in t else [0.0, 1.0] for t in texts]

    text = "topic one topic two topic three. other subject entirely. back to topic."
    assert len(split_semantic(text, embed=embed, chunk_size=64)) >= 2


def test_parse_and_pipeline(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("# Notes\n\nQdrant stores vectors. Kuzu stores graphs.\n", encoding="utf-8")
    doc = parse_markdown(source, collection="docs")
    assert doc.title == "Notes"
    pipeline = IngestionPipeline(BM25Index(), DenseIndex(), None, None, chunk_size=64, semantic_chunking=False)
    chunks = pipeline.ingest_path(tmp_path, collection="docs", recursive=True)
    assert len(chunks) >= 1
    assert len(pipeline.documents()) == 1


def test_utils() -> None:
    assert token_count("a b c") == 3
    assert tokenize("Hello WORLD") == ["hello", "world"]
    assert new_id("x").startswith("x_")