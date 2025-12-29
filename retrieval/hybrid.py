"""Retrieval stack - BM25, dense, RRF, reranker, hybrid."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from ..llm.client import OllamaEmbeddings, cosine
from ..utils import RetrievalError, tokenize

_K1 = 1.2
_B = 0.75


class BM25Index:
    def __init__(self) -> None:
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._doc_texts: dict[str, str] = {}
        self._avgdl = 0.0
        self._total_docs = 0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        self._doc_texts[doc_id] = text
        self._doc_lengths[doc_id] = len(tokens)
        for term, count in Counter(tokens).items():
            self._postings.setdefault(term, {})[doc_id] = count
        self._total_docs = len(self._doc_lengths)
        self._avgdl = sum(self._doc_lengths.values()) / self._total_docs if self._total_docs else 0.0

    def add_many(self, docs: dict[str, str]) -> None:
        for doc_id, text in docs.items():
            self.add(doc_id, text)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._total_docs == 0:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        scores: dict[str, float] = {}
        for term in set(terms):
            postings = self._postings.get(term, {})
            df = len(postings)
            idf = math.log(1 + (self._total_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                dl = self._doc_lengths[doc_id]
                denom = tf + _K1 * (1 - _B + _B * dl / self._avgdl) if self._avgdl else tf + _K1
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (_K1 + 1)) / denom
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

    def text(self, doc_id: str) -> str:
        return self._doc_texts[doc_id]

    def __len__(self) -> int:
        return self._total_docs


class DenseIndex:
    def __init__(self, embeddings: OllamaEmbeddings | None = None) -> None:
        self._embeddings = embeddings
        self._vectors: dict[str, np.ndarray] = {}
        self._texts: dict[str, str] = {}

    def add(self, doc_id: str, text: str, vector: np.ndarray) -> None:
        self._vectors[doc_id] = vector
        self._texts[doc_id] = text

    def add_text(self, doc_id: str, text: str) -> None:
        if self._embeddings is None:
            raise RuntimeError("dense index has no embeddings client")
        self.add(doc_id, text, self._embeddings.embed(text))

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        scored = [(doc_id, cosine(query_vector, vec)) for doc_id, vec in self._vectors.items()]
        return sorted(scored, key=lambda kv: (-kv[1], kv[0]))[:top_k]

    def search_text(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._embeddings is None:
            raise RuntimeError("dense index has no embeddings client")
        return self.search(self._embeddings.embed(query), top_k=top_k)

    def text(self, doc_id: str) -> str:
        return self._texts[doc_id]

    def __len__(self) -> int:
        return len(self._vectors)


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60, top_k: int = 12) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]


def normalized_scores(ranking: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not ranking:
        return []
    best = max(score for _, score in ranking)
    if best <= 0.0:
        return [(doc_id, 0.0) for doc_id, _ in ranking]
    return [(doc_id, score / best) for doc_id, score in ranking]


def merge_unique(rankings: list[list[tuple[str, float]]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ranking in rankings:
        for doc_id, _score in ranking:
            if doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)
    return out


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._available = False

    def _load(self) -> bool:
        if self._available:
            return True
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

            self._model = CrossEncoder(self._model_name)
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def rerank(self, query: str, candidates: list[tuple[str, str, float]], top_k: int = 10) -> list[tuple[str, float]]:
        if not self._load():
            return [(doc_id, score) for doc_id, _text, score in candidates[:top_k]]
        assert self._model is not None
        scores = self._model.predict([[query, text] for _doc_id, text, _score in candidates])  # type: ignore[no-untyped-call]
        ordered = sorted(zip(candidates, scores), key=lambda kv: -float(kv[1]))
        return [(doc_id, float(score)) for (doc_id, _text, _score), score in ordered[:top_k]]


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        embeddings: OllamaEmbeddings | None = None,
        reranker: CrossEncoderReranker | None = None,
        top_k: int = 12,
        rrf_k: int = 60,
        rerank: bool = True,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._embeddings = embeddings
        self._reranker = reranker or CrossEncoderReranker()
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._rerank = rerank

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        k = top_k or self._top_k
        bm25_hits = self._bm25.search(query, top_k=k * 3)
        dense_hits: list[tuple[str, float]] = []
        if self._embeddings is not None and len(self._dense) > 0:
            dense_hits = self._dense.search(self._embeddings.embed(query), top_k=k * 3)
        if not bm25_hits and not dense_hits:
            return []
        fused = rrf_fuse([bm25_hits, dense_hits], k=self._rrf_k, top_k=k * 2)
        if self._rerank and fused:
            candidates = [
                (doc_id, self._bm25.text(doc_id), score) for doc_id, score in fused if doc_id in self._bm25._doc_texts  # noqa: SLF001
            ]
            if not candidates:
                return fused
            return self._reranker.rerank(query, candidates, top_k=k)
        return fused[:k]

    def chunk_text(self, chunk_id: str) -> str:
        try:
            return self._bm25.text(chunk_id)
        except KeyError as exc:
            raise RetrievalError(f"unknown chunk {chunk_id}") from exc