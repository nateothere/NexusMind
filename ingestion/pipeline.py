"""Ingestion - schemas, chunkers, parsers, enrichment, pipeline."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from ..graph.store import GraphStore
from ..llm.client import OllamaEmbeddings
from ..retrieval.hybrid import BM25Index, DenseIndex
from ..utils import IngestionError, new_id, token_count, tokenize, unique

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


class Chunk(BaseModel):
    id: str
    doc_id: str
    collection: str
    text: str
    page: int | None = None
    heading: str | None = None
    tokens: int = 0


class Document(BaseModel):
    id: str
    collection: str
    source: str
    title: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def split_recursive(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    if token_count(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    sections = _split_on_headings(text)
    chunks: list[str] = []
    buffer = ""
    for section in sections:
        if token_count(buffer + section) <= chunk_size:
            buffer += section
            continue
        if buffer:
            chunks.append(buffer.strip())
        buffer = section
    if buffer:
        chunks.append(buffer.strip())
    return _split_oversized(chunks, chunk_size, overlap)


def _split_on_headings(text: str) -> list[str]:
    parts = _HEADING_RE.split(text)
    sections: list[str] = []
    if parts and parts[0].strip():
        sections.append(parts[0])
    for i in range(1, len(parts) - 1, 2):
        sections.append(f"{parts[i]} {parts[i + 1]}")
    return sections


def _split_oversized(chunks: list[str], chunk_size: int, overlap: int) -> list[str]:
    out: list[str] = []
    for chunk in chunks:
        if token_count(chunk) <= chunk_size:
            out.append(chunk)
            continue
        words = chunk.split()
        step = max(1, chunk_size - overlap)
        start = 0
        while start < len(words):
            out.append(" ".join(words[start : start + chunk_size]))
            start += step
    return [c for c in out if c.strip()]


def split_semantic(text: str, embed, chunk_size: int = 512) -> list[str]:
    if embed is None:
        return split_recursive(text, chunk_size=chunk_size, overlap=64)
    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if len(sentences) < 2:
        return split_recursive(text, chunk_size=chunk_size, overlap=64)
    vectors = embed(sentences)
    breaks: list[int] = []
    for i in range(1, len(vectors)):
        if _sim(vectors[i - 1], vectors[i]) < 0.45:
            breaks.append(i)
    chunks: list[str] = []
    start = 0
    for end in breaks + [len(sentences)]:
        piece = " ".join(sentences[start:end])
        if piece.strip():
            chunks.append(piece)
        start = end
    return chunks


def _sim(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def parse_markdown(path: Path, collection: str = "default") -> Document:
    if not path.exists():
        raise IngestionError(f"file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        id=f"doc-{path.stem}",
        collection=collection,
        source=str(path),
        title=_title_from_path(path),
        content=text,
        metadata={"format": "markdown" if path.suffix.lower() in (".md", ".markdown") else "text"},
    )


def _title_from_path(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return path.stem.replace("_", " ").replace("-", " ").title()


class DoclingParser:
    def __init__(self) -> None:
        self._docling: object | None = None

    def _ensure(self) -> None:
        if self._docling is None:
            try:
                from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

                self._docling = DocumentConverter()
            except ImportError as exc:
                raise IngestionError(
                    "docling not installed; install the gpu extras or set ingestion.parser=native"
                ) from exc

    def parse(self, path: Path, collection: str = "default") -> Document:
        self._ensure()
        assert self._docling is not None
        result = self._docling.convert(str(path))  # type: ignore[attr-defined]
        text = result.document.export_to_markdown()  # type: ignore[attr-defined]
        return Document(
            id=f"doc-{path.stem}",
            collection=collection,
            source=str(path),
            title=_title_from_path(path),
            content=text,
            metadata={"format": path.suffix.lower().lstrip(".")},
        )


def extract_keywords(text: str, top: int = 8) -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "that", "this", "are", "was", "were",
        "from", "have", "has", "had", "not", "but", "you", "your", "will",
        "can", "all", "its", "about", "into", "than", "them", "then",
    }
    counts = Counter(t for t in tokenize(text) if t not in stopwords and len(t) > 2)
    return [word for word, _count in counts.most_common(top)]


def extract_entities(text: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"\b([A-Z][a-zA-Z0-9_/-]{2,}(?:\s+[A-Z][a-zA-Z0-9_/-]+)*)\b", text):
        found.append(match.group(1))
    return unique(found)[:12]


def summarize_chunk(text: str, max_chars: int = 240) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= max_chars:
        return flat
    cut = flat[:max_chars]
    last = cut.rfind(". ")
    return cut[: last + 1] if last > max_chars // 2 else cut + "…"


def enrich(text: str) -> dict[str, object]:
    return {
        "keywords": extract_keywords(text),
        "entities": extract_entities(text),
        "summary": summarize_chunk(text),
    }


class IngestionPipeline:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        embeddings: OllamaEmbeddings | None,
        graph: GraphStore | None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        semantic_chunking: bool = True,
        parser: str = "native",
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._embeddings = embeddings
        self._graph = graph
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._semantic_chunking = semantic_chunking
        self._parser = parser
        self._documents: dict[str, Document] = {}
        self._chunks_by_doc: dict[str, list[Chunk]] = {}

    def ingest_path(self, path: str | Path, collection: str = "default", recursive: bool = False) -> list[Chunk]:
        p = Path(path)
        if p.is_dir():
            targets = sorted(p.rglob("*") if recursive else p.glob("*"))
            chunks: list[Chunk] = []
            for target in targets:
                if target.is_file() and target.suffix.lower() in (".md", ".markdown", ".txt"):
                    chunks.extend(self.ingest_file(target, collection))
            return chunks
        return self.ingest_file(p, collection)

    def ingest_file(self, path: str | Path, collection: str = "default") -> list[Chunk]:
        p = Path(path)
        if not p.exists():
            raise IngestionError(f"file not found: {p}")
        if self._parser == "docling":
            document = DoclingParser().parse(p, collection)
        else:
            document = parse_markdown(p, collection)
        return self.ingest_document(document, collection)

    def ingest_document(self, document: Document, collection: str = "default") -> list[Chunk]:
        if self._semantic_chunking and self._embeddings is not None:
            texts = split_semantic(
                document.content,
                embed=lambda texts: [self._embeddings.embed(t).tolist() for t in texts],
                chunk_size=self._chunk_size,
            )
        else:
            texts = split_recursive(document.content, self._chunk_size, self._chunk_overlap)
        chunks: list[Chunk] = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                id=new_id("chunk"),
                doc_id=document.id,
                collection=collection,
                text=text,
                page=i,
                tokens=len(text.split()),
            )
            self._bm25.add(chunk.id, text)
            if self._embeddings is not None:
                self._dense.add(chunk.id, text, self._embeddings.embed(text))
            chunks.append(chunk)
        self._documents[document.id] = document
        self._chunks_by_doc[document.id] = chunks
        return chunks

    def documents(self) -> list[Document]:
        return list(self._documents.values())