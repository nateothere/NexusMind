"""NexusMind facade - wires config, retrieval, graph, agents, memory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .agents.runtime import Critic, Planner, Researcher, Writer
from .config import (
    AgentsConfig,
    IngestionConfig,
    MemoryConfig,
    RetrieverConfig,
    Settings,
    config_from_yaml,
)
from .graph.store import GraphExpander, GraphStore
from .ingestion.pipeline import Document, IngestionPipeline
from .llm.client import LLMClient, OllamaEmbeddings
from .memory.buffer import ConversationBuffer, EpisodicMemory
from .retrieval.hybrid import BM25Index, CrossEncoderReranker, DenseIndex, HybridRetriever
from .tools.registry import Tool, ToolRegistry, register_builtins
from .utils import NexusMindError, now_ms


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    collection: str
    text: str
    score: float
    rank: int
    index: str


class Citation(BaseModel):
    index: int
    source: str
    page: int | None = None
    score: float = 0.0


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[dict[str, object]] = Field(default_factory=list)
    cost_tokens: int = 0
    latency_ms: int = 0


class NexusMind:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: RetrieverConfig | None = None,
        agents: AgentsConfig | None = None,
        ingestion: IngestionConfig | None = None,
        memory: MemoryConfig | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.retriever_cfg = retriever or RetrieverConfig()
        self.agents_cfg = agents or AgentsConfig()
        self.ingestion_cfg = ingestion or IngestionConfig()
        self.memory_cfg = memory or MemoryConfig()
        self._llm = LLMClient(self.settings)
        self._embeddings = OllamaEmbeddings(self.settings)
        self._bm25 = BM25Index()
        self._dense = DenseIndex(self._embeddings)
        self._reranker = CrossEncoderReranker(self.settings.reranker_model)
        self._retriever = HybridRetriever(
            self._bm25,
            self._dense,
            embeddings=self._embeddings,
            reranker=self._reranker,
            top_k=self.retriever_cfg.top_k,
            rrf_k=self.retriever_cfg.rrf_k,
            rerank=self.retriever_cfg.rerank,
        )
        self._graph = GraphStore(self.settings.kuzu_path)
        self._expander = GraphExpander(self._graph)
        self._pipeline = IngestionPipeline(
            self._bm25,
            self._dense,
            self._embeddings,
            self._graph,
            chunk_size=self.ingestion_cfg.chunk_size,
            chunk_overlap=self.ingestion_cfg.chunk_overlap,
            semantic_chunking=self.ingestion_cfg.semantic_chunking,
            parser=self.ingestion_cfg.parser,
        )
        self._tools = ToolRegistry()
        register_builtins(self._tools, retriever=self._retriever)
        self._buffer = ConversationBuffer(self.memory_cfg.buffer_turns)
        self._episodic = EpisodicMemory(self._dense)
        self._collections: set[str] = set()

    @classmethod
    def from_config(cls, path: str | Path = "config.yaml") -> NexusMind:
        merged = config_from_yaml(path)
        return cls(
            settings=merged.get("settings"),
            retriever=merged.get("retriever"),
            agents=merged.get("agents"),
            ingestion=merged.get("ingestion"),
            memory=merged.get("memory"),
        )

    async def close(self) -> None:
        await self._llm.close()
        self._embeddings.close()
        self._graph.close()

    def ingest(self, source: str, recursive: bool = False, collection: str = "default") -> list[str]:
        chunks = self._pipeline.ingest_path(source, collection=collection, recursive=recursive)
        if chunks:
            self._collections.add(collection)
        return [chunk.id for chunk in chunks]

    def collections(self) -> list[str]:
        return sorted(self._collections)

    def document(self, doc_id: str) -> dict[str, object] | None:
        for document in self._pipeline.documents():
            if document.id == doc_id:
                return document.model_dump()
        return None

    def delete_document(self, doc_id: str) -> None:
        found = self.document(doc_id)
        if found is None:
            raise NexusMindError(f"document not found: {doc_id}")
        self._graph.remove_collection(str(found.get("collection", "default")))
        self._collections.discard(str(found.get("collection", "default")))

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        graph_expansion: int | None = None,
        collection: str = "default",
    ) -> list[RetrievedChunk]:
        hits = self._retriever.retrieve(query, top_k=top_k)
        hops = graph_expansion if graph_expansion is not None else self.retriever_cfg.graph_expansion
        extra = ""
        if hops > 0:
            extra = self._expander.context_block(query, collection=collection, hops=hops)
        out: list[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(hits):
            text = self._retriever.chunk_text(chunk_id)
            if extra and rank == 0:
                text = f"{extra}\n\n{text}"
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id="",
                    collection=collection,
                    text=text,
                    score=score,
                    rank=rank,
                    index="hybrid",
                )
            )
        return out

    async def ask(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool | None = None,
        graph_expansion: int | None = None,
        collection: str = "default",
    ) -> Answer:
        start = now_ms()
        context_hits = self.retrieve(query, top_k=top_k, graph_expansion=graph_expansion, collection=collection)
        context = "\n\n".join(hit.text for hit in context_hits[: self.retriever_cfg.top_k])
        planner = Planner(self._llm, budget=4_000)
        sub_questions = await planner.plan(query)
        researcher = Researcher(self._llm, self._tools, budget=24_000)
        facts: list[str] = []
        for question in sub_questions[:2]:
            async for event in researcher.run(question, context):
                if event.get("event") == "done" and event.get("result"):
                    facts.append(str(event["result"]))
        critic = Critic(self._llm, budget=8_000)
        grounded, note = await critic.check("\n".join(facts), context)
        if not grounded:
            extra_hits = self.retrieve(note, top_k=3, graph_expansion=0, collection=collection)
            context = f"{context}\n\n{extra_hits[0].text if extra_hits else ''}"
        writer = Writer(self._llm, budget=8_000)
        text = await writer.write("\n".join(facts), query)
        citations = [
            Citation(index=i, source=f"chunk:{hit.chunk_id}", score=hit.score)
            for i, hit in enumerate(context_hits[:5])
        ]
        if self.memory_cfg.episodic:
            self._episodic.remember(query, text)
        self._buffer.add("user", query)
        self._buffer.add("assistant", text)
        return Answer(
            text=text,
            citations=citations,
            latency_ms=now_ms() - start,
            cost_tokens=sum(len(f.split()) for f in facts) + len(text.split()),
        )

    async def ask_stream(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool | None = None,
        graph_expansion: int | None = None,
        collection: str = "default",
    ) -> AsyncIterator[dict[str, object]]:
        answer = await self.ask(query, top_k=top_k, rerank=rerank, graph_expansion=graph_expansion, collection=collection)
        for token in answer.text.split():
            yield {"event": "token", "delta": token + " "}
        for i, citation in enumerate(answer.citations):
            yield {"event": "citation", "index": i, "score": citation.score, "source": citation.source}
        yield {"event": "done", "latency_ms": answer.latency_ms, "tokens": answer.cost_tokens}

    def register_tool(self, fn: Any) -> Tool:
        tool_obj: Tool = fn
        self._tools.register(tool_obj)
        return tool_obj

    def register_tool_by_schema(self, name: str, description: str, schema: dict[str, Any]) -> None:
        def _passthrough(**kwargs: Any) -> dict[str, Any]:
            return {"arguments": kwargs}

        self._tools.register(Tool(name, description, _passthrough, schema))

    def tool_schemas(self) -> list[dict[str, Any]]:
        return self._tools.schemas()

    def memory_stats(self) -> dict[str, int]:
        return {"buffer_turns": len(self._buffer), "episodes": self._episodic.count()}