"""RAGAS-style metrics + evaluation harness."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.client import LLMClient
from ..retrieval.hybrid import HybridRetriever
from ..utils import tokenize

_GOLDEN_KEYS = {"question", "answer", "relevant_chunks"}
_FAITHFULNESS_PROMPT = """Rate faithfulness of the answer to the given
context on a scale of 0.0 to 1.0 (1.0 = every claim is supported).
Return only a JSON number."""


def context_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    hits = 0.0
    total = 0.0
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk in relevant:
            hits += 1
            total += hits / rank
    return total / len(retrieved)


def context_recall(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved)) / len(relevant)


def faithfulness_simple(answer: str, context: str) -> float:
    context_tokens = set(tokenize(context))
    if not context_tokens:
        return 0.0
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 1.0
    supported = sum(1 for tok in answer_tokens if tok in context_tokens)
    return supported / len(answer_tokens)


async def faithfulness_llm(llm: LLMClient, answer: str, context: str) -> float:
    response = await llm.complete(
        [llm.system(_FAITHFULNESS_PROMPT), llm.user(f"Context:\n{context}\n\nAnswer:\n{answer}")]
    )
    try:
        return max(0.0, min(1.0, float(response.strip())))
    except ValueError:
        return 0.0


def latency_percentiles(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0}
    ordered = sorted(latencies)

    def percentile(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {"p50": percentile(0.5), "p90": percentile(0.9), "p95": percentile(0.95)}


def summarize_scores(scores: list[float]) -> float:
    return statistics.fmean(scores) if scores else 0.0


@dataclass
class EvalResult:
    suite: str
    samples: int = 0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    latency: dict[str, float] = field(default_factory=lambda: {"p50": 0.0, "p90": 0.0, "p95": 0.0})

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "samples": self.samples,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "latency": self.latency,
        }


def load_golden(path: str | Path) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        if not _GOLDEN_KEYS.issubset(sample):
            raise ValueError(f"golden sample missing keys: {sorted(_GOLDEN_KEYS)}")
        samples.append(sample)
    return samples


def run_suite(retriever: HybridRetriever, samples: list[dict[str, object]], top_k: int = 6) -> EvalResult:
    result = EvalResult(suite="rag")
    precisions: list[float] = []
    recalls: list[float] = []
    faithfulness: list[float] = []
    latencies: list[float] = []
    for sample in samples:
        question = str(sample["question"])
        relevant = set(str(c) for c in sample["relevant_chunks"])
        start = time.perf_counter()
        hits = retriever.retrieve(question, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
        retrieved = [chunk_id for chunk_id, _score in hits]
        precisions.append(context_precision(retrieved, relevant))
        recalls.append(context_recall(retrieved, relevant))
        context = "\n\n".join(retriever.chunk_text(c) for c in retrieved)
        faithfulness.append(faithfulness_simple(str(sample["answer"]), context))
        result.samples += 1
    result.context_precision = summarize_scores(precisions)
    result.context_recall = summarize_scores(recalls)
    result.faithfulness = summarize_scores(faithfulness)
    result.answer_relevancy = summarize_scores(precisions)
    result.latency = latency_percentiles(latencies)
    return result


def write_report(result: EvalResult, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{result.suite} eval</title>
<style>body{{font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px;text-align:left}}</style>
</head><body><h1>NexusMind eval: {result.suite}</h1>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Samples</td><td>{result.samples}</td></tr>
<tr><td>Faithfulness</td><td>{result.faithfulness:.3f}</td></tr>
<tr><td>Answer relevancy</td><td>{result.answer_relevancy:.3f}</td></tr>
<tr><td>Context precision</td><td>{result.context_precision:.3f}</td></tr>
<tr><td>Context recall</td><td>{result.context_recall:.3f}</td></tr>
<tr><td>p50 / p90 / p95 latency</td><td>{result.latency['p50']:.0f} / {result.latency['p90']:.0f} / {result.latency['p95']:.0f} ms</td></tr>
</table></body></html>"""
    out.write_text(html, encoding="utf-8")