from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from nexusmind.agents.runtime import AgentRuntime, BudgetExceededError, Critic, Planner, Writer
from nexusmind.eval.harness import load_golden, run_suite, write_report
from nexusmind.llm.client import ChatMessage, LLMClient
from nexusmind.mcp.server import MCPServer
from nexusmind.retrieval.hybrid import BM25Index, DenseIndex, HybridRetriever
from nexusmind.tools.registry import ToolRegistry, tool


class _StubLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        super().__init__(__import__("nexusmind.config", fromlist=["Settings"]).Settings())

    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        if self.responses:
            return self.responses.pop(0)
        return ""

    async def close(self) -> None:
        pass


@tool
def get_ticket_status(ticket_id: str) -> dict:
    """Fetch the current status of a Jira ticket."""
    return {"ticket_id": ticket_id, "status": "in_progress"}


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(add)
    return registry


def test_tool_decorator() -> None:
    assert "ticket_id" in get_ticket_status.schema["properties"]
    assert get_ticket_status.invoke({"ticket_id": "JIRA-42"})["status"] == "in_progress"
    try:
        get_ticket_status.invoke({})
        assert False, "expected validation failure"
    except Exception:
        pass


def test_registry() -> None:
    registry = _registry()
    assert registry.get("add") is add
    assert registry.names() == ["add"]
    assert registry.schemas()[0]["name"] == "add"
    try:
        asyncio.run(registry.call("nope", {}))
        assert False, "expected ToolError"
    except Exception:
        pass


def test_runtime_finishes() -> None:
    llm = _StubLLM(
        [
            json.dumps({"type": "tool", "tool": "add", "arguments": {"a": 1, "b": 2}, "result": ""}),
            json.dumps({"type": "finish", "result": "three"}),
        ]
    )
    events: list[dict[str, object]] = []

    async def run() -> None:
        async for event in AgentRuntime(llm, _registry(), budget_tokens=1000, max_iterations=4).run(
            "sys", "user", {"type": "object"}
        ):
            events.append(event.to_dict())

    asyncio.run(run())
    assert any(e.get("result") == "three" for e in events if e["event"] == "done")


def test_runtime_budget_exceeded() -> None:
    llm = _StubLLM([json.dumps({"type": "finish", "result": " ".join(["x"] * 5000)})])

    async def run() -> None:
        async for _event in AgentRuntime(llm, _registry(), budget_tokens=100, max_iterations=2).run(
            "sys", "user", {"type": "object"}
        ):
            pass

    try:
        asyncio.run(run())
        assert False, "expected BudgetExceededError"
    except BudgetExceededError:
        pass


def test_runtime_halt_on_max_iterations() -> None:
    llm = _StubLLM([json.dumps({"type": "tool", "tool": "add", "arguments": {"a": 1, "b": 1}, "result": ""})] * 10)
    events: list[dict[str, object]] = []

    async def run() -> None:
        async for event in AgentRuntime(llm, _registry(), budget_tokens=100_000, max_iterations=2).run(
            "sys", "user", {"type": "object"}
        ):
            events.append(event.to_dict())

    asyncio.run(run())
    done = [e for e in events if e["event"] == "done"]
    assert done and done[-1].get("halted") is True


def test_planner_and_critic_and_writer() -> None:
    plan = asyncio.run(Planner(_StubLLM([json.dumps({"type": "finish", "result": "sub one\nsub two"})])).plan("q"))
    assert plan == ["sub one", "sub two"]
    grounded, _note = asyncio.run(Critic(_StubLLM([json.dumps({"type": "finish", "result": "grounded"})])).check("a", "c"))
    assert grounded is True
    grounded, note = asyncio.run(
        Critic(_StubLLM([json.dumps({"type": "tool", "tool": "x", "result": "ungrounded: claim"})])).check("a", "c")
    )
    assert grounded is False and "ungrounded" in note
    text = asyncio.run(Writer(_StubLLM([json.dumps({"type": "finish", "result": "final answer"})])).write("f", "q"))
    assert text == "final answer"


def test_eval_metrics_and_suite(tmp_path) -> None:
    from nexusmind.eval.harness import context_precision, context_recall, faithfulness_simple

    assert context_precision(["a", "b", "c"], {"a"}) == pytest.approx(1 / 3)
    assert context_recall(["a", "b"], {"a", "b", "c"}) == 2 / 3
    assert faithfulness_simple("alpha beta", "alpha beta gamma") == 1.0
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text(
        '{"question": "q1", "answer": "a1", "relevant_chunks": ["c1"]}\n'
        '{"question": "q2", "answer": "a2", "relevant_chunks": ["c2"]}\n',
        encoding="utf-8",
    )
    samples = load_golden(dataset)
    assert len(samples) == 2
    bm25 = BM25Index()
    bm25.add("c1", "q1 a1 q1 a1")
    bm25.add("c2", "q2 a2 q2 a2")
    result = run_suite(HybridRetriever(bm25, DenseIndex(), rerank=False, top_k=2), samples, top_k=2)
    assert result.samples == 2
    assert result.context_recall > 0.5
    out = tmp_path / "report.html"
    write_report(result, out)
    assert "Faithfulness" in out.read_text(encoding="utf-8")


def test_mcp_server_roundtrip() -> None:
    server = MCPServer("nexusmind", "0.3.0")
    server.add_tool("add", "Add two numbers", {"type": "object"}, lambda args: args["a"] + args["b"])
    init = json.loads(server.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})))
    assert init["result"]["serverInfo"]["name"] == "nexusmind"
    listed = json.loads(server.handle(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})))
    assert listed["result"]["tools"][0]["name"] == "add"
    called = json.loads(
        server.handle(
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "add", "arguments": {"a": 2, "b": 3}}})
        )
    )
    assert "5" in called["result"]["content"][0]["text"]
    error = json.loads(
        server.handle(json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}))
    )
    assert error["error"]["code"] == -32602
    not_found = json.loads(server.handle(json.dumps({"jsonrpc": "2.0", "id": 5, "method": "bogus", "params": {}})))
    assert not_found["error"]["code"] == -32601