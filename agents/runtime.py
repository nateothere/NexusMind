"""Async agent runtime + the four roles."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..llm.client import ChatMessage, LLMClient
from ..tools.registry import ToolRegistry
from ..utils import NexusMindError, token_count


class BudgetExceededError(NexusMindError):
    pass


class AgentEvent:
    def __init__(self, kind: str, payload: dict[str, object]) -> None:
        self.kind = kind
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        return {"event": self.kind, **self.payload}


class AgentRuntime:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        budget_tokens: int = 24_000,
        max_iterations: int = 8,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._budget = budget_tokens
        self._max_iterations = max_iterations

    async def run(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float | None = None,
    ) -> AsyncIterator[AgentEvent]:
        messages: list[ChatMessage] = [self._llm.system(system), self._llm.user(user)]
        spent = 0
        for _iteration in range(self._max_iterations):
            yield AgentEvent("think", {"iteration": _iteration})
            response = await self._llm.complete(messages, temperature=temperature, json_schema=schema)
            spent += token_count(response)
            if spent > self._budget:
                raise BudgetExceededError(f"budget exceeded ({spent} tokens)")
            try:
                action = json.loads(response)
            except json.JSONDecodeError as exc:
                raise NexusMindError("agent returned invalid JSON") from exc
            yield AgentEvent("action", {"action": action})
            if action.get("type") == "finish":
                yield AgentEvent("done", {"result": action.get("result", ""), "tokens": spent})
                return
            tool_name = action.get("tool")
            args = action.get("arguments", {})
            if not isinstance(args, dict) or not tool_name:
                yield AgentEvent("error", {"message": "malformed tool action"})
                continue
            result = await self._tools.call(str(tool_name), args)
            spent += token_count(str(result))
            if spent > self._budget:
                raise BudgetExceededError(f"budget exceeded ({spent} tokens)")
            yield AgentEvent("tool_result", {"tool": tool_name, "result": result})
            messages.append(self._llm.assistant(json.dumps(action)))
            messages.append(self._llm.user(f"Tool {tool_name} returned: {json.dumps(result)[:4000]}"))
        yield AgentEvent("done", {"result": "", "tokens": spent, "halted": True})


_PLANNER_PROMPT = """You are the Planner. Decompose the user's question into
sub-questions when it is multi-hop or entity-centric. Output JSON:
{"type": "finish"|"tool", "result": "sub-question 1\nsub-question 2",
"reason": "why"}. For simple questions, finish with the query unchanged."""

_RESEARCHER_PROMPT = """You are the Researcher. Answer the question using the
retrieved context and the available tools. Follow ReAct: think, then either
call a tool or finish. Output JSON: {"type": "tool"|"finish",
"tool": "<name>", "arguments": {...}, "result": "answer or 'need more context'"}.
Never invent facts not present in the context."""

_CRITIC_PROMPT = """You are the Critic. Verify the Researcher's answer: every
claim must be grounded in the retrieved context. If any claim is ungrounded,
output {"type": "tool", "tool": "retriever_search", "arguments": {"query":
"<reformulated query>"}, "result": "ungrounded: <claim>"}. Otherwise output
{"type": "finish", "result": "grounded"}."""

_WRITER_PROMPT = """You are the Writer. Synthesize the final answer from the
verified facts. Cite sources inline as [n] where n matches the context index.
Output JSON: {"type": "finish", "result": "the final answer text"}."""

_PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["finish", "tool"]},
        "result": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["type", "result"],
    "additionalProperties": False,
}

_AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["tool", "finish"]},
        "tool": {"type": "string"},
        "arguments": {"type": "object"},
        "result": {"type": "string"},
    },
    "required": ["type", "result"],
    "additionalProperties": False,
}


class Planner:
    def __init__(self, llm: LLMClient, budget: int = 4_000) -> None:
        self._llm = llm
        self._budget = budget

    async def plan(self, query: str) -> list[str]:
        response = await self._llm.complete(
            [self._llm.system(_PLANNER_PROMPT), self._llm.user(query)],
            json_schema=_PLANNER_SCHEMA,
        )
        data = json.loads(response)
        return [line.strip() for line in data.get("result", "").splitlines() if line.strip()]


class Researcher:
    def __init__(self, llm: LLMClient, tools: ToolRegistry, budget: int = 24_000) -> None:
        self._runtime = AgentRuntime(llm, tools, budget_tokens=budget, max_iterations=8)

    async def run(self, question: str, context: str) -> AsyncIterator[dict[str, object]]:
        async for event in self._runtime.run(
            _RESEARCHER_PROMPT, f"Context:\n{context}\n\nQuestion: {question}", _AGENT_SCHEMA
        ):
            yield event.to_dict()


class Critic:
    def __init__(self, llm: LLMClient, budget: int = 8_000) -> None:
        self._llm = llm
        self._budget = budget

    async def check(self, answer: str, context: str) -> tuple[bool, str]:
        response = await self._llm.complete(
            [self._llm.system(_CRITIC_PROMPT), self._llm.user(f"Context:\n{context}\n\nAnswer: {answer}")],
            json_schema=_AGENT_SCHEMA,
        )
        data = json.loads(response)
        return data.get("type") == "finish", str(data.get("result", ""))


class Writer:
    def __init__(self, llm: LLMClient, budget: int = 8_000) -> None:
        self._llm = llm
        self._budget = budget

    async def write(self, facts: str, question: str) -> str:
        response = await self._llm.complete(
            [self._llm.system(_WRITER_PROMPT), self._llm.user(f"Question: {question}\n\nVerified facts:\n{facts}")],
            json_schema=_AGENT_SCHEMA,
        )
        data = json.loads(response)
        result = str(data.get("result", ""))
        return result[: self._budget] if token_count(result) > self._budget else result