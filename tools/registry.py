"""Typed tool system - @tool decorator, registry, builtins, MCP bridge."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import create_model

from ..utils import ToolError

_TOOL_MODEL_ATTR = "_model"


class Tool:
    def __init__(self, name: str, description: str, func: Callable[..., Any], schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.schema = schema

    def invoke(self, arguments: dict[str, Any]) -> Any:
        try:
            validated = self._model.model_validate(arguments)
            return self.func(**validated.model_dump())
        except Exception as exc:
            raise ToolError(f"tool {self.name} failed: {exc}") from exc


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": tool.name, "description": tool.description, "parameters": tool.schema}
            for tool in self._tools.values()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}")
        return tool.invoke(arguments)


def tool(func: Callable[..., Any]) -> Tool:
    name = func.__name__
    doc = inspect.getdoc(func)
    description = doc.strip().splitlines()[0] if doc else func.__name__
    signature = inspect.signature(func)
    fields: dict[str, tuple[Any, ...]] = {}
    for param_name, param in signature.parameters.items():
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"tool {name}: parameter {param_name} needs a type annotation")
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[param_name] = (param.annotation, default)
    model = create_model(f"{name}_args", **fields)  # type: ignore[arg-type, call-overload]
    tool_obj = Tool(name, description, func, model.model_json_schema())
    setattr(tool_obj, _TOOL_MODEL_ATTR, model)
    return tool_obj


def retriever_search(retriever: Any) -> Tool:
    @tool
    def search(query: str, top_k: int = 6) -> dict[str, Any]:
        """Search the knowledge base and return the top chunks with scores."""
        hits = retriever.retrieve(query, top_k=top_k)
        return {
            "results": [
                {"chunk_id": chunk_id, "score": round(score, 4), "text": retriever.chunk_text(chunk_id)[:1500]}
                for chunk_id, score in hits
            ]
        }

    return search


from datetime import date  # noqa: E402


@tool
def get_date() -> str:
    """Return today's date (YYYY-MM-DD)."""
    return date.today().isoformat()


@tool
def count_words(text: str) -> int:
    """Count the words in a piece of text."""
    from ..utils import tokenize

    return len(tokenize(text))


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def register_builtins(registry: ToolRegistry, retriever: Any = None) -> None:
    for builtin in (get_date, count_words, add):
        registry.register(builtin)
    if retriever is not None:
        registry.register(retriever_search(retriever))


class MCPToolClient:
    def __init__(self, server_url: str, timeout: float = 30.0) -> None:
        self._url = server_url.rstrip("/") + "/mcp"
        self._client = httpx.Client(timeout=timeout)

    def list_tools(self) -> list[dict[str, Any]]:
        resp = self._client.post(
            self._url,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        resp.raise_for_status()
        return resp.json().get("result", {}).get("tools", [])

    def attach(self, registry: ToolRegistry) -> list[str]:
        attached: list[str] = []
        for entry in self.list_tools():
            name = entry["name"]

            def call(args: dict[str, Any], _name: str = name) -> Any:
                resp = self._client.post(
                    self._url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": _name, "arguments": args},
                    },
                )
                resp.raise_for_status()
                return resp.json().get("result", {})

            registry.register(Tool(name, entry.get("description", ""), call, entry.get("inputSchema", {"type": "object"})))
            attached.append(name)
        return attached

    def close(self) -> None:
        self._client.close()