"""Minimal Model Context Protocol server over stdio (JSON-RPC 2.0)."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

PROTOCOL_VERSION = "2024-11-05"


class MCPServer:
    """A tiny MCP server: initialize, tools/list, tools/call."""

    def __init__(self, name: str, version: str) -> None:
        self._name = name
        self._version = version
        self._tools: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def add_tool(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        self._tools[name] = {"name": name, "description": description, "inputSchema": schema}
        self._handlers[name] = handler

    def _respond(self, req_id: int, result: Any) -> str:
        return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _error(self, req_id: int, code: int, message: str) -> str:
        return json.dumps(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
        )

    def handle(self, message: str) -> str | None:
        """Process one JSON-RPC message; returns a response (or None)."""
        try:
            request = json.loads(message)
        except json.JSONDecodeError:
            return self._error(-32700, -32700, "parse error")
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {}) or {}
        if method == "initialize":
            return self._respond(
                req_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self._name, "version": self._version},
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return self._respond(req_id, {"tools": list(self._tools.values())})
        if method == "tools/call":
            name = params.get("name", "")
            handler = self._handlers.get(name)
            if handler is None:
                return self._error(req_id, -32602, f"unknown tool: {name}")
            try:
                result = handler(params.get("arguments", {}))
            except Exception as exc:  # noqa: BLE001 - report any tool failure to the client
                return self._error(req_id, -32603, str(exc))
            return self._respond(req_id, {"content": [{"type": "text", "text": json.dumps(result)}]})
        return self._error(req_id, -32601, f"method not found: {method}")

    def serve_stdio(self) -> None:
        """Blocking stdio loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            response = self.handle(line)
            if response is not None:
                print(response, flush=True)