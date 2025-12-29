"""LLM client (OpenAI-compatible) + Ollama embeddings."""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel

from ..config import Settings
from ..utils import ModelUnavailableError

_SYSTEM = "system"
_USER = "user"
_ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMClient:
    def __init__(self, settings: Settings, timeout: float = 120.0) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(base_url=settings.llm_base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(f"LLM endpoint unreachable: {exc}") from exc

    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature if temperature is not None else self._settings.temperature,
            "stream": False,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": json_schema.get("name", "result"), "strict": True, "schema": json_schema},
            }
        data = await self._post(payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ModelUnavailableError("LLM returned an empty completion") from exc

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self._settings.temperature,
            "stream": True,
        }
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(f"LLM stream failed: {exc}") from exc

    @staticmethod
    def system(content: str) -> ChatMessage:
        return ChatMessage(role=_SYSTEM, content=content)

    @staticmethod
    def user(content: str) -> ChatMessage:
        return ChatMessage(role=_USER, content=content)

    @staticmethod
    def assistant(content: str) -> ChatMessage:
        return ChatMessage(role=_ASSISTANT, content=content)


class OllamaEmbeddings:
    def __init__(self, settings: Settings, timeout: float = 60.0) -> None:
        self._settings = settings
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def embed(self, text: str) -> np.ndarray:
        try:
            resp = self._client.post(
                f"{self._settings.llm_base_url}/embeddings",
                json={"model": self._settings.embed_model, "prompt": text},
            )
            resp.raise_for_status()
            vec = np.asarray(resp.json()["embedding"], dtype=np.float32)
        except (httpx.HTTPError, KeyError) as exc:
            raise ModelUnavailableError(f"embedding endpoint unreachable: {exc}") from exc
        return normalize_vector(vec)

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b)) / denom


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0 or math.isclose(norm, 1.0):
        return vec
    return vec / norm