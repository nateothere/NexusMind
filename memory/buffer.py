"""Memory tiers - conversation buffer and episodic memory."""

from __future__ import annotations

from ..ingestion.pipeline import Chunk
from ..retrieval.hybrid import DenseIndex
from ..utils import new_id


class ConversationBuffer:
    """Rolling window of recent conversation turns."""

    def __init__(self, max_turns: int = 8) -> None:
        self._max_turns = max_turns
        self._turns: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]

    def history(self) -> list[dict[str, str]]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns = []

    def __len__(self) -> int:
        return len(self._turns)


class EpisodicMemory:
    """Long-term memory: past answers stored back into the dense index."""

    def __init__(self, dense: DenseIndex) -> None:
        self._dense = dense
        self._episodes: list[dict[str, str]] = []

    def remember(self, question: str, answer: str) -> str:
        """Store an episode and index it for future recall."""
        episode_id = new_id("episode")
        self._episodes.append({"id": episode_id, "question": question, "answer": answer})
        text = f"Q: {question}\nA: {answer}"
        if self._dense._embeddings is not None:  # noqa: SLF001
            self._dense.add(episode_id, text, self._dense._embeddings.embed(text))  # noqa: SLF001
        return episode_id

    def recall(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        """Return past episodes relevant to a query (best effort)."""
        if self._dense._embeddings is None:  # noqa: SLF001
            return self._episodes[-top_k:]
        hits = self._dense.search_text(query, top_k=top_k)
        by_id = {episode["id"]: episode for episode in self._episodes}
        return [by_id[chunk_id] for chunk_id, _score in hits if chunk_id in by_id]

    def count(self) -> int:
        return len(self._episodes)