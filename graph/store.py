"""Knowledge graph - SQLite store, entity extraction, expansion."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..llm.client import LLMClient
from ..utils import new_id, unique


class Entity:
    def __init__(self, entity_id: str, name: str, kind: str, attributes: dict[str, str]) -> None:
        self.id = entity_id
        self.name = name
        self.kind = kind
        self.attributes = attributes

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "name": self.name, "kind": self.kind, "attributes": self.attributes}


class Relation:
    def __init__(self, relation_id: str, src: str, dst: str, predicate: str, weight: float) -> None:
        self.id = relation_id
        self.src = src
        self.dst = dst
        self.predicate = predicate
        self.weight = weight

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "src": self.src, "dst": self.dst, "predicate": self.predicate, "weight": self.weight}


class GraphStore:
    def __init__(self, path: str | Path = "./data/graph") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, name TEXT, "
            "kind TEXT, attributes TEXT, collection TEXT)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS relations (id TEXT PRIMARY KEY, src TEXT, "
            "dst TEXT, predicate TEXT, weight REAL, collection TEXT)"
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def upsert_entity(self, name: str, kind: str, collection: str, attributes: dict[str, str] | None = None) -> Entity:
        row = self._db.execute(
            "SELECT * FROM entities WHERE name = ? AND collection = ?", (name, collection)
        ).fetchone()
        if row is not None:
            return Entity(row["id"], row["name"], row["kind"], json.loads(row["attributes"]))
        entity = Entity(new_id("ent"), name, kind, attributes or {})
        self._db.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
            (entity.id, entity.name, entity.kind, json.dumps(entity.attributes), collection),
        )
        self._db.commit()
        return entity

    def add_relation(self, src: str, dst: str, predicate: str, collection: str, weight: float = 1.0) -> Relation:
        relation = Relation(new_id("rel"), src, dst, predicate, weight)
        self._db.execute(
            "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?)",
            (relation.id, relation.src, relation.dst, relation.predicate, relation.weight, collection),
        )
        self._db.commit()
        return relation

    def expand(self, entity_ids: list[str], hops: int = 1, collection: str = "default") -> list[dict[str, object]]:
        if hops < 1 or not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        neighbors: list[dict[str, object]] = []
        seen: set[str] = set(entity_ids)
        frontier = list(entity_ids)
        for _hop in range(hops):
            next_frontier: list[str] = []
            rows = self._db.execute(
                f"SELECT r.id AS rid, r.src, r.dst, r.predicate, r.weight, "
                f"e.id AS eid, e.name AS ename, e.kind AS ekind, e.attributes AS eattributes "
                f"FROM relations r JOIN entities e ON e.id = r.dst "
                f"WHERE r.src IN ({placeholders}) AND r.collection = ?",
                (*frontier, collection),
            ).fetchall()
            for row in rows:
                entity = Entity(row["eid"], row["ename"], row["ekind"], json.loads(row["eattributes"]))
                if entity.id in seen:
                    continue
                seen.add(entity.id)
                relation = Relation(row["rid"], row["src"], row["dst"], row["predicate"], row["weight"])
                neighbors.append({"relation": relation.to_dict(), "entity": entity.to_dict()})
                next_frontier.append(entity.id)
            frontier = next_frontier
            if not frontier:
                break
        return neighbors

    def list_entities(self, collection: str, limit: int = 500) -> list[Entity]:
        rows = self._db.execute(
            "SELECT * FROM entities WHERE collection = ? LIMIT ?", (collection, limit)
        ).fetchall()
        return [Entity(row["id"], row["name"], row["kind"], json.loads(row["attributes"])) for row in rows]

    def remove_collection(self, collection: str) -> None:
        self._db.execute("DELETE FROM relations WHERE collection = ?", (collection,))
        self._db.execute("DELETE FROM entities WHERE collection = ?", (collection,))
        self._db.commit()

    def entity_count(self, collection: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) AS c FROM entities WHERE collection = ?", (collection,)
        ).fetchone()
        return int(row["c"])


_ENTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "attributes": {"type": "object"},
                },
                "required": ["name", "kind", "attributes"],
                "additionalProperties": False,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"src": {"type": "string"}, "dst": {"type": "string"}, "predicate": {"type": "string"}},
                "required": ["src", "dst", "predicate"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relations"],
    "additionalProperties": False,
}

_EXTRACT_PROMPT = """Extract the entities and the relations between them from
the chunk below. Entity names must match the text exactly (no paraphrasing).
Predicates must be snake_case verbs ("depends_on", "implements", "breaks").
Return only the JSON object."""


class EntityExtractor:
    def __init__(self, llm: LLMClient, store: GraphStore) -> None:
        self._llm = llm
        self._store = store

    async def extract(self, chunks: list[tuple[str, str]], collection: str) -> None:
        for chunk_id, text in chunks:
            result = await self._llm.complete(
                [self._llm.system(_EXTRACT_PROMPT), self._llm.user(f"Chunk ({chunk_id}):\n{text[:4000]}")],
                json_schema=_ENTITY_SCHEMA,
            )
            data = json.loads(result)
            entities: dict[str, str] = {}
            for item in data.get("entities", []):
                name = str(item["name"]).strip()
                if not name:
                    continue
                entity = self._store.upsert_entity(
                    name, str(item.get("kind", "concept")), collection,
                    {str(k): str(v) for k, v in item.get("attributes", {}).items()},
                )
                entities[name] = entity.id
            for item in data.get("relations", []):
                src = entities.get(str(item["src"]))
                dst = entities.get(str(item["dst"]))
                if src and dst:
                    self._store.add_relation(src, dst, str(item["predicate"]), collection)


def link_entities(store: GraphStore, text: str, collection: str, limit: int = 6) -> list[str]:
    candidates: list[str] = []
    for entity in store.list_entities(collection):
        if any(word in text.lower() for word in entity.name.lower().split()):
            candidates.append(entity.id)
    return unique(candidates)[:limit]


class GraphExpander:
    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def expand(self, query: str, collection: str = "default", hops: int = 1) -> list[str]:
        if hops < 1:
            return []
        entity_ids = link_entities(self._store, query, collection)
        if not entity_ids:
            return []
        neighbors = self._store.expand(entity_ids, hops=hops, collection=collection)
        return unique([str(entry["entity"]["name"]) for entry in neighbors])  # type: ignore[index]

    def context_block(self, query: str, collection: str = "default", hops: int = 1) -> str:
        names = self.expand(query, collection=collection, hops=hops)
        if not names:
            return ""
        return "Related entities: " + ", ".join(names) + "."