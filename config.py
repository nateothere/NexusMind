"""Configuration - pydantic-settings over env vars, YAML for behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrieverConfig(BaseModel):
    """Retrieval behavior."""

    top_k: int = 12
    rerank: bool = True
    graph_expansion: int = 1
    rrf_k: int = 60
    hyde: bool = False
    multi_query: bool = False


class AgentsConfig(BaseModel):
    """Agent runtime limits."""

    max_iterations: int = 8
    budget_tokens: int = 60_000
    temperature: float = 0.2


class IngestionConfig(BaseModel):
    """Ingestion pipeline behavior."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_chunking: bool = True
    parser: str = "native"
    recursive: bool = False


class MemoryConfig(BaseModel):
    """Memory tiers."""

    buffer_turns: int = 8
    episodic: bool = True


class Settings(BaseSettings):
    """Environment-driven settings (NEXUSMIND_* prefix)."""

    model_config = SettingsConfigDict(env_prefix="NEXUSMIND_", env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:14b-instruct"
    embed_model: str = "bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    qdrant_url: str = ""
    kuzu_path: str = "./data/graph"
    redis_url: str = "redis://localhost:6379/0"
    data_dir: str = "./data"
    api_key: str = ""
    max_context_tokens: int = 32_768
    temperature: float = 0.2
    log_level: str = "INFO"
    langfuse_key: str = ""

    @model_validator(mode="after")
    def _check_temperature(self) -> Settings:
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be within [0, 1]")
        return self


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML behavior file into plain dicts."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def config_from_yaml(path: str | Path, settings: Settings | None = None) -> dict[str, Any]:
    """Merge YAML sections into a config dict for engine construction."""
    yaml_data = load_yaml_config(path)
    out: dict[str, Any] = {}
    if "retriever" in yaml_data:
        out["retriever"] = RetrieverConfig(**yaml_data["retriever"])
    if "agents" in yaml_data:
        out["agents"] = AgentsConfig(**yaml_data["agents"])
    if "ingestion" in yaml_data:
        out["ingestion"] = IngestionConfig(**yaml_data["ingestion"])
    if "memory" in yaml_data:
        out["memory"] = MemoryConfig(**yaml_data["memory"])
    if settings is not None:
        out["settings"] = settings
    return out