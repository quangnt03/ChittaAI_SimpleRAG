"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed application settings loaded from ``.env.local``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env.local",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    milvus_url: str = "http://localhost"
    milvus_port: int = Field(default=19530, ge=1, le=65535)
    milvus_collection: str = "rag_documents"
    milvus_hnsw_m: int = Field(default=16, ge=2)
    milvus_hnsw_ef_construction: int = Field(default=200, ge=1)
    milvus_hnsw_ef: int = Field(default=64, ge=1)
    milvus_bm25_k1: float = Field(default=1.2, gt=0)
    milvus_bm25_b: float = Field(default=0.75, ge=0, le=1)
    milvus_sparse_drop_ratio: float = Field(default=0.0, ge=0, lt=1)
    milvus_candidate_multiplier: int = Field(default=4, ge=1)
    milvus_rrf_k: int = Field(default=60, ge=1, lt=16384)
    milvus_consistency_level: Literal[
        "Strong", "Session", "Bounded", "Eventually"
    ] = "Session"
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        """Ensure every chunk advances through the source document."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, loading and validating them once."""
    return Settings()
