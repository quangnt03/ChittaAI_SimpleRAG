"""Abstract contract for vector-store implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class Embedding:
    """A document chunk, its vector, and optional search score."""

    vector: list[float]
    document_id: str = ""
    chunk_id: Optional[str] = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


class BaseVectorStore(ABC):
    """Connect to and perform basic operations on a vector store."""

    @abstractmethod
    def connect(self, url: str, port: int, password: str | None = None) -> None:
        """Connect to the vector store endpoint."""
        raise NotImplementedError

    @abstractmethod
    def insert(self, embeddings: list[Embedding]) -> None:
        """Insert a batch of embedding vectors."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: Embedding, top_k: int) -> list[Embedding]:
        """Return the ``top_k`` stored embeddings nearest to ``query``."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete every chunk belonging to a document."""
        raise NotImplementedError

    @abstractmethod
    def delete_chunk(self, document_id: str, chunk_id: str) -> None:
        """Delete one chunk belonging to a document."""
        raise NotImplementedError
