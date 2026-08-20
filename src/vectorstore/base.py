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
    score_kind: str = "cosine"


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

    def hybrid_search(self, query: Embedding, top_k: int) -> list[Embedding]:
        """Search dense and lexical representations when the backend supports it.

        Backends without native hybrid search retain useful behavior by falling
        back to their dense-vector implementation.
        """
        return self.search(query, top_k)

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete every chunk belonging to a document."""
        raise NotImplementedError

    @abstractmethod
    def delete_chunk(self, document_id: str, chunk_id: str) -> None:
        """Delete one chunk belonging to a document."""
        raise NotImplementedError
