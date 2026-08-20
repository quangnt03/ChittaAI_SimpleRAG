"""Abstract contract for RAG retrieval implementations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from langchain_core.documents import Document


class BaseRetriever(ABC):
    """Retrieve query-relevant documents from an index."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Convert a text query into a vector accepted by the index."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query_embedding: Sequence[float], *, top_k: int = 4
    ) -> list[tuple[Document, float]]:
        """Search with an embedded query and return ranked documents."""
        raise NotImplementedError

    def retrieve(
        self, query: str, *, top_k: int = 4
    ) -> list[tuple[Document, float]]:
        """Embed ``query`` and return ranked documents with confidence scores."""
        return self.search(self.embed_query(query), top_k=top_k)
