"""Abstract contract for RAG retrieval implementations."""

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseRetriever(ABC):
    """Retrieve query-relevant documents from an index."""

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 4) -> list[Document]:
        """Return up to ``top_k`` documents ordered by relevance."""
        raise NotImplementedError
