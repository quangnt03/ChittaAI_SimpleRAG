"""Abstract contract for RAG indexing implementations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from langchain_core.documents import Document


class BaseIndexer(ABC):
    """Store documents in the index used by a RAG pipeline."""

    @abstractmethod
    def index(self, documents: Sequence[Document]) -> list[str]:
        """Index ``documents`` and return their backend-assigned identifiers.

        Implementations are responsible for any chunking, embedding, and
        persistence required by their indexing backend.
        """
        raise NotImplementedError
    
    @abstractmethod
    def splitter(self, documents: Sequence[Document]) -> list[Document]:
        """Split ``documents`` into the chunks that will be indexed."""
        raise NotImplementedError
