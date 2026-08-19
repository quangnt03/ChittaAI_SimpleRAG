"""Abstract contract for RAG response-generation implementations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from langchain_core.documents import Document


class BaseGenerator(ABC):
    """Generate an answer grounded in retrieved context."""

    @abstractmethod
    def generate(self, query: str, context: Sequence[Document]) -> str:
        """Generate an answer to ``query`` using the ranked ``context``."""
        raise NotImplementedError
