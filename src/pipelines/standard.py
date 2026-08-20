"""Standard retrieval-augmented generation pipeline."""

from collections.abc import Sequence

from langchain_core.documents import Document

from generation.base import BaseGenerator
from retrieval.base import BaseRetriever


SearchResult = tuple[Document, float]


class StandardPipeline:
    """Compose query embedding, vector search, and grounded generation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
    ) -> None:
        self._retriever = retriever
        self._generator = generator

    def embed_query(self, query: str) -> list[float]:
        """Embed ``query`` using the configured retriever."""
        return self._retriever.embed_query(query)

    def search(
        self, query_embedding: Sequence[float], *, top_k: int = 4
    ) -> list[SearchResult]:
        """Find the most relevant documents for ``query_embedding``."""
        return self._retriever.search(query_embedding, top_k=top_k)

    def generate(self, query: str, matches: Sequence[SearchResult]) -> str:
        """Generate an answer grounded in the documents from ``matches``."""
        return self._generator.generate(
            query,
            [document for document, _confidence in matches],
        )

    def run(self, query: str, *, top_k: int = 4) -> str:
        """Run embedding, search, and generation for one query."""
        query_embedding = self.embed_query(query)
        matches = self.search(query_embedding, top_k=top_k)
        return self.generate(query, matches)
