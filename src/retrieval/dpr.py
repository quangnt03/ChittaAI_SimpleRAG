"""Dense passage retrieval backed by a vector store."""

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from vectorstore.base import BaseVectorStore, Embedding

from .base import BaseRetriever


class Retriever(BaseRetriever):
    """Retrieve passages with native hybrid search when the store supports it."""

    def __init__(
        self,
        embedding_model: OpenAIEmbeddings,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store

    def embed_query(self, query: str) -> list[float]:
        """Embed a non-empty query with the configured embedding model."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        return list(self._embedding_model.embed_query(query))

    def search(
        self, query_embedding: Sequence[float], *, top_k: int = 4
    ) -> list[tuple[Document, float]]:
        """Return dense-only matches for an already embedded query."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = list(query_embedding)
        if not vector:
            raise ValueError("query_embedding cannot be empty")

        results = self._vector_store.search(
            Embedding(vector=vector),
            top_k,
        )

        return self._to_documents(results)

    def retrieve(
        self, query: str, *, top_k: int = 4
    ) -> list[tuple[Document, float]]:
        """Embed ``query`` and run dense plus BM25 retrieval with RRF fusion."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_embedding = self.embed_query(query)
        results = self._vector_store.hybrid_search(
            Embedding(vector=query_embedding, text=query),
            top_k,
        )
        return self._to_documents(results)

    @staticmethod
    def _to_documents(
        results: Sequence[Embedding],
    ) -> list[tuple[Document, float]]:
        matches: list[tuple[Document, float]] = []
        for result in results:
            if result.score is None:
                confidence = 0.0
            elif result.score_kind == "confidence":
                confidence = max(0.0, min(1.0, result.score))
            else:
                confidence = max(0.0, min(1.0, (result.score + 1.0) / 2.0))
            matches.append(
                (
                    Document(
                        page_content=result.text,
                        metadata={
                            **result.metadata,
                            "document_id": result.document_id,
                            "chunk_id": result.chunk_id,
                        },
                    ),
                    confidence,
                )
            )
        return matches
