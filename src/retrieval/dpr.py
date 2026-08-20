"""Dense passage retrieval backed by a vector store."""

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from vectorstore.base import BaseVectorStore, Embedding

from .base import BaseRetriever


class Retriever(BaseRetriever):
    """Retrieve passages using query embeddings and a vector store."""

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
        """Return passages matching an embedded query and their confidence."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = list(query_embedding)
        if not vector:
            raise ValueError("query_embedding cannot be empty")

        results = self._vector_store.search(
            Embedding(vector=vector),
            top_k,
        )

        matches: list[tuple[Document, float]] = []
        for result in results:
            confidence = (
                0.0
                if result.score is None
                else max(0.0, min(1.0, (result.score + 1.0) / 2.0))
            )
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
