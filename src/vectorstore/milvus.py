"""Milvus implementation of the vector-store contract."""

import json

from pymilvus import MilvusClient

from .base import BaseVectorStore, Embedding


class MilvusVectorStore(BaseVectorStore):
    """Store and search document-chunk embeddings in Milvus."""

    def __init__(self, collection_name: str) -> None:
        self._collection_name = collection_name
        self._client: MilvusClient | None = None

    def connect(self, url: str, port: int, password: str | None = None) -> None:
        """Connect to Milvus, using ``password`` as a token when provided."""
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")

        endpoint = url.rstrip("/")
        if "://" not in endpoint:
            endpoint = f"http://{endpoint}"

        self._client = MilvusClient(
            uri=f"{endpoint}:{port}",
            token=password or "",
        )

    def insert(self, embeddings: list[Embedding]) -> None:
        """Create the collection when needed and insert embedding records."""
        if self._client is None:
            raise RuntimeError("connect must be called before insert")
        if not embeddings:
            return

        dimension = len(embeddings[0].vector)
        if dimension == 0:
            raise ValueError("embedding vectors cannot be empty")
        for embedding in embeddings:
            if len(embedding.vector) != dimension:
                raise ValueError("all embedding vectors must have the same dimension")
            if not embedding.document_id or not embedding.chunk_id:
                raise ValueError("document_id and chunk_id are required for insertion")

        if not self._client.has_collection(collection_name=self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=dimension,
                primary_field_name="chunk_id",
                id_type="string",
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
                max_length=512,
            )

        self._client.insert(
            collection_name=self._collection_name,
            data=[
                {
                    "vector": embedding.vector,
                    "document_id": embedding.document_id,
                    "chunk_id": embedding.chunk_id,
                    "text": embedding.text,
                    "metadata": embedding.metadata,
                }
                for embedding in embeddings
            ],
        )

    def search(self, query: Embedding, top_k: int) -> list[Embedding]:
        """Search Milvus with cosine similarity and return scored records."""
        if self._client is None:
            raise RuntimeError("connect must be called before search")
        if not query.vector:
            raise ValueError("query embedding cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        result = self._client.search(
            collection_name=self._collection_name,
            data=[query.vector],
            limit=top_k,
            output_fields=["vector", "document_id", "text", "metadata"],
            search_params={"metric_type": "COSINE"},
        )
        if not result:
            return []

        matches: list[Embedding] = []
        for hit in result[0]:
            entity = hit["entity"]
            chunk_id=str(hit["id"]) if "id" in hit else None,
            matches.append(
                Embedding(
                    vector=list(entity["vector"]),
                    document_id=str(entity["document_id"]),
                    chunk_id=chunk_id,
                    text=str(entity["text"]),
                    metadata=entity.get("metadata") or {},
                    score=float(hit["distance"]),
                )
            )
        return matches

    def delete_document(self, document_id: str) -> None:
        """Delete all embeddings associated with ``document_id``."""
        if self._client is None:
            raise RuntimeError("connect must be called before deletion")
        self._client.delete(
            collection_name=self._collection_name,
            filter=f"document_id == {json.dumps(document_id)}",
        )

    def delete_chunk(self, document_id: str, chunk_id: str) -> None:
        """Delete one chunk, constrained to its parent document."""
        if self._client is None:
            raise RuntimeError("connect must be called before deletion")
        self._client.delete(
            collection_name=self._collection_name,
            filter=(
                f"document_id == {json.dumps(document_id)} and "
                f"chunk_id == {json.dumps(chunk_id)}"
            ),
        )
