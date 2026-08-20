"""Simple document indexer."""

from collections.abc import Sequence
from uuid import uuid4

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vectorstore.base import BaseVectorStore, Embedding

from .base import BaseIndexer


class Indexer(BaseIndexer):
    """Split, embed, and store documents in a vector store."""

    def __init__(
        self,
        embedding_model: OpenAIEmbeddings,
        vector_store: BaseVectorStore,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def splitter(self, documents: Sequence[Document]) -> list[Document]:
        """Split documents recursively while retaining their metadata."""
        prepared_documents: list[Document] = []
        for document in documents:
            metadata = dict(document.metadata)
            metadata["document_id"] = str(
                document.id or metadata.get("document_id") or uuid4()
            )
            prepared_documents.append(
                Document(page_content=document.page_content, metadata=metadata)
            )
        return self._splitter.split_documents(prepared_documents)

    def index(self, documents: Sequence[Document]) -> list[str]:
        """Embed document chunks and insert them into Milvus."""
        chunks = self.splitter(documents)
        if not chunks:
            return []

        vectors = self._embedding_model.embed_documents(
            [chunk.page_content for chunk in chunks]
        )

        embeddings = [
            Embedding(
                vector=vector,
                document_id=str(chunk.metadata["document_id"]),
                chunk_id=str(uuid4()),
                text=chunk.page_content,
                metadata={
                    key: value
                    for key, value in chunk.metadata.items()
                    if key != "document_id"
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._vector_store.insert(embeddings)
        return [embedding.chunk_id for embedding in embeddings]
