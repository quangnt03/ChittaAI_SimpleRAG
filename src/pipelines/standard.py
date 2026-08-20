"""Standard retrieval-augmented generation pipeline."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from generation.base import BaseGenerator
from retrieval.base import BaseRetriever


SearchResult = tuple[Document, float]
CITATION_QUOTE_MAX_CHARS = 240


@dataclass(frozen=True, slots=True)
class Citation:
    """A retrieved document source associated with an answer."""

    number: int
    source: str
    confidence: float
    quote: str = ""


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """One assistant message and the documents that support it."""

    message: str
    citations: tuple[Citation, ...] = ()


def _metadata_text(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _source_identity(document: Document, fallback: int) -> str:
    metadata = document.metadata
    for key in ("source", "document_id", "file_name", "relative_path", "chunk_id"):
        value = _metadata_text(metadata, key)
        if value is not None:
            return f"{key}:{value}"
    return f"retrieved:{fallback}"


def _source_label(document: Document, fallback: int) -> str:
    metadata = document.metadata
    for key in ("relative_path", "file_name", "source", "document_id", "chunk_id"):
        value = _metadata_text(metadata, key)
        if value is not None:
            return value
    return f"Retrieved document {fallback}"


def _document_quote(document: Document) -> str:
    """Return a compact, single-line quote from a retrieved chunk."""
    quote = " ".join(document.page_content.split())
    if len(quote) <= CITATION_QUOTE_MAX_CHARS:
        return quote
    return quote[: CITATION_QUOTE_MAX_CHARS - 3].rstrip() + "..."


def _prepare_citations(
    matches: Sequence[SearchResult],
) -> tuple[list[Document], tuple[Citation, ...]]:
    """Number retrieved sources and deduplicate chunks from the same document."""
    citation_numbers: dict[str, int] = {}
    citations: list[Citation] = []
    documents: list[Document] = []

    for position, (document, confidence) in enumerate(matches, start=1):
        identity = _source_identity(document, position)
        citation_number = citation_numbers.get(identity)
        if citation_number is None:
            citation_number = len(citations) + 1
            citation_numbers[identity] = citation_number
            citations.append(
                Citation(
                    number=citation_number,
                    source=_source_label(document, position),
                    confidence=confidence,
                    quote=_document_quote(document),
                )
            )
        elif confidence > citations[citation_number - 1].confidence:
            citations[citation_number - 1] = replace(
                citations[citation_number - 1],
                confidence=confidence,
                quote=_document_quote(document),
            )

        documents.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "citation_number": citation_number,
                    "citation_source": citations[citation_number - 1].source,
                },
            )
        )

    return documents, tuple(citations)


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
        """Run dense-only search for an already embedded query."""
        return self._retriever.search(query_embedding, top_k=top_k)

    def retrieve(self, query: str, *, top_k: int = 4) -> list[SearchResult]:
        """Run the retriever's complete query flow, including hybrid search."""
        return self._retriever.retrieve(query, top_k=top_k)

    def generate(self, query: str, matches: Sequence[SearchResult]) -> str:
        """Generate an answer grounded in the documents from ``matches``."""
        return self._generator.generate(
            query,
            [document for document, _confidence in matches],
        )

    def run(self, query: str, *, top_k: int = 4) -> str:
        """Run embedding, search, and generation for one query."""
        matches = self.retrieve(query, top_k=top_k)
        return self.generate(query, matches)

    def run_turn(
        self,
        query: str,
        *,
        history: Sequence[BaseMessage] = (),
        top_k: int = 4,
    ) -> ChatResponse:
        """Run one citation-aware turn using temporary conversation history."""
        matches = self.retrieve(query, top_k=top_k)
        documents, citations = _prepare_citations(matches)
        message = self._generator.generate_turn(
            query,
            documents,
            history=history,
        )
        return ChatResponse(message=message, citations=citations)
