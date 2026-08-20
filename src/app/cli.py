
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TextIO
import argparse
import shutil
import sys
import textwrap

from configs import Settings, get_settings
from generation import Generator
from index import Indexer
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pipelines import ChatResponse, StandardPipeline
from retrieval import Retriever
from vectorstore import MilvusIndexConfig, MilvusSearchConfig, MilvusVectorStore


class ChatSession:
    """Keep one process-local RAG conversation in memory."""

    def __init__(self, pipeline: StandardPipeline, *, top_k: int = 4) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._pipeline = pipeline
        self._top_k = top_k
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> tuple[BaseMessage, ...]:
        """Return an immutable view of the current run's chat history."""
        return tuple(self._messages)

    def send(self, message: str) -> ChatResponse:
        """Send one user message and remember the completed turn in memory."""
        message = message.strip()
        if not message:
            raise ValueError("message cannot be empty")

        response = self._pipeline.run_turn(
            message,
            history=self.messages,
            top_k=self._top_k,
        )
        self._messages.extend(
            [
                HumanMessage(content=message),
                AIMessage(content=response.message),
            ]
        )
        return response


@dataclass(frozen=True, slots=True)
class RAGApplication:
    """Expose document indexing and querying through shared RAG components."""

    indexer: Indexer
    pipeline: StandardPipeline
    vector_store: MilvusVectorStore

    def index(self, documents: Sequence[Document]) -> list[str]:
        """Split, embed, and persist ``documents`` in the configured index."""
        return self.indexer.index(documents)

    def query(self, query: str, *, top_k: int = 4) -> str:
        """Retrieve relevant chunks and generate a grounded answer."""
        return self.pipeline.run(query, top_k=top_k)

    def new_chat_session(self, *, top_k: int = 4) -> ChatSession:
        """Create a temporary session that is discarded when the process exits."""
        return ChatSession(self.pipeline, top_k=top_k)


def build_application(settings: Settings | None = None) -> RAGApplication:
    """Construct the standard RAG application from typed settings."""
    resolved_settings = settings if settings is not None else get_settings()

    embedding_model = OpenAIEmbeddings(
        model=resolved_settings.openai_embedding_model,
        api_key=resolved_settings.openai_api_key,
    )

    vector_store = MilvusVectorStore(
        resolved_settings.milvus_collection,
        index_config=MilvusIndexConfig(
            hnsw_m=resolved_settings.milvus_hnsw_m,
            hnsw_ef_construction=resolved_settings.milvus_hnsw_ef_construction,
            bm25_k1=resolved_settings.milvus_bm25_k1,
            bm25_b=resolved_settings.milvus_bm25_b,
        ),
        search_config=MilvusSearchConfig(
            hnsw_ef=resolved_settings.milvus_hnsw_ef,
            sparse_drop_ratio=resolved_settings.milvus_sparse_drop_ratio,
            candidate_multiplier=resolved_settings.milvus_candidate_multiplier,
            rrf_k=resolved_settings.milvus_rrf_k,
            consistency_level=resolved_settings.milvus_consistency_level,
        ),
    )
    vector_store.connect(
        resolved_settings.milvus_url,
        resolved_settings.milvus_port,
    )

    indexer = Indexer(
        embedding_model,
        vector_store,
        chunk_size=resolved_settings.chunk_size,
        chunk_overlap=resolved_settings.chunk_overlap,
    )
    retriever = Retriever(embedding_model, vector_store)
    generator = Generator(
        ChatOpenAI(
            model=resolved_settings.openai_chat_model,
            api_key=resolved_settings.openai_api_key,
        )
    )
    pipeline = StandardPipeline(retriever, generator)

    return RAGApplication(
        indexer=indexer,
        pipeline=pipeline,
        vector_store=vector_store,
    )


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open a temporary RAG chat session, or pass one message for "
            "non-interactive use."
        ),
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Optional one-turn message; omit it to open interactive chat.",
    )
    parser.add_argument(
        "--document-text",
        action="append",
        default=[],
        help="Source text to index before querying; may be supplied more than once.",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=4,
        help="Number of chunks to retrieve (default: 4).",
    )
    return parser


def _message_box(title: str, message: str, *, width: int) -> str:
    """Render a compact terminal-safe message box."""
    width = max(32, width)
    content_width = width - 4
    prefix = f"┌─ {title} "
    top = prefix + "─" * max(0, width - len(prefix) - 1) + "┐"
    bottom = "└" + "─" * (width - 2) + "┘"
    lines: list[str] = []
    for logical_line in message.splitlines() or [""]:
        lines.extend(
            textwrap.wrap(logical_line, width=content_width) or [""]
        )
    body = [f"│ {line:<{content_width}} │" for line in lines]
    return "\n".join([top, *body, bottom])


def _render_user_message(message: str, *, output: TextIO) -> None:
    terminal_width = min(shutil.get_terminal_size(fallback=(88, 24)).columns, 100)
    output.write(f"\n{_message_box('You', message, width=terminal_width)}\n")
    output.flush()


def _render_response(response: ChatResponse, *, output: TextIO) -> None:
    output.write("\nAssistant\n─────────\n")
    output.write(response.message.rstrip() + "\n")
    output.write("\nSources\n")
    if response.citations:
        for citation in response.citations:
            output.write(
                f"[{citation.number}] {citation.source} "
                f"(relevance {citation.confidence:.0%})\n"
            )
            if citation.quote:
                output.write(f"    “{citation.quote}”\n")
    else:
        output.write("No related documents retrieved.\n")
    output.flush()


def run_chat_cli(
    session: ChatSession,
    *,
    input_func: Callable[[str], str] | None = None,
    output: TextIO | None = None,
) -> int:
    """Run an interactive, process-local chat loop."""
    input_func = input if input_func is None else input_func
    output = sys.stdout if output is None else output
    output.write(
        "Temporary chat session — history is cleared when this process exits.\n"
        "Type /exit or /quit to finish.\n"
    )
    output.flush()

    while True:
        try:
            message = input_func("\nYou › ").strip()
        except (EOFError, KeyboardInterrupt):
            output.write("\nSession ended.\n")
            return 0

        if message.casefold() in {"/exit", "/quit"}:
            output.write("Session ended.\n")
            return 0
        if not message:
            continue

        _render_user_message(message, output=output)
        response = session.send(message)
        _render_response(response, output=output)
