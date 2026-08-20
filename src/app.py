"""Application assembly and command-line entry point for RAG Chitta."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from configs import Settings, get_settings
from generation import Generator
from index import Indexer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pipelines import StandardPipeline
from retrieval import Retriever
from vectorstore import MilvusVectorStore


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


def build_application(settings: Settings | None = None) -> RAGApplication:
    """Construct the standard RAG application from typed settings."""
    resolved_settings = settings if settings is not None else get_settings()

    embedding_model = OpenAIEmbeddings(
        model=resolved_settings.openai_embedding_model,
        api_key=resolved_settings.openai_api_key,
    )

    vector_store = MilvusVectorStore(resolved_settings.milvus_collection)
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
        description="Index optional document text and query the RAG pipeline.",
    )
    parser.add_argument("query", help="Question to answer from indexed context.")
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line indexing and query workflow."""
    arguments = _build_parser().parse_args(argv)
    application = build_application()

    if arguments.document_text:
        application.index(
            [Document(page_content=text) for text in arguments.document_text]
        )

    print(application.query(arguments.query, top_k=arguments.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
