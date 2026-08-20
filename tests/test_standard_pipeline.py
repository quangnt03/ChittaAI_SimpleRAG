"""Unit tests for the standard RAG pipeline."""

import unittest
from collections.abc import Sequence

from generation.base import BaseGenerator
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pipelines import StandardPipeline
from retrieval.base import BaseRetriever


class StubRetriever(BaseRetriever):
    """Deterministic retriever used to verify pipeline orchestration."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.searches: list[tuple[list[float], int]] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [1.0, 2.0]

    def search(
        self, query_embedding: Sequence[float], *, top_k: int = 4
    ) -> list[tuple[Document, float]]:
        self.searches.append((list(query_embedding), top_k))
        return [(Document(page_content="grounded context"), 0.9)]


class StubGenerator(BaseGenerator):
    """Deterministic generator used to capture generated context."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Document]]] = []

    def generate(self, query: str, context: Sequence[Document]) -> str:
        self.calls.append((query, list(context)))
        return "grounded answer"

    def chat(self, messages: Sequence[BaseMessage]) -> str:
        return "chat answer"


class StandardPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = StubRetriever()
        self.generator = StubGenerator()
        self.pipeline = StandardPipeline(self.retriever, self.generator)

    def test_individual_stages_delegate_to_components(self) -> None:
        embedding = self.pipeline.embed_query("question")
        matches = self.pipeline.search(embedding, top_k=2)
        answer = self.pipeline.generate("question", matches)

        self.assertEqual(embedding, [1.0, 2.0])
        self.assertEqual(self.retriever.searches, [([1.0, 2.0], 2)])
        self.assertEqual(answer, "grounded answer")
        self.assertEqual(
            [document.page_content for document in self.generator.calls[0][1]],
            ["grounded context"],
        )

    def test_run_composes_embedding_search_and_generation(self) -> None:
        answer = self.pipeline.run("question", top_k=3)

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(self.retriever.queries, ["question"])
        self.assertEqual(self.retriever.searches, [([1.0, 2.0], 3)])
        self.assertEqual(self.generator.calls[0][0], "question")

    def test_retrieve_remains_a_compatible_convenience_method(self) -> None:
        matches = self.retriever.retrieve("question", top_k=1)

        self.assertEqual(self.retriever.queries, ["question"])
        self.assertEqual(self.retriever.searches, [([1.0, 2.0], 1)])
        self.assertEqual(matches[0][0].page_content, "grounded context")


if __name__ == "__main__":
    unittest.main()
