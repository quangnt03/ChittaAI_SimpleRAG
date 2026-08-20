"""Build DeepEval test cases from the RAG Chitta benchmark and pipeline."""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from langchain_core.documents import Document
from pipelines import SearchResult, StandardPipeline


BENCHMARK_COLUMNS = ("question", "answer", "reference", "document")
DEFAULT_RAG_THRESHOLD = 0.7


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One labelled RAG question with its answer and supporting evidence."""

    question: str
    answer: str
    reference: str
    document: str

    @property
    def identifier(self) -> str:
        """Return a readable pytest identifier for this benchmark row."""
        return self.document.partition("#")[2] or self.document


def load_benchmark(path: str | Path) -> tuple[BenchmarkCase, ...]:
    """Load and validate the four-column benchmark CSV."""
    benchmark_path = Path(path)
    with benchmark_path.open(newline="", encoding="utf-8") as benchmark_file:
        reader = csv.DictReader(benchmark_file)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != BENCHMARK_COLUMNS:
            raise ValueError(
                f"benchmark columns must be {BENCHMARK_COLUMNS}, got {fieldnames}"
            )

        cases: list[BenchmarkCase] = []
        for row_number, row in enumerate(reader, start=2):
            values = {column: (row.get(column) or "").strip() for column in fieldnames}
            missing = [column for column, value in values.items() if not value]
            if missing:
                raise ValueError(
                    f"benchmark row {row_number} has empty fields: {', '.join(missing)}"
                )
            cases.append(
                BenchmarkCase(
                    question=values["question"],
                    answer=values["answer"],
                    reference=values["reference"],
                    document=values["document"],
                )
            )

    if not cases:
        raise ValueError("benchmark must contain at least one case")
    if len({case.question for case in cases}) != len(cases):
        raise ValueError("benchmark questions must be unique")
    return tuple(cases)


def build_rag_metrics(
    *,
    threshold: float = DEFAULT_RAG_THRESHOLD,
    model: str | None = None,
    include_reason: bool = True,
    async_mode: bool = True,
) -> list[
    AnswerRelevancyMetric
    | FaithfulnessMetric
    | ContextualRelevancyMetric
    | ContextualPrecisionMetric
    | ContextualRecallMetric
]:
    """Create DeepEval's complete five-metric single-turn RAG suite."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    options: dict[str, object] = {
        "threshold": threshold,
        "include_reason": include_reason,
        "async_mode": async_mode,
    }
    if model is not None and model.strip():
        options["model"] = model.strip()

    return [
        AnswerRelevancyMetric(**options),
        FaithfulnessMetric(**options),
        ContextualRelevancyMetric(**options),
        ContextualPrecisionMetric(**options),
        ContextualRecallMetric(**options),
    ]


def build_deepeval_test_case(
    benchmark_case: BenchmarkCase,
    *,
    actual_output: str,
    retrieval_context: Sequence[str],
    retrieval_scores: Sequence[float] = (),
) -> LLMTestCase:
    """Map one pipeline result to the fields required by all five metrics."""
    actual_output = actual_output.strip()
    contexts = [context.strip() for context in retrieval_context if context.strip()]
    if not actual_output:
        raise ValueError("actual_output cannot be empty")
    if not contexts:
        raise ValueError("retrieval_context cannot be empty")

    return LLMTestCase(
        name=benchmark_case.identifier,
        input=benchmark_case.question,
        actual_output=actual_output,
        expected_output=benchmark_case.answer,
        context=[benchmark_case.reference],
        retrieval_context=contexts,
        metadata={
            "benchmark_document": benchmark_case.document,
            "benchmark_reference": benchmark_case.reference,
            "retrieval_scores": list(retrieval_scores),
        },
    )


def run_pipeline_case(
    pipeline: StandardPipeline,
    benchmark_case: BenchmarkCase,
    *,
    top_k: int = 4,
) -> LLMTestCase:
    """Run retrieval and generation once, preserving ranked chunks for DeepEval."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    matches: list[SearchResult] = pipeline.retrieve(
        benchmark_case.question,
        top_k=top_k,
    )
    actual_output = pipeline.generate(benchmark_case.question, matches)
    documents: list[Document] = [document for document, _score in matches]
    return build_deepeval_test_case(
        benchmark_case,
        actual_output=actual_output,
        retrieval_context=[document.page_content for document in documents],
        retrieval_scores=[score for _document, score in matches],
    )
