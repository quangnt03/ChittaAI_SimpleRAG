"""DeepEval contract tests and opt-in end-to-end RAG quality evaluation."""

import os
from pathlib import Path
from uuid import uuid4

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from langchain_core.documents import Document

from evaluation import (
    BenchmarkCase,
    build_deepeval_test_case,
    build_rag_metrics,
    load_benchmark,
    run_pipeline_case,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "qa_reference_document.csv"
SOURCE_PATH = PROJECT_ROOT / "data" / "test_passages.txt"
BENCHMARK_CASES = load_benchmark(BENCHMARK_PATH)
RUN_RAG_EVALS = os.getenv("RUN_RAG_EVALS", "").casefold() in {"1", "true", "yes"}


def _environment_number(name: str, default: str, *, minimum: float) -> float:
    value = float(os.getenv(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def test_benchmark_has_ten_verbatim_reference_cases() -> None:
    source_text = SOURCE_PATH.read_text(encoding="utf-8")

    assert len(BENCHMARK_CASES) == 10
    assert len({case.question for case in BENCHMARK_CASES}) == 10
    assert all(case.reference in source_text for case in BENCHMARK_CASES)


def test_build_rag_metrics_covers_all_five_deepeval_rag_metrics() -> None:
    metrics = build_rag_metrics(
        threshold=0.7,
        model="gpt-4.1",
        include_reason=True,
        async_mode=False,
    )

    assert [type(metric) for metric in metrics] == [
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
    ]
    assert all(metric.threshold == 0.7 for metric in metrics)
    assert all(metric.include_reason is True for metric in metrics)


def test_build_deepeval_test_case_populates_every_rag_field() -> None:
    benchmark_case = BENCHMARK_CASES[0]
    test_case = build_deepeval_test_case(
        benchmark_case,
        actual_output="Cobalt Observatory uses the Beacon instrument. [1]",
        retrieval_context=[benchmark_case.reference, "Unrelated context."],
        retrieval_scores=[0.95, 0.1],
    )

    assert test_case.input == benchmark_case.question
    assert test_case.actual_output.endswith("[1]")
    assert test_case.expected_output == benchmark_case.answer
    assert test_case.context == [benchmark_case.reference]
    assert test_case.retrieval_context == [
        benchmark_case.reference,
        "Unrelated context.",
    ]
    assert test_case.metadata["benchmark_document"] == benchmark_case.document
    assert test_case.metadata["retrieval_scores"] == [0.95, 0.1]


class _StubPipeline:
    def embed_query(self, query: str) -> list[float]:
        assert query
        return [1.0]

    def search(
        self, query_embedding: list[float], *, top_k: int = 4
    ) -> list[tuple[Document, float]]:
        assert query_embedding == [1.0]
        assert top_k == 2
        return [(Document(page_content=BENCHMARK_CASES[0].reference), 0.97)]

    def retrieve(
        self, query: str, *, top_k: int = 4
    ) -> list[tuple[Document, float]]:
        return self.search(self.embed_query(query), top_k=top_k)

    def generate(
        self, query: str, matches: list[tuple[Document, float]]
    ) -> str:
        assert query
        assert matches
        return BENCHMARK_CASES[0].answer


def test_run_pipeline_case_preserves_actual_ranked_context() -> None:
    test_case = run_pipeline_case(
        _StubPipeline(),  # type: ignore[arg-type]
        BENCHMARK_CASES[0],
        top_k=2,
    )

    assert test_case.actual_output == BENCHMARK_CASES[0].answer
    assert test_case.retrieval_context == [BENCHMARK_CASES[0].reference]
    assert test_case.metadata["retrieval_scores"] == [0.97]


@pytest.fixture(scope="session")
def live_rag_application():
    if not RUN_RAG_EVALS:
        pytest.skip("set RUN_RAG_EVALS=1 to run live OpenAI and Milvus evaluations")

    from app import build_application
    from utils import TextFileLoader

    application = build_application()
    document_id = f"deepeval-benchmark-{uuid4()}"
    documents = TextFileLoader(
        SOURCE_PATH,
        metadata={"document_id": document_id, "dataset": "deepeval-benchmark"},
    ).load()
    application.index(documents)
    try:
        yield application
    finally:
        application.vector_store.delete_document(document_id)


@pytest.mark.rag_eval
@pytest.mark.skipif(not RUN_RAG_EVALS, reason="live RAG evaluations are opt-in")
@pytest.mark.parametrize(
    "benchmark_case",
    BENCHMARK_CASES,
    ids=lambda case: case.identifier,
)
def test_rag_quality_with_all_five_deepeval_metrics(
    live_rag_application,
    benchmark_case: BenchmarkCase,
) -> None:
    top_k = int(_environment_number("RAG_EVAL_TOP_K", "4", minimum=1))
    threshold = _environment_number("RAG_EVAL_THRESHOLD", "0.7", minimum=0)
    if threshold > 1:
        raise ValueError("RAG_EVAL_THRESHOLD must be at most 1")
    judge_model = os.getenv("RAG_EVAL_MODEL", "gpt-4.1").strip() or None

    test_case = run_pipeline_case(
        live_rag_application.pipeline,
        benchmark_case,
        top_k=top_k,
    )
    assert_test(
        test_case=test_case,
        metrics=build_rag_metrics(threshold=threshold, model=judge_model),
        run_async=True,
    )
