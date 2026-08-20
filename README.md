# RAG Chitta

A minimal RAG implementation with OpenAI embeddings and chat generation backed
by Milvus-native hybrid retrieval.

Each chunk is indexed in one Milvus collection through three complementary
indexes/features:

- HNSW approximate nearest-neighbor search with cosine similarity over dense
  OpenAI embeddings.
- Milvus's server-side BM25 function and sparse inverted index over analyzed
  chunk text.
- Reciprocal rank fusion (RRF) over an expanded candidate pool from both paths.

The collection also stores metadata as JSON, enables analyzed text matching,
and has an inverted scalar index on `document_id` for filtered deletion.

## Project structure

```text
RAG_Chitta/
├── benchmarks/              # Labelled RAG evaluation dataset
│   └── qa_reference_document.csv
├── data/                    # Source documents ingested by the application
│   └── test_passages.txt
├── src/
│   ├── app/                 # Application assembly and chat CLI
│   ├── configs/             # Environment-backed configuration
│   ├── evaluation/          # DeepEval test-case and metric helpers
│   ├── generation/          # Grounded response generation
│   ├── index/               # Document chunking and indexing
│   ├── pipelines/           # End-to-end RAG orchestration and citations
│   ├── retrieval/           # Dense passage retrieval
│   ├── utils/               # Document loading utilities
│   └── vectorstore/         # Milvus vector-store integration
├── tests/                   # Unit tests and opt-in live RAG evaluations
├── .env.example             # Environment variable template
├── docker-compose.yml       # Local Milvus services
├── main.py                  # Development CLI entry point
├── pyproject.toml           # Project metadata, dependencies, and commands
└── uv.lock                  # Reproducible dependency lockfile
```

## Chat CLI

Configuration is loaded from `.env.local`. The existing `OPENAI_API_KEY` is
used by the OpenAI clients; 

Documents will live in and be ingested from `data` directory. Supported document types include: `.txt`.
Please locate all documents in `data` before start the application

Start Milvus, then open the interactive CLI:

```console
docker compose up -d
uv run app
```

`uv run rag-chitta` is an equivalent descriptive alias.

Each process creates exactly one temporary in-memory chat session. Follow-up
messages see earlier turns from that run, and the history is discarded on
`/exit`, `/quit`, Ctrl+C, or EOF. Every response is followed by the related
retrieved document citations, relevance scores, and quoted source excerpts.

Optional text can be indexed before the session starts:

```console
uv run app --document-text "Chitta uses Milvus for retrieval."
```

Pass a positional message for one-turn, non-interactive use:

```console
uv run app "What does the source say?"
```

`uv run python main.py` is also available as a development entry point.

## Python API

```python
from langchain_core.documents import Document

from src.app import build_application

application = build_application()
application.index(
    [
        Document(
            page_content="Your source text",
            metadata={"source": "example.txt"},
        )
    ]
)

session = application.new_chat_session(top_k=4)
response = session.send("What does the source say?")
print(response.message)
print(response.citations)
```

Normal application queries use hybrid retrieval. `StandardPipeline.search()`
remains available as a dense-only stage for callers that already have a query
embedding, while `StandardPipeline.retrieve()` runs the complete dense + BM25 +
RRF flow. Dense confidence is cosine similarity normalized from `[-1, 1]` to
`[0, 1]`. Hybrid confidence is the Milvus RRF score normalized against the
theoretical best score across both retrieval paths; the raw score is retained
in document metadata as `milvus_rrf_score`.

### Milvus index lifecycle

The first indexed batch determines the dense-vector dimension. The index
builder then creates the explicit schema, BM25 function, HNSW index, sparse
BM25 index, scalar index, and loads the collection. Later batches validate that
their vectors keep the same dimension.

BM25 is a collection-creation feature. If `MILVUS_COLLECTION` already points to
the old dense-only schema, the application raises a migration error and does
not delete data. Preserve any needed data, then either choose a new collection
name or explicitly rebuild the old collection before re-indexing documents.

The `.env.example` file exposes HNSW construction/search parameters, BM25
parameters, sparse search pruning, the per-path candidate multiplier, RRF `k`,
and Milvus consistency level. The defaults favor correctness for immediate
index-then-query CLI use (`Session` consistency and no sparse pruning).

## Evaluation

The benchmark in `benchmarks/qa_reference_document.csv` contains ten labelled cases with four fields: the user question, the expected answer, a verbatim supporting
reference, and the source document passage. The reference validates the label; the evaluation itself uses the chunks retrieved at test time.

### Results so far

Status as of 2026-08-20:

| Check | Result | Notes |
| --- | --- | --- |
| Complete local test suite | **Passed** — 30 passed, 11 skipped | The skipped cases are the opt-in live Milvus integration and ten live DeepEval cases. |
| Live Milvus hybrid integration | **Passed** — 1 passed | Verified real collection creation, HNSW/COSINE ANN, server-side BM25 sparse generation, RRF fusion, result parsing, and cleanup against `milvusdb/milvus:v3.0.0`. |
| Live ten-case DeepEval suite | **Blocked before scoring** | `rag_documents` still has the previous dense-only schema. No quality metric scores have been produced yet. |

The DeepEval attempt stopped with `MilvusCollectionSchemaError` because the old
`rag_documents` collection does not contain the new schema-defined
`dense_vector`, `sparse_vector`, analyzed `text`, `document_id`, and `metadata`
fields or the BM25 function. This is an index migration prerequisite, not a
retrieval-quality failure. The implementation deliberately preserves the old
collection instead of deleting indexed data automatically.

Before the next live evaluation, either preserve and explicitly rebuild the
old collection or select a fresh collection name. The non-destructive option
is:

```powershell
$env:MILVUS_COLLECTION = "rag_documents_hybrid"
```

The evaluation fixture will create the hybrid collection during indexing. Once
the ten cases complete, record the five per-case metric scores and aggregate
pass/fail counts here.

### Metric scorecard

Every case is converted to one DeepEval `LLMTestCase` and evaluated with the
complete single-turn RAG suite:

| Component | Metric | Test-case evidence | What a low score means |
| --- | --- | --- | --- |
| Generator | Answer Relevancy | `input`, `actual_output` | The answer does not address the question. |
| Generator | Faithfulness | `actual_output`, `retrieval_context` | The answer contains claims unsupported by retrieved chunks. |
| Retriever | Contextual Relevancy | `input`, `retrieval_context` | Retrieved chunks contain too much unrelated material. |
| Retriever | Contextual Precision | `input`, `expected_output`, ranked `retrieval_context` | Relevant chunks are ranked below irrelevant chunks. |
| Retriever | Contextual Recall | `expected_output`, `retrieval_context` | Retrieval omitted information needed for the ideal answer. |

The pipeline is executed once per row. Its generated answer becomes
`actual_output`, the CSV answer becomes `expected_output`, and its ranked search
results become `retrieval_context`. This prevents the evaluator from scoring a
different set of chunks than the generator actually received.

### Gate and diagnosis

The initial pass threshold is `0.7` for every metric. A benchmark row fails if
any metric is below threshold, which prevents a good average from hiding one
bad question or one broken pipeline component. Keep the judge model, top-K,
chunking configuration, dataset, and threshold fixed when comparing runs.

### How to Run

Fast contract tests do not call OpenAI or Milvus:

```console
uv run pytest tests/test_rag_deepeval.py -m "not rag_eval"
```

Run the standalone Milvus integration check without calling OpenAI:

```powershell
docker compose up -d
$env:RUN_MILVUS_INTEGRATION = "1"
uv run pytest tests/test_milvus_hybrid_integration.py
```

The live test starts from the benchmark source, indexes it with a unique
temporary document ID, evaluates all ten cases with all five metrics, and
deletes those indexed chunks afterward. Start Milvus, then run in PowerShell:

```powershell
docker compose up -d
$env:MILVUS_COLLECTION = "rag_documents_hybrid"
$env:RUN_RAG_EVALS = "1"
$env:RAG_EVAL_MODEL = "gpt-4.1"
uv run deepeval test run tests/test_rag_deepeval.py
```

Optional controls:

- `RAG_EVAL_THRESHOLD` (default `0.7`)
- `RAG_EVAL_TOP_K` (default `4`)
- `RAG_EVAL_MODEL` (default `gpt-4.1`)

The live suite invokes the RAG generator and LLM judges, so it is intentionally
excluded from ordinary unit-test runs.

### DeepEval references

- [RAG evaluation quickstart](https://deepeval.com/docs/getting-started-rag)
- [Answer Relevancy](https://deepeval.com/docs/metrics-answer-relevancy)
- [Faithfulness](https://deepeval.com/docs/metrics-faithfulness)
- [Contextual Relevancy](https://deepeval.com/docs/metrics-contextual-relevancy)
- [Contextual Precision](https://deepeval.com/docs/metrics-contextual-precision)
- [Contextual Recall](https://deepeval.com/docs/metrics-contextual-recall)
