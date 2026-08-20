# RAG Chitta

A minimal RAG implementation with OpenAI embeddings and chat generation backed
by Milvus dense retrieval.

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

Retrieval confidence is Milvus cosine similarity normalized from `[-1, 1]` to
`[0, 1]`.
