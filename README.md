# RAG Chitta

A minimal RAG implementation with OpenAI embeddings and chat generation backed
by Milvus dense retrieval.

## Usage

Configuration is loaded from `.env.local`. The existing `OPENAI_API_KEY` is
used by the OpenAI clients; the application never reads or prints it directly.

```python
from langchain_core.documents import Document

from src.app import build_application

application = build_application()
application.index([Document(page_content="Your source text")])

question = "What does the source say?"
answer = application.query(question, top_k=4)

# Each stage is also available independently.
query_embedding = application.pipeline.embed_query(question)
matches = application.pipeline.search(query_embedding, top_k=4)
answer = application.pipeline.generate(question, matches)
```

The same flow is available from the command line:

```console
uv run python src/main.py --document-text "Your source text" "What does the source say?"
```

Retrieval confidence is Milvus cosine similarity normalized from `[-1, 1]` to
`[0, 1]`.
