"""OpenAI chat-model wrapper for RAG query and chat modes."""

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .base import BaseGenerator


class Generator(BaseGenerator):
    """Wrap an OpenAI chat model for grounded queries and direct chat."""

    def __init__(self, client: ChatOpenAI) -> None:
        self._client = client

    def generate(self, query: str, context: Sequence[Document]) -> str:
        """Answer one query using only the supplied retrieved context."""
        context_text = "\n\n".join(
            f"[{position}] {document.page_content}"
            for position, document in enumerate(context, start=1)
        )
        return self.chat(
            [
                SystemMessage(
                    content=(
                        "Answer using only the supplied context. "
                        "If the context is insufficient, say so."
                    )
                ),
                HumanMessage(
                    content=f"Context:\n{context_text}\n\nQuestion: {query}"
                ),
            ]
        )

    def chat(self, messages: Sequence[BaseMessage]) -> str:
        """Send a chat history to OpenAI and return its text response."""
        return self._client.invoke(list(messages)).text
