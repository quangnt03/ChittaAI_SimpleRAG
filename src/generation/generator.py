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
        return self.generate_turn(query, context)

    def generate_turn(
        self,
        query: str,
        context: Sequence[Document],
        *,
        history: Sequence[BaseMessage] = (),
    ) -> str:
        """Answer one turn using retrieved context and temporary history."""
        context_text = "\n\n".join(
            (
                f"[{document.metadata.get('citation_number', position)}] "
                f"Source: {document.metadata.get('citation_source', 'document')}\n"
                f"{document.page_content}"
            )
            for position, document in enumerate(context, start=1)
        ) or "No related documents were retrieved."
        return self.chat(
            [
                SystemMessage(
                    content=(
                        "Answer using only the retrieved context in the latest "
                        "user message. Conversation history may clarify what the "
                        "user means, but it is not evidence. Cite every supported "
                        "claim with the matching bracketed source number, such as "
                        "[1]. If the context is insufficient, say so."
                    )
                ),
                *history,
                HumanMessage(
                    content=(
                        f"Retrieved context:\n{context_text}\n\n"
                        f"Current question: {query}"
                    )
                ),
            ]
        )

    def chat(self, messages: Sequence[BaseMessage]) -> str:
        """Send a chat history to OpenAI and return its text response."""
        return self._client.invoke(list(messages)).text
