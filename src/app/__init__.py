"""Application assembly and command-line entry point for RAG Chitta."""
from typing import Sequence, Callable, TextIO
from langchain_core.documents import Document
import sys

from .cli import (
    ChatSession,
    RAGApplication,
    _build_parser,
    _render_user_message,
    _render_response,
    build_application,
    run_chat_cli,
)

def main(
    argv: Sequence[str] | None = None,
    *,
    input_func: Callable[[str], str] | None = None,
    output: TextIO | None = None,
) -> int:
    """Run a one-turn query or an interactive temporary chat session."""
    output = sys.stdout if output is None else output
    arguments = _build_parser().parse_args(argv)
    application = build_application()

    if arguments.document_text:
        application.index(
            [
                Document(
                    page_content=text,
                    metadata={"source": f"command-line document {position}"},
                )
                for position, text in enumerate(arguments.document_text, start=1)
            ]
        )

    session = application.new_chat_session(top_k=arguments.top_k)
    if arguments.query is not None:
        _render_user_message(arguments.query, output=output)
        response = session.send(arguments.query)
        _render_response(response, output=output)
        return 0

    return run_chat_cli(session, input_func=input_func, output=output)


if __name__ == "__main__":
    raise SystemExit(main())
