"""Unit tests for application assembly and the command-line entry point."""

import unittest
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import ChatSession, RAGApplication, build_application, main
from pipelines import ChatResponse, Citation


class ChatSessionTests(unittest.TestCase):
    def test_session_keeps_completed_turns_only_in_memory(self) -> None:
        pipeline = Mock()
        pipeline.run_turn.side_effect = [
            ChatResponse(message="first answer"),
            ChatResponse(message="second answer"),
        ]
        session = ChatSession(pipeline, top_k=2)

        session.send("first question")
        session.send("follow-up question")

        first_call, second_call = pipeline.run_turn.call_args_list
        self.assertEqual(first_call.kwargs["history"], ())
        self.assertEqual(first_call.kwargs["top_k"], 2)
        second_history = second_call.kwargs["history"]
        self.assertEqual(len(second_history), 2)
        self.assertIsInstance(second_history[0], HumanMessage)
        self.assertIsInstance(second_history[1], AIMessage)
        self.assertEqual(second_history[0].content, "first question")
        self.assertEqual(second_history[1].content, "first answer")
        self.assertEqual(len(session.messages), 4)

    def test_session_rejects_empty_messages(self) -> None:
        session = ChatSession(Mock())

        with self.assertRaises(ValueError):
            session.send("  ")


class RAGApplicationTests(unittest.TestCase):
    def test_application_delegates_indexing_and_queries(self) -> None:
        indexer = Mock()
        indexer.index.return_value = ["chunk-id"]
        pipeline = Mock()
        pipeline.run.return_value = "answer"
        application = RAGApplication(
            indexer=indexer,
            pipeline=pipeline,
            vector_store=Mock(),
        )
        documents = [Document(page_content="source")]

        self.assertEqual(application.index(documents), ["chunk-id"])
        self.assertEqual(application.query("question", top_k=2), "answer")
        indexer.index.assert_called_once_with(documents)
        pipeline.run.assert_called_once_with("question", top_k=2)

    @patch("app.cli.StandardPipeline")
    @patch("app.cli.Generator")
    @patch("app.cli.Retriever")
    @patch("app.cli.Indexer")
    @patch("app.cli.MilvusVectorStore")
    @patch("app.cli.ChatOpenAI")
    @patch("app.cli.OpenAIEmbeddings")
    def test_build_application_shares_dependencies(
        self,
        embedding_type: Mock,
        chat_type: Mock,
        vector_store_type: Mock,
        indexer_type: Mock,
        retriever_type: Mock,
        generator_type: Mock,
        pipeline_type: Mock,
    ) -> None:
        settings = SimpleNamespace(
            openai_embedding_model="embedding-model",
            openai_chat_model="chat-model",
            openai_api_key=Mock(),
            milvus_collection="documents",
            milvus_url="http://milvus",
            milvus_port=19530,
            chunk_size=500,
            chunk_overlap=50,
        )

        application = build_application(settings)  # type: ignore[arg-type]

        embedding_type.assert_called_once_with(
            model="embedding-model",
            api_key=settings.openai_api_key,
        )
        vector_store_type.assert_called_once_with("documents")
        vector_store_type.return_value.connect.assert_called_once_with(
            "http://milvus",
            19530,
        )
        indexer_type.assert_called_once_with(
            embedding_type.return_value,
            vector_store_type.return_value,
            chunk_size=500,
            chunk_overlap=50,
        )
        retriever_type.assert_called_once_with(
            embedding_type.return_value,
            vector_store_type.return_value,
        )
        chat_type.assert_called_once_with(
            model="chat-model",
            api_key=settings.openai_api_key,
        )
        generator_type.assert_called_once_with(chat_type.return_value)
        pipeline_type.assert_called_once_with(
            retriever_type.return_value,
            generator_type.return_value,
        )
        self.assertIs(application.indexer, indexer_type.return_value)
        self.assertIs(application.pipeline, pipeline_type.return_value)
        self.assertIs(application.vector_store, vector_store_type.return_value)

    @patch("app.build_application")
    def test_main_indexes_documents_and_renders_cited_answer(
        self,
        build_application_mock: Mock,
    ) -> None:
        application = build_application_mock.return_value
        session = application.new_chat_session.return_value
        session.send.return_value = ChatResponse(
            message="grounded answer [1]",
            citations=(
                Citation(
                    number=1,
                    source="first.txt",
                    confidence=0.9,
                    quote="Quoted source passage.",
                ),
            ),
        )
        output = StringIO()

        exit_code = main(
            [
                "--document-text",
                "first source",
                "--document-text",
                "second source",
                "--top-k",
                "2",
                "question",
            ],
            output=output,
        )

        self.assertEqual(exit_code, 0)
        rendered_output = output.getvalue()
        self.assertIn("┌─ You", rendered_output)
        self.assertIn("question", rendered_output)
        self.assertIn("grounded answer [1]", rendered_output)
        self.assertIn("[1] first.txt (relevance 90%)", rendered_output)
        self.assertIn("“Quoted source passage.”", rendered_output)
        application.index.assert_called_once()
        indexed_documents = application.index.call_args.args[0]
        self.assertEqual(
            [document.page_content for document in indexed_documents],
            ["first source", "second source"],
        )
        self.assertEqual(
            [document.metadata["source"] for document in indexed_documents],
            ["command-line document 1", "command-line document 2"],
        )
        application.new_chat_session.assert_called_once_with(top_k=2)
        session.send.assert_called_once_with("question")

    @patch("app.build_application")
    def test_main_runs_messages_in_interactive_order(
        self,
        build_application_mock: Mock,
    ) -> None:
        session = build_application_mock.return_value.new_chat_session.return_value
        session.send.side_effect = [
            ChatResponse(message="first answer"),
            ChatResponse(message="second answer"),
        ]
        user_messages = iter(["first question", "follow-up", "/exit"])
        output = StringIO()

        exit_code = main(
            [],
            input_func=lambda _prompt: next(user_messages),
            output=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [call.args[0] for call in session.send.call_args_list],
            ["first question", "follow-up"],
        )
        rendered_output = output.getvalue()
        self.assertLess(
            rendered_output.index("first answer"),
            rendered_output.index("follow-up"),
        )
        self.assertIn("history is cleared", rendered_output)
        self.assertTrue(rendered_output.rstrip().endswith("Session ended."))


if __name__ == "__main__":
    unittest.main()
