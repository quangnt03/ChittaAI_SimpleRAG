"""Unit tests for application assembly and the command-line entry point."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import RAGApplication, build_application, main


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

    @patch("main.StandardPipeline")
    @patch("main.Generator")
    @patch("main.Retriever")
    @patch("main.Indexer")
    @patch("main.MilvusVectorStore")
    @patch("main.ChatOpenAI")
    @patch("main.OpenAIEmbeddings")
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

    @patch("main.build_application")
    def test_main_indexes_documents_and_prints_answer(
        self,
        build_application_mock: Mock,
    ) -> None:
        application = build_application_mock.return_value
        application.query.return_value = "grounded answer"
        output = StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "--document-text",
                    "first source",
                    "--document-text",
                    "second source",
                    "--top-k",
                    "2",
                    "question",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().strip(), "grounded answer")
        application.index.assert_called_once()
        indexed_documents = application.index.call_args.args[0]
        self.assertEqual(
            [document.page_content for document in indexed_documents],
            ["first source", "second source"],
        )
        application.query.assert_called_once_with("question", top_k=2)


if __name__ == "__main__":
    unittest.main()
