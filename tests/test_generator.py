"""Tests for citation-aware OpenAI message construction."""

import unittest
from unittest.mock import Mock

from generation import Generator
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class GeneratorTests(unittest.TestCase):
    def test_generate_turn_includes_history_context_and_citation_instructions(
        self,
    ) -> None:
        client = Mock()
        client.invoke.return_value = AIMessage(content="Grounded answer [1]")
        generator = Generator(client)
        history = [
            HumanMessage(content="Earlier question"),
            AIMessage(content="Earlier answer"),
        ]
        context = [
            Document(
                page_content="Supporting passage",
                metadata={"citation_number": 1, "citation_source": "source.txt"},
            )
        ]

        answer = generator.generate_turn(
            "Follow-up question",
            context,
            history=history,
        )

        self.assertEqual(answer, "Grounded answer [1]")
        messages = client.invoke.call_args.args[0]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIn("Cite every supported claim", messages[0].content)
        self.assertEqual(messages[1:3], history)
        self.assertIn("[1] Source: source.txt", messages[-1].content)
        self.assertIn("Supporting passage", messages[-1].content)
        self.assertIn("Follow-up question", messages[-1].content)


if __name__ == "__main__":
    unittest.main()
