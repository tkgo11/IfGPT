#!/usr/bin/env python3
"""Behavioral tests for chatbot.py using the separate corpus."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import chatbot


class ChatbotTests(unittest.TestCase):
    conversations: ClassVar[dict[str, str]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.conversations = chatbot.load_conversations()

    def test_normalize_live_input_only(self) -> None:
        self.assertEqual(
            chatbot.normalize_live_input("  How can I decide?!  "),
            "how can i decide",
        )
        self.assertEqual(chatbot.normalize_live_input("CAFÉ"), "café")

    def test_exact_lookup_with_case_space_and_punctuation_variants(self) -> None:
        stored_prompt = (
            "how can i decide whether to ask for more time before agreeing to something"
        )
        expected = self.conversations[stored_prompt]
        live_variant = (
            "  How can I decide whether to ask for more time "
            "before agreeing to something?  "
        )
        self.assertEqual(chatbot.respond(live_variant, self.conversations), expected)
        # Stored data is not altered by the runtime lookup.
        self.assertIn(stored_prompt, self.conversations)

    def test_basic_if_elif_responses(self) -> None:
        self.assertEqual(
            chatbot.respond("   ", self.conversations),
            "Please enter a message so I can respond.",
        )
        self.assertEqual(
            chatbot.respond("hello!", self.conversations),
            "Hello. Ask me a question or describe something you would like help with.",
        )
        self.assertEqual(
            chatbot.respond("Thanks", self.conversations), "You are welcome."
        )
        self.assertEqual(
            chatbot.respond("who are you?", self.conversations),
            "I am a small rule-based chatbot using a separate JSONL conversation file.",
        )
        self.assertEqual(chatbot.respond("quit", self.conversations), "Goodbye.")
        self.assertTrue(chatbot.is_exit_command("QUIT!"))

    def test_generic_fallback(self) -> None:
        response = chatbot.respond(
            "please compose a sonnet about an orbital telescope", self.conversations
        )
        self.assertIn("do not have an exact stored response", response)

    def test_malformed_dataset_is_rejected(self) -> None:
        malformed_record = {"id": 1, "category": "test", "user_message": "sample"}
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "bad.jsonl"
            data_file.write_text(json.dumps(malformed_record) + "\n", encoding="utf-8")
            with self.assertRaises(chatbot.DatasetError):
                chatbot.load_conversations(data_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
