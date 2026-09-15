#!/usr/bin/env python3
"""Behavioral tests for chatbot.py using the separate corpus."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path
from typing import ClassVar

import chatbot

CHATBOT = Path(__file__).resolve().with_name("chatbot.py")


def run_cli(
    *args: str,
    stdin: str = "",
    cwd: Path | None = None,
    script: Path = CHATBOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=30,
    )


def write_jsonl(
    directory: Path, records: list[object], name: str = "data.jsonl"
) -> Path:
    data_file = directory / name
    data_file.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return data_file


def record(
    record_id: object = 1,
    category: object = "test",
    user_message: object = "sample prompt",
    assistant_response: object = "sample response",
) -> dict[str, object]:
    return {
        "id": record_id,
        "category": category,
        "user_message": user_message,
        "assistant_response": assistant_response,
    }


class NormalizeTests(unittest.TestCase):
    def test_case_punctuation_and_whitespace(self) -> None:
        self.assertEqual(
            chatbot.normalize_live_input("  How can I decide?!  "),
            "how can i decide",
        )
        self.assertEqual(chatbot.normalize_live_input("a\tb\nc  d"), "a b c d")
        self.assertEqual(chatbot.normalize_live_input("!!!"), "")

    def test_unicode_nfc_and_casefold(self) -> None:
        self.assertEqual(chatbot.normalize_live_input("CAFÉ"), "café")
        self.assertEqual(
            chatbot.normalize_live_input(unicodedata.normalize("NFD", "café")),
            "café",
        )
        self.assertEqual(
            chatbot.normalize_live_input("ＣＡＦＥ"),  # noqa: RUF001
            "ｃａｆｅ",  # noqa: RUF001
        )

    def test_non_string_input_rejected(self) -> None:
        for bad in (None, 5, ["hello"], b"bytes"):
            with self.assertRaises(TypeError):
                chatbot.normalize_live_input(bad)  # type: ignore[arg-type]

    def test_idempotent_on_comparison_form(self) -> None:
        once = chatbot.normalize_live_input("Some   WEIRD  Input!?")
        self.assertEqual(chatbot.normalize_live_input(once), once)


class LoadConversationsTests(unittest.TestCase):
    def test_loads_exact_keys_without_normalizing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = write_jsonl(
                Path(directory),
                [record(user_message="Stored PROMPT with punct!?")],
            )
            conversations = chatbot.load_conversations(data_file)
        self.assertEqual(
            conversations, {"Stored PROMPT with punct!?": "sample response"}
        )

    def test_missing_file(self) -> None:
        with self.assertRaises(chatbot.DatasetError):
            chatbot.load_conversations("/nonexistent/definitely/missing.jsonl")

    def test_blank_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "data.jsonl"
            data_file.write_text(json.dumps(record()) + "\n\n", encoding="utf-8")
            with self.assertRaises(chatbot.DatasetError):
                chatbot.load_conversations(data_file)

    def test_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "data.jsonl"
            data_file.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(chatbot.DatasetError):
                chatbot.load_conversations(data_file)

    def test_missing_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = write_jsonl(
                Path(directory), [{"id": 1, "category": "test", "user_message": "x"}]
            )
            with self.assertRaises(chatbot.DatasetError):
                chatbot.load_conversations(data_file)

    def test_extra_field_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = write_jsonl(Path(directory), [{**record(), "extra": "ignored"}])
            self.assertEqual(
                chatbot.load_conversations(data_file),
                {"sample prompt": "sample response"},
            )

    def test_non_string_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = write_jsonl(Path(directory), [record(assistant_response=123)])
            with self.assertRaises(chatbot.DatasetError):
                chatbot.load_conversations(data_file)

    def test_duplicate_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = write_jsonl(
                Path(directory),
                [record(1), record(2, category="other")],
            )
            with self.assertRaises(chatbot.DatasetError):
                chatbot.load_conversations(data_file)

    def test_non_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "data.jsonl"
            data_file.write_bytes(b'{"id":1}\n\xff\xfe\n')
            with self.assertRaises(chatbot.DatasetError):
                chatbot.load_conversations(data_file)


class RespondTests(unittest.TestCase):
    conversations: ClassVar[dict[str, str]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.conversations = chatbot.load_conversations()

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

    def test_empty_and_whitespace_input(self) -> None:
        for empty in ("", "   ", "!!!", "\t\n"):
            self.assertEqual(
                chatbot.respond(empty, self.conversations),
                "Please enter a message so I can respond.",
            )

    def test_exit_commands(self) -> None:
        for command in ("quit", "exit", "bye", "goodbye", "stop", "QUIT!"):
            self.assertEqual(chatbot.respond(command, self.conversations), "Goodbye.")
            self.assertTrue(chatbot.is_exit_command(command))
        self.assertFalse(chatbot.is_exit_command("hello"))

    def test_greetings_thanks_farewells(self) -> None:
        self.assertEqual(
            chatbot.respond("hello!", self.conversations),
            "Hello. Ask me a question or describe something you would like help with.",
        )
        self.assertEqual(
            chatbot.respond("Thanks", self.conversations), "You are welcome."
        )
        for farewell in ("see you", "see you later", "farewell"):
            self.assertEqual(chatbot.respond(farewell, self.conversations), "Goodbye.")

    def test_help_and_identity(self) -> None:
        self.assertIn(
            "exact conversation prompts",
            chatbot.respond("what can you do?", self.conversations),
        )
        self.assertEqual(
            chatbot.respond("who are you?", self.conversations),
            "I am a small rule-based chatbot using a separate JSONL conversation file.",
        )

    def test_generic_fallback(self) -> None:
        response = chatbot.respond(
            "please compose a sonnet about an orbital telescope", self.conversations
        )
        self.assertIn("do not have an exact stored response", response)

    def test_default_conversations_empty(self) -> None:
        self.assertEqual(
            chatbot.respond("hello"),
            "Hello. Ask me a question or describe something you would like help with.",
        )
        self.assertIn("do not have an exact stored response", chatbot.respond("xyzzy"))

    def test_rules_take_precedence_over_corpus(self) -> None:
        conversations = {"quit": "stored response", "hello": "stored hello"}
        self.assertEqual(chatbot.respond("quit", conversations), "Goodbye.")
        self.assertNotEqual(chatbot.respond("hello", conversations), "stored hello")

    def test_nfd_input_matches_nfc_stored_key(self) -> None:
        nfd_key = unicodedata.normalize("NFD", "café plan")
        conversations = {"café plan": "stored"}
        self.assertEqual(chatbot.respond(nfd_key, conversations), "stored")

    def test_malformed_dataset_is_rejected(self) -> None:
        malformed_record = {"id": 1, "category": "test", "user_message": "sample"}
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "bad.jsonl"
            data_file.write_text(json.dumps(malformed_record) + "\n", encoding="utf-8")
            with self.assertRaises(chatbot.DatasetError):
                chatbot.load_conversations(data_file)


class CliTests(unittest.TestCase):
    def test_oneshot_returns_response_only(self) -> None:
        result = run_cli("hello")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            result.stdout,
            "Hello. Ask me a question or describe something you would "
            "like help with.\n",
        )
        self.assertEqual(result.stderr, "")

    def test_oneshot_corpus_hit(self) -> None:
        result = run_cli(
            "how can i decide whether to ask for more time before agreeing to something"
        )
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.endswith("\n"))
        self.assertGreater(len(result.stdout.strip()), 50)
        self.assertNotIn("You:", result.stdout)

    def test_oneshot_exit_command(self) -> None:
        result = run_cli("quit")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "Goodbye.\n")

    def test_help_flag(self) -> None:
        result = run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout)

    def test_repl_transcript(self) -> None:
        result = run_cli(stdin="hello\nbye\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            result.stdout,
            "Rule-based chatbot ready. Type 'quit' to exit.\n"
            "You: Bot: Hello. Ask me a question or describe something "
            "you would like help with.\n"
            "You: Bot: Goodbye.\n",
        )

    def test_repl_eof_exits(self) -> None:
        result = run_cli(stdin="")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Goodbye.", result.stdout)

    def test_missing_corpus_warns_on_stderr_not_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "chatbot.py"
            script.write_text(CHATBOT.read_text(encoding="utf-8"), encoding="utf-8")
            result = run_cli(stdin="quit\n", script=script)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Data warning:", result.stderr)
        self.assertNotIn("Data warning:", result.stdout)
        self.assertIn("Bot: Goodbye.", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
