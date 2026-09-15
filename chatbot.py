#!/usr/bin/env python3
"""A small rule-based chatbot backed by a separate JSONL conversation corpus.

Only live user input is normalized.  Stored ``user_message`` values are loaded
and compared exactly as they appear in ``conversations.jsonl``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path

DATA_FILE = Path(__file__).with_name("conversations.jsonl")
PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

GREETINGS = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
FAREWELLS = {"goodbye", "bye", "see you", "see you later", "farewell"}
THANKS = {"thanks", "thank you", "thanks a lot", "thank you very much"}
EXIT_COMMANDS = {"quit", "exit", "bye", "goodbye", "stop"}
HELP_REQUESTS = {
    "help",
    "what can you do",
    "how do you work",
    "how can you help me",
}
IDENTITY_QUESTIONS = {"who are you", "what are you", "what is this"}


class DatasetError(ValueError):
    """Raised when the separate conversation data cannot be loaded safely."""


def normalize_live_input(text: str) -> str:
    """Return the comparison form for a live input string only.

    The JSONL corpus is deliberately not normalized or rewritten by this
    function.  It is assumed to contain direct-authored comparison-form
    ``user_message`` values.
    """
    if not isinstance(text, str):
        raise TypeError("message must be a string")

    normalized = unicodedata.normalize("NFC", text).casefold().strip()
    normalized = PUNCTUATION_RE.sub("", normalized)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def load_conversations(data_file: Path | str = DATA_FILE) -> dict[str, str]:
    """Load exact authored prompt-to-response mappings from a JSONL file."""
    path = Path(data_file)
    conversations: dict[str, str] = {}

    try:
        source = path.open("r", encoding="utf-8")
    except OSError as error:
        raise DatasetError(f"Could not open data file '{path}': {error}") from error

    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise DatasetError(f"Blank record at line {line_number}.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetError(
                    f"Invalid JSON at line {line_number}: {error.msg}"
                ) from error

            required_fields = {"id", "category", "user_message", "assistant_response"}
            if not isinstance(record, dict) or not required_fields.issubset(record):
                raise DatasetError(f"Invalid record schema at line {line_number}.")
            if not isinstance(record["user_message"], str) or not isinstance(
                record["assistant_response"], str
            ):
                raise DatasetError(
                    f"Non-string conversation text at line {line_number}."
                )
            if record["user_message"] in conversations:
                raise DatasetError(
                    f"Duplicate user_message at line {line_number}: "
                    f"{record['user_message']!r}"
                )

            # Important: do not normalize, alter, or rewrite stored data here.
            conversations[record["user_message"]] = record["assistant_response"]

    return conversations


def is_exit_command(message: str) -> bool:
    """Return whether a normalized message is an explicit exit command."""
    return normalize_live_input(message) in EXIT_COMMANDS


def respond(message: str, conversations: Mapping[str, str] | None = None) -> str:
    """Return a chatbot response using explicit if/elif/else logic."""
    if conversations is None:
        conversations = {}

    normalized = normalize_live_input(message)

    if not normalized:
        return "Please enter a message so I can respond."
    elif normalized in EXIT_COMMANDS:
        return "Goodbye."
    elif normalized in GREETINGS:
        return (
            "Hello. Ask me a question or describe something you would like help with."
        )
    elif normalized in THANKS:
        return "You are welcome."
    elif normalized in FAREWELLS:
        return "Goodbye."
    elif normalized in HELP_REQUESTS:
        return (
            "I can answer the exact conversation prompts in my local data file and "
            "handle basic greetings, thanks, help, and exit commands."
        )
    elif normalized in IDENTITY_QUESTIONS:
        return (
            "I am a small rule-based chatbot using a separate JSONL conversation file."
        )
    elif normalized in conversations:
        # Lookup is exact against the authored, untouched corpus key.
        return conversations[normalized]
    else:
        return (
            "I do not have an exact stored response for that message. "
            "Try rephrasing it as a more specific everyday question."
        )


def main() -> None:
    """Run the interactive command-line chatbot."""
    try:
        conversations = load_conversations()
    except DatasetError as error:
        print(f"Data warning: {error}")
        print("The chatbot will continue with basic rule-based responses only.")
        conversations = {}

    print("Rule-based chatbot ready. Type 'quit' to exit.")
    while True:
        try:
            message = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        print(f"Bot: {respond(message, conversations)}")
        if is_exit_command(message):
            break


if __name__ == "__main__":
    main()
