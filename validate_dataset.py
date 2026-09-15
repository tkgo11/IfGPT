#!/usr/bin/env python3
"""Validate the separate authored conversation corpus without modifying it."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

DEFAULT_DATA_FILE = Path(__file__).with_name("conversations.jsonl")
PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
NON_SPACE_WHITESPACE_RE = re.compile(r"[^\S ]")
MULTI_SPACE_RE = re.compile(r" {2,}")
REQUIRED_FIELDS = {"id", "category", "user_message", "assistant_response"}

# Prompts that a built-in rule handles before corpus lookup are
# unreachable data; flag them. Skipped when chatbot.py is absent.
_RESERVED_PROMPTS: frozenset[str] = frozenset()
try:
    import chatbot as _chatbot
except ImportError:
    pass
else:
    _RESERVED_PROMPTS = frozenset(
        _chatbot.GREETINGS
        | _chatbot.FAREWELLS
        | _chatbot.THANKS
        | _chatbot.EXIT_COMMANDS
        | _chatbot.HELP_REQUESTS
        | _chatbot.IDENTITY_QUESTIONS
    )


def comparison_form_error(text: str) -> str | None:
    """Return a description if an authored prompt is not in comparison form."""
    if unicodedata.normalize("NFC", text) != text:
        return "is not NFC normalized"
    if text != text.casefold():
        return "is not lowercase/casefolded"
    if text != text.strip():
        return "has leading or trailing whitespace"
    if PUNCTUATION_RE.search(text):
        return "contains punctuation"
    if NON_SPACE_WHITESPACE_RE.search(text):
        return "contains non-space whitespace"
    if MULTI_SPACE_RE.search(text):
        return "contains repeated whitespace"
    return None


def validate(data_file: Path, expected_count: int = 2400) -> list[str]:
    """Return every schema, uniqueness, ID, and comparison-form error found."""
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 1
    ):
        raise ValueError("expected_count must be a positive integer")

    errors: list[str] = []
    seen_ids: dict[int, int] = {}
    seen_messages: dict[str, int] = {}
    record_count = 0

    try:
        source = data_file.open("r", encoding="utf-8")
    except OSError as error:
        return [f"Could not open '{data_file}': {error}"]

    try:
        with source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    errors.append(f"line {line_number}: blank JSONL record")
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append(f"line {line_number}: invalid JSON ({error.msg})")
                    continue
                except (RecursionError, ValueError) as error:
                    # Pathological JSON that is not a JSONDecodeError:
                    # nesting past the recursion limit or integer tokens
                    # beyond the digit cap.
                    errors.append(f"line {line_number}: invalid JSON ({error})")
                    continue

                record_count += 1
                if not isinstance(record, dict):
                    errors.append(f"line {line_number}: record is not an object")
                    continue
                if set(record) != REQUIRED_FIELDS:
                    errors.append(
                        f"line {line_number}: fields must be exactly "
                        f"{sorted(REQUIRED_FIELDS)}"
                    )
                    continue

                record_id = record["id"]
                category = record["category"]
                message = record["user_message"]
                response = record["assistant_response"]
                if (
                    not isinstance(record_id, int)
                    or isinstance(record_id, bool)
                    or record_id < 1
                ):
                    errors.append(f"line {line_number}: id must be a positive integer")
                elif record_id in seen_ids:
                    errors.append(
                        f"line {line_number}: duplicate id {record_id} "
                        f"(first seen at line {seen_ids[record_id]})"
                    )
                else:
                    seen_ids[record_id] = line_number

                if not isinstance(category, str) or not category.strip():
                    errors.append(
                        f"line {line_number}: category must be a non-empty string"
                    )
                if not isinstance(message, str) or not message:
                    errors.append(
                        f"line {line_number}: user_message must be a non-empty string"
                    )
                else:
                    form_error = comparison_form_error(message)
                    if form_error:
                        errors.append(f"line {line_number}: user_message {form_error}")
                    if message in _RESERVED_PROMPTS:
                        errors.append(
                            f"line {line_number}: user_message {message!r} is "
                            "shadowed by a built-in rule"
                        )
                    if message in seen_messages:
                        errors.append(
                            f"line {line_number}: duplicate user_message "
                            f"{message!r} (first seen at line "
                            f"{seen_messages[message]})"
                        )
                    else:
                        seen_messages[message] = line_number
                if not isinstance(response, str) or not response.strip():
                    errors.append(
                        f"line {line_number}: assistant_response must be a "
                        "non-empty string"
                    )
                else:
                    try:
                        response.encode("utf-8")
                    except UnicodeEncodeError:
                        errors.append(
                            f"line {line_number}: assistant_response is not "
                            "UTF-8 encodable"
                        )
    except UnicodeDecodeError as error:
        errors.append(f"'{data_file}' is not valid UTF-8: {error}")
    except OSError as error:
        errors.append(f"Error reading '{data_file}': {error}")
    except MemoryError:
        errors.append(f"'{data_file}' is too large to read")

    if record_count != expected_count:
        errors.append(f"record count is {record_count}; expected {expected_count}")
    if seen_ids:
        # Lazy range checks: never materialize or iterate range(1, N + 1)
        # when --expected-count is large. islice stops after the first 10
        # ascending missing ids; extras iterate the small seen set.
        missing_ids = list(
            itertools.islice(
                (i for i in range(1, expected_count + 1) if i not in seen_ids),
                10,
            )
        )
        extra_ids = sorted(i for i in seen_ids if not 1 <= i <= expected_count)[:10]
        if missing_ids:
            errors.append(f"missing ids: {missing_ids}")
        if extra_ids:
            errors.append(f"ids outside expected range: {extra_ids}")

    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the authored JSONL conversation corpus "
        "without modifying it."
    )
    parser.add_argument(
        "data_file",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="JSONL corpus path (default: conversations.jsonl next to this script)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=2400,
        metavar="N",
        help="expected record count and id range 1..N (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.expected_count < 1:
        parser.error("--expected-count must be a positive integer")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    errors = validate(args.data_file, expected_count=args.expected_count)
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} issue(s)")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"VALIDATION PASSED: {args.expected_count} "
        f"record{'s' if args.expected_count != 1 else ''}, unique ids and "
        "prompts, valid schema, valid comparison form."
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        # stdout is block-buffered on pipes; flush now so a downstream-
        # closed pipe raises here, inside the handler's reach.
        sys.stdout.flush()
    except BrokenPipeError:
        # stdout was closed early (e.g. piped into `head`). Redirect the
        # file descriptor so interpreter shutdown does not re-raise.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(0) from None
    raise SystemExit(exit_code)
