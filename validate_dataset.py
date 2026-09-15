#!/usr/bin/env python3
"""Validate the separate authored conversation corpus without modifying it."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

DEFAULT_DATA_FILE = Path(__file__).with_name("conversations.jsonl")
PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
MULTI_WHITESPACE_RE = re.compile(r"\s{2,}")
REQUIRED_FIELDS = {"id", "category", "user_message", "assistant_response"}


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
    if MULTI_WHITESPACE_RE.search(text):
        return "contains repeated whitespace"
    return None


def validate(data_file: Path, expected_count: int = 2400) -> list[str]:
    """Return every schema, uniqueness, ID, and comparison-form error found."""
    errors: list[str] = []
    seen_ids: dict[int, int] = {}
    seen_messages: dict[str, int] = {}
    record_count = 0

    try:
        source = data_file.open("r", encoding="utf-8")
    except OSError as error:
        return [f"Could not open '{data_file}': {error}"]

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
                if message in seen_messages:
                    errors.append(
                        f"line {line_number}: duplicate user_message {message!r} "
                        f"(first seen at line {seen_messages[message]})"
                    )
                else:
                    seen_messages[message] = line_number
            if not isinstance(response, str) or not response.strip():
                errors.append(
                    f"line {line_number}: assistant_response must be a non-empty string"
                )

    if record_count != expected_count:
        errors.append(f"record count is {record_count}; expected {expected_count}")
    if len(seen_ids) == expected_count:
        expected_ids = set(range(1, expected_count + 1))
        missing_ids = sorted(expected_ids - set(seen_ids))
        extra_ids = sorted(set(seen_ids) - expected_ids)
        if missing_ids:
            errors.append(f"missing ids: {missing_ids[:10]}")
        if extra_ids:
            errors.append(f"ids outside expected range: {extra_ids[:10]}")

    return errors


def main() -> int:
    data_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_FILE
    errors = validate(data_file)
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} issue(s)")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "VALIDATION PASSED: 2400 records, unique ids and prompts, "
        "valid schema, valid comparison form."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
