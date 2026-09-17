"""Shared JSONL fixture builders for the test suites."""

from __future__ import annotations

import json
from pathlib import Path


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


def write_jsonl(directory: Path, records: list[object]) -> Path:
    data_file = directory / "data.jsonl"
    data_file.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return data_file
