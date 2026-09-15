#!/usr/bin/env python3
"""Tests for validate_dataset.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unicodedata
import unittest
from pathlib import Path

import chatbot
import validate_dataset as vd
from fixtures import record, write_jsonl

VALIDATOR = Path(__file__).resolve().with_name("validate_dataset.py")


class ComparisonFormTests(unittest.TestCase):
    def test_valid_form(self) -> None:
        self.assertIsNone(vd.comparison_form_error("a clean lower case prompt"))

    def test_each_violation_detected(self) -> None:
        cases = {
            "not nfc": (unicodedata.normalize("NFD", "café"), "NFC"),
            "uppercase": ("Has Capitals", "lowercase"),
            "edge whitespace": (" padded ", "leading or trailing"),
            "punctuation": ("has punct!", "punctuation"),
            "tab whitespace": ("a\tb", "non-space whitespace"),
            "repeated whitespace": ("double  space", "repeated whitespace"),
        }
        for name, (text, expected) in cases.items():
            with self.subTest(name=name):
                self.assertIn(expected, vd.comparison_form_error(text) or "")

    def test_agrees_with_chatbot_normalize_on_corpus(self) -> None:
        # comparison_form_error(m) is None iff normalize(m) == m; both must
        # hold for every stored message (H3).
        for message in chatbot.load_conversations():
            self.assertIsNone(vd.comparison_form_error(message))
            self.assertEqual(chatbot.normalize_live_input(message), message)


class ValidateTests(unittest.TestCase):
    def validate_records(
        self, records: list[object], expected_count: int | None = None
    ) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            data_file = write_jsonl(Path(directory), records)
            if expected_count is None:
                return vd.validate(data_file)
            return vd.validate(data_file, expected_count=expected_count)

    def test_real_corpus_is_clean(self) -> None:
        self.assertEqual(vd.validate(vd.DEFAULT_DATA_FILE), [])

    def test_small_valid_file(self) -> None:
        errors = self.validate_records(
            [record(1, user_message="one"), record(2, user_message="two")],
            expected_count=2,
        )
        self.assertEqual(errors, [])

    def test_blank_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "data.jsonl"
            data_file.write_text(json.dumps(record()) + "\n\n", encoding="utf-8")
            errors = vd.validate(data_file, expected_count=1)
        self.assertTrue(any("blank" in e for e in errors))

    def test_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "data.jsonl"
            data_file.write_text("{oops}\n", encoding="utf-8")
            errors = vd.validate(data_file, expected_count=1)
        self.assertTrue(any("invalid JSON" in e for e in errors))

    def test_non_object_record(self) -> None:
        errors = self.validate_records([["not", "an", "object"]], expected_count=1)
        self.assertTrue(any("not an object" in e for e in errors))

    def test_missing_and_extra_fields(self) -> None:
        missing = self.validate_records([{"id": 1}], expected_count=1)
        extra = self.validate_records([{**record(), "x": 1}], expected_count=1)
        self.assertTrue(any("fields must be exactly" in e for e in missing))
        self.assertTrue(any("fields must be exactly" in e for e in extra))

    def test_bad_ids(self) -> None:
        for bad_id in (0, -1, True, "1", 1.5):
            with self.subTest(bad_id=bad_id):
                errors = self.validate_records([record(bad_id)], expected_count=1)
                self.assertTrue(any("positive integer" in e for e in errors))

    def test_duplicate_id(self) -> None:
        errors = self.validate_records(
            [record(1, user_message="a"), record(1, user_message="b")],
            expected_count=2,
        )
        self.assertTrue(any("duplicate id 1" in e for e in errors))

    def test_empty_category(self) -> None:
        errors = self.validate_records([record(category="  ")], expected_count=1)
        self.assertTrue(any("category" in e for e in errors))

    def test_empty_message(self) -> None:
        errors = self.validate_records([record(user_message="")], expected_count=1)
        self.assertTrue(any("user_message" in e for e in errors))

    def test_duplicate_message(self) -> None:
        errors = self.validate_records([record(1), record(2)], expected_count=2)
        self.assertTrue(any("duplicate user_message" in e for e in errors))

    def test_empty_response(self) -> None:
        errors = self.validate_records(
            [record(assistant_response="  ")], expected_count=1
        )
        self.assertTrue(any("assistant_response" in e for e in errors))

    def test_count_mismatch(self) -> None:
        errors = self.validate_records([record(1)], expected_count=5)
        self.assertTrue(any("record count is 1" in e for e in errors))
        self.assertTrue(any("missing ids" in e for e in errors))

    def test_out_of_range_ids_reported_when_count_differs(self) -> None:
        # Regression for the len(seen_ids)==expected_count gate: extra ids
        # must be reported even when the record count already mismatches.
        errors = self.validate_records(
            [record(1), record(2), record(99)], expected_count=2
        )
        self.assertTrue(any("record count is 3" in e for e in errors))
        self.assertTrue(any("outside expected range" in e for e in errors))

    def test_huge_expected_count_does_not_hang(self) -> None:
        # The id-range check must not materialize set(range(1, N + 1)).
        result: list[list[str]] = []
        worker = threading.Thread(
            target=lambda: result.append(
                self.validate_records([record(1)], expected_count=10**9)
            ),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive(), "validate() hung on huge count")
        errors = result[0]
        self.assertTrue(any("record count is 1" in e for e in errors))
        self.assertTrue(any("missing ids" in e for e in errors))

    def test_expected_count_param_validated(self) -> None:
        for bad in (0, -3, 2.5, "5", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.validate_records([record(1)], expected_count=bad)  # type: ignore[arg-type]

    def test_unopenable_file(self) -> None:
        errors = vd.validate(Path("/nonexistent/missing.jsonl"))
        self.assertEqual(len(errors), 1)
        self.assertIn("Could not open", errors[0])

    def test_non_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "data.jsonl"
            data_file.write_bytes(b'{"id":1}\n\xff\xfe\n')
            errors = vd.validate(data_file, expected_count=1)
        self.assertTrue(any("not valid UTF-8" in e for e in errors))

    def test_deeply_nested_json_reported_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "data.jsonl"
            data_file.write_text("[" * 2000 + "]" * 2000 + "\n", encoding="utf-8")
            errors = vd.validate(data_file, expected_count=1)
        self.assertTrue(any("invalid JSON" in e for e in errors))

    def test_oversized_int_token_reported_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "data.jsonl"
            data_file.write_text("9" * 5000 + "\n", encoding="utf-8")
            errors = vd.validate(data_file, expected_count=1)
        self.assertTrue(any("invalid JSON" in e for e in errors))

    def test_rule_shadowed_prompt_flagged(self) -> None:
        errors = self.validate_records(
            [record(1, user_message="hello")], expected_count=1
        )
        self.assertTrue(any("shadowed by a built-in rule" in e for e in errors))

    def test_surrogate_response_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "data.jsonl"
            data_file.write_text(
                '{"id":1,"category":"t","user_message":"hi",'
                '"assistant_response":"\\ud800 x"}\n',
                encoding="utf-8",
            )
            errors = vd.validate(data_file, expected_count=1)
        self.assertTrue(any("not UTF-8 encodable" in e for e in errors))


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(VALIDATOR), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_default_run_passes(self) -> None:
        result = self.run_cli()
        self.assertEqual(result.returncode, 0)
        self.assertIn("VALIDATION PASSED: 2400 records", result.stdout)

    def test_expected_count_flag(self) -> None:
        result = self.run_cli("--expected-count", "2400")
        self.assertEqual(result.returncode, 0)
        result = self.run_cli("--expected-count", "5")
        self.assertEqual(result.returncode, 1)
        self.assertIn("record count is 2400; expected 5", result.stdout)

    def test_invalid_expected_count(self) -> None:
        result = self.run_cli("--expected-count", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive integer", result.stderr)

    def test_help_flag(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout)

    def test_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "small.jsonl"
            data_file.write_text(json.dumps(record(1)) + "\n", encoding="utf-8")
            result = self.run_cli(str(data_file), "--expected-count", "1")
        self.assertEqual(result.returncode, 0)
        self.assertIn("VALIDATION PASSED: 1 record,", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
