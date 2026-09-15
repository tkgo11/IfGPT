# Rule-Based JSONL Chatbot

This project is a standard-library Python chatbot with a **separate, directly authored JSON Lines corpus**. The current finalized corpus contains **2,400 unique conversations**, reflecting the requested rounding-down cutoff from the records completed so far.

> The chatbot normalizes **only live user input**. It never changes, normalizes, or rewrites the stored corpus at runtime.

| File | Purpose |
|---|---|
| `chatbot.py` | Interactive and one-shot rule-based chatbot with explicit `if`/`elif`/`else` response logic. |
| `conversations.jsonl` | Separate authored corpus containing 2,400 JSONL records. |
| `validate_dataset.py` | Streaming validator for schema, count, IDs, unique prompts, and comparison-form rules. |
| `test_chatbot.py` | Behavioral tests for normalization, exact lookup, rules, fallback, loader error paths, and CLI behavior. |
| `test_validate_dataset.py` | Tests for every validator error branch, the comparison-form rules, and the validator CLI. |
| `AUTHORING_STANDARD.md` | The direct-authorship and uniqueness standard used for the corpus. |

## Data Format

Each line in `conversations.jsonl` is one JSON object with exactly four fields.

```json
{"id": 1, "category": "daily_planning", "user_message": "example comparison form", "assistant_response": "Stored response text."}
```

The stored `user_message` field is authored directly in **comparison form**. It uses NFC Unicode, lowercase text, no punctuation, no leading or trailing whitespace, and single internal spaces. The chatbot uses each value directly as a dictionary key and does not modify the file.

| Runtime step | Behavior |
|---|---|
| Live-input normalization | NFC normalization, `casefold()`, trimming, punctuation removal using the regex `[^\w\s]`, and whitespace collapsing. |
| Stored-data handling | Exact loading only; no stored message is normalized at runtime. |
| Corpus lookup | Direct dictionary lookup from normalized live input to stored response. |
| Rule logic | Explicit `if`/`elif`/`else` handling for empty input, exit commands, greetings, thanks, farewells, help, identity questions, exact corpus lookup, and fallback. |

## Run the Chatbot

Run the chatbot interactively from the project directory (the scripts are also executable, so `./chatbot.py` works).

```bash
python3 chatbot.py
```

Type `quit`, `exit`, `bye`, `goodbye`, or `stop` to leave the interactive session.

To answer a single message and exit, pass it as an argument.

```bash
python3 chatbot.py "hello"
```

If the message itself starts with `-`, separate it with `--` (standard argument parsing).

```bash
python3 chatbot.py -- "-weird prompt"
```

## Validate and Test

Validate the corpus without changing it.

```bash
python3 validate_dataset.py
```

Pass a path to validate a different file, and `--expected-count N` when the corpus is deliberately extended or cut.

```bash
python3 validate_dataset.py path/to/corpus.jsonl --expected-count 2500
```

Run the behavioral test suites.

```bash
python3 -m unittest -v test_chatbot.py test_validate_dataset.py
```

The validator checks that the corpus has exactly the expected number of records (2,400 by default), all IDs are unique and cover the expected range, every prompt is unique, each record has the required schema, and every stored prompt complies with the comparison-form standard.

Lint and type checks run the same way locally and in CI (`.github/workflows/ci.yml`); install the pinned tools with `pip install "ruff==0.16.7" "mypy==2.3.1"` first.

```bash
ruff check . && ruff format --check .
python3 -m mypy
```

## Design Limits

This is intentionally a **rule-based exact-lookup chatbot**, not a generative model. It responds with the stored response only when the normalized live input exactly matches an authored `user_message`. A non-matching input receives a generic fallback response. The separate corpus can be extended in the future by directly authoring additional distinct JSONL records and updating the validator's expected count deliberately.
