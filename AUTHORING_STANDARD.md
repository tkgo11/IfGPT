# Conversation Authoring and Quality Standard

## Purpose

This project’s corpus must contain **100,000 independently authored conversational exchanges**. It must not be inflated with lightly edited copies, fixed-template substitutions, cosmetic punctuation changes, name/location swaps, or synonym swaps.

## Authoring unit

A record is one complete exchange with a user request and an assistant reply. Each record must address a different practical intent, situation, constraint, goal, trade-off, explanation, or interactional purpose. A record may share a broad domain with another record, but it must not restate the same underlying request with superficial changes.

## Prohibited practices

The following are prohibited from the final corpus:

| Prohibited pattern | Example |
|---|---|
| Greeting variants treated as unique | “Hello”, “Hi”, “Hello!” |
| Parameter substitution | Reusing “How do I pack for a trip to X?” by changing `X` |
| Mechanical tone or synonym changes | “Please help me…” versus “Could you help me…” for the same request |
| Punctuation/case changes | “Can you help?” versus “can you help” |
| Repeated fact patterns | Multiple records with the same task, conditions, and requested outcome |
| Empty or content-free exchanges | Generic “Tell me something” prompts without distinct purpose |

## Direct authoring approach

Every final record is authored directly by Manus for this project. No external language model, downloaded conversation corpus, template renderer, synonym swapper, parameter substitution process, or paraphrase generator may write records into the final JSONL file.

Each exchange must be conceived as a standalone real-life situation and written as a complete user request with a useful self-contained response. The audit process records authoring batches, content hashes, rejection reasons, and acceptance counts. It writes only vetted records to the final JSONL file.

## Required record schema

```json
{"id": 1, "category": "planning", "user_message": "i have three deadlines tomorrow and only two focused hours this evening how should i decide what to do first", "assistant_response": "Start by listing each deadline, the consequence of delay, and the smallest useful next action. Work first on the item with the highest impact or least flexibility, then time-box the remaining tasks so you make visible progress on each."}
```

`user_message` and `assistant_response` are immutable authored fields. **The final data file has no derived lookup key.** To enable comparison while altering only live user input, the authoring standard requires the stored `user_message` to be written originally in the documented comparison form: NFC Unicode, lowercase, single spaces, and no terminal punctuation. This is the conversation’s authored form in the data file, not a post-processing change.

## Quality gates

A record is accepted only after it passes these controls:

1. It conforms to the JSON schema and has useful, safe English text.
2. Its normalized user-message key has not appeared previously.
3. Its whole-record hash has not appeared previously.
4. It does not cross a lexical-similarity threshold against accepted records within the same category.
5. It does not duplicate an earlier record’s concrete task and expected answer structure according to direct editorial review and systematic lexical-similarity screening.
6. It does not contain unsupported claims of live browsing, personal experience, professional certification, or urgent-risk advice.

## Input-only normalization contract

At runtime, the chatbot performs normalization only on the newly received user message. It applies Unicode NFC normalization, `casefold()`, terminal-punctuation simplification, trimming, and whitespace collapsing. It compares that value directly to the stored authored `user_message`; it never changes the stored authored message or reply.

A paraphrase that is not an exact normalized input match is intentionally routed to the simple `if`/`elif`/`else` fallback responder.

## Dataset claim

The final README will accurately describe the corpus as **original conversational content authored directly by Manus for this project and filtered for exact, lexical, and editorial semantic-duplication checks**. It will not claim that the corpus was copied from observed conversations or that semantic uniqueness can be mathematically proved for every possible interpretation of natural language.
