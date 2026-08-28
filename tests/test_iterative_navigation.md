# `test_iterative_navigation.py`

## Purpose

Protects the answer-free navigation action language, lenient small-model parsing, prompt validation, and archive-preserving query rewrites.

## Coverage

- Labeled lookup/answer actions parse correctly.
- Bare questions become lookups, bare values become answers, and empty output is invalid.
- A mislabeled affirmative final fact stops only when its subject is grounded in the current carrier; ungrounded or negative facts continue.
- Navigation prompts require both a question and retrieved pages.
- Query rewriting leaves pinned and cold archive text unchanged.
- A one-edit fictional-name drift is corrected from selected source text, while unrelated source values cause no change.
- A bounded repair prompt includes the repeated output and explicitly requires a different action.
- The fixed instruction preserves who/where/what targets and demonstrates treaty and quotation rewrites.
- A quoted intermediate answer to a who-question triggers one answer-free target-type repair, while person and what-answers do not.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Iterative experiment | Parser never receives an expected answer | Answer-aware fallback logic |
| Cold KV reuse | Rewritten tasks retain identical blocks/pinned context | Rebuilding or modifying cold sources |
