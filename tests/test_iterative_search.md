# test_iterative_search.py

## Purpose

Checks that the sentinel-search protocol retains the real question and evidence without leaking the ground-truth answer.

## Components

### `test_sentinel_search_prompt_preserves_question_without_answer_leakage`
- **Does**: Verifies question inclusion, answer exclusion, explicit period protocol, and unchanged candidate blocks.

### `test_gate_ignores_formatting_but_stops_at_first_sentinel_or_content`
- **Does**: Verifies control/formatting tokens are skipped, explicit negative-readiness prose and tokenizer-merged period surfaces remain misses, and visible answer content marks a hit.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `iterative_search.py` | Search prompts are answer-free and use the exact original question | Answer leakage or evidence mutation |
