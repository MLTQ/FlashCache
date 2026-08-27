# test_synthetic.py

## Purpose

Checks that the synthetic needle generator is reproducible and exposes exactly one ground-truth record.

## Components

### `test_task_has_one_relevant_record_and_is_reproducible`
- **Does**: Verifies seed stability, block count, unique target identifier, answer provenance, and chat-ready message fields.

### `test_answer_scoring_ignores_format_but_not_other_numbers`
- **Does**: Accepts prose, capitalization, value-only numeric answers, and categorical names while rejecting different and containing numbers.

### `test_task_supports_a_custom_target_without_duplicate_answer_values`
- **Does**: Verifies custom identifiers and pressure values propagate through provenance, query, answer, and blocks without duplicating the answer in a distractor.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `synthetic.py` | One task contains exactly one configured target record and correctness is value-based | Task or scoring schema changes |
