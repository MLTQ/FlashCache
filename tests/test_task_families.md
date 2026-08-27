# test_task_families.py

## Purpose

Verifies that every controlled answer-space family is reproducible, has exact provenance, and supports format-insensitive categorical correctness scoring.

## Components

### `test_each_family_is_reproducible_with_one_answer_bearing_block`
- **Does**: Checks all declared families for deterministic construction, exact block count, query provenance, and one uniquely answer-bearing block.

### `test_categorical_answer_matching_ignores_format_and_case`
- **Does**: Accepts capitalization, bold markup, and whitespace variation while rejecting a longer name that merely contains the target name.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `task_families.py` | Every declared family produces one uniquely scorable relevant block | Duplicate answers or missing query keys |
| `synthetic.py` | Complete answer phrases are matched case-insensitively with word boundaries | Substring-only correctness |
