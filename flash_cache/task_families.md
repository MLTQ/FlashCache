# task_families.py

## Purpose

Builds controlled single-needle tasks across numeric and categorical answer spaces. All history and literary material is deliberately fictional so model pretraining cannot supply the answer without the flashed block.

## Components

### `TASK_FAMILIES`
- **Does**: Declares the supported `valve_pressure`, `history_person`, `book_quote`, and `history_place` families.

### `make_experiment_task`
- **Does**: Dispatches to the numeric valve generator or one of three categorical task generators.
- **Rationale**: A common task interface lets the same cache preparation, flashing, influence metrics, and correctness evaluation run unchanged across answer types.

### `_assemble_categorical_task`
- **Does**: Selects one target by variant, shuffles same-template distractors by seed, and emits exact provenance plus a format-insensitive answer match.
- **Rationale**: Separating target choice from block placement distinguishes task-content effects from shuffle effects.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment runner | Every family returns one `SyntheticNeedleTask` with exactly one relevant block | Multiple relevant records or a different task schema |
| Correctness evaluation | Categorical answers appear in exactly one source block | Reusing the target answer in distractors |
| Reproducibility | `family`, `variant`, `seed`, and `block_count` fully determine a task | Nondeterministic record selection |

## Notes

- Fictional archives test in-context retrieval rather than factual recall.
- Names and places use multi-token answers; book questions require matching a quoted line rather than a numeric identifier.
