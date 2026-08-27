# synthetic.py

## Purpose

Generates deterministic single-needle tasks with exact block provenance and a unique answer. Target identifiers and pressures are configurable. Distractors use the same sentence template and pressure vocabulary so the task tests identifier-specific influence, not obvious formatting differences.

## Components

### `SyntheticNeedleTask`
- **Does**: Carries raw and chat-ready system/query text, task-family provenance, target key, historical blocks, relevant block ID, seed, answer, and format-insensitive match phrase.

### `make_needle_task`
- **Does**: Shuffles one configurable target valve/pressure record among similarly worded records for other valves.
- **Rationale**: Varying target values tests whether a selector generalizes beyond one answer's tokenization, while same-template distractors challenge naive perturbation ranking.

### `contains_answer_value`
- **Does**: Scores free-form generations by the answer's numeric value, independent of surrounding prose or units.
- **Rationale**: Small-model formatting compliance is not part of the retrieval milestone.

### `contains_answer_text`
- **Does**: Case-insensitively matches a complete expected answer phrase while allowing arbitrary surrounding prose and whitespace.
- **Rationale**: Nonnumeric task families need semantic outcome scoring without exact response-format requirements.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `probing.py` | Exactly one `relevant_block_id` indexes `blocks`, with system/query messages available for chat serialization | Multiple relevant records or message schema changes |
| Diverse task factory | `task_family`, `target_key`, and `answer_match` describe every task without depending on numeric fields | Removing generic provenance |
| Experiment logging | Same seed and block count reproduce identical text | Changing RNG or construction order |
| Generation evaluation | Any standalone occurrence of the configured target value counts as correct | Requiring an exact response string |
