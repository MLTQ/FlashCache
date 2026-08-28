# multi_hop_tasks.py

## Purpose

Builds deterministic calibration and multi-hop questions requiring one to four separately cached facts. The streaming algorithm is not told which blocks are relevant or how many relations the answer requires.

## Components

### `MultiHopNeedleTask`
- **Does**: Extends the ordinary task schema with every relevant physical block ID and the evaluation-only hop depth.
- **Interacts with**: `tokenize_task` in `probing.py`, which consumes the inherited text fields without special handling.

### `make_multi_hop_task`
- **Does**: Constructs a direct preference calibration or a wife-to-person relationship chain ending in a favorite food, mixes it with same-domain distractors, and shuffles all pages by seed.
- **Rationale**: No individual page contains both the query path and answer. Success requires information to survive across page flashes.

### `_extended_distractors`

- **Does**: Preserves the original 16-note distractor pool for existing trials and deterministically appends unique relationship/preference filler notes only when a scale test requests more.
- **Rationale**: Retrieval latency and memory behavior need 32–128 page archives without changing the established 12-page task distribution.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Carrier-stream runner | `relevant_block_ids` is in logical chain order while `blocks` is physical shuffled order | Reordering the provenance tuple |
| Existing tokenizer/cache preparation | Inherited `SyntheticNeedleTask` fields retain their meanings | Removing or renaming inherited fields |
| Correctness evaluator | `answer_match` occurs only in the final relevant record | Reusing an answer in distractors |

## Notes

- Hop depth counts source blocks, not generated tokens. Depth one is a carrier-retention calibration; depths two through four require relational composition.
- Relevant IDs and answers are evaluation labels only; the streaming algorithm never receives them.
- Existing tasks needing at most 16 distractors are bit-for-bit unchanged because the filler pool is not extended for them.
