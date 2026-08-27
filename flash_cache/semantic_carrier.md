# `semantic_carrier.py`

## Purpose

Tests whether model-generated factual tokens can carry useful information forward while cold KV pages are flashed one at a time. It separates the effect of the visible scratchpad text from any extra page-conditioned state retained in those tokens.

## Experiment contract

For every page in physical corpus order:

1. Insert that page's independently encoded KV between the pinned and recent cache regions.
2. Let the model greedily generate a fixed-width factual note while the page is present.
3. Process the generated note tokens under the same page, remove only the page KV, and retain all newly appended KV.
4. Continue with the next page without using relevance labels, target matching, an answer oracle, or a known hop count.

The exact-replay control starts from the same semantic-task baseline and processes the exact same input token IDs at the exact same positions, but never inserts a page. Its final prompt and answer horizon are identical. Therefore:

- If both paths improve similarly, the visible scratchpad explains the gain.
- If the page-conditioned path improves more, hidden page-conditioned KV contributes beyond the visible text.
- If neither improves, this carrier does not rescue the tested task/model configuration.

Two answer-free note-selection modes are available:

- `sequential` lets each newly flashed page and all previously retained note state jointly choose the next note tokens. This most directly tests accumulating response poisoning, but can develop first-page or recency attractors.
- `isolated` lets the same model propose each page's note from the static semantic prompt plus that page alone. It then forces those proposed tokens through the accumulating page-conditioned cache. This prevents earlier notes from changing what the current page writes while keeping every page, the poisoned-KV path, and exact replay.

## Components

### `make_semantic_carrier_task`

Builds generic scratchpad instructions containing the original question but neither the answer nor page relevance.

### `run_semantic_carrier`

Generates and commits one fixed-width note per page in sequential or isolated-selection mode. It records generated text, exact token IDs, source/relevance telemetry, cache lengths, and the tensor delta between each page-conditioned appended span and a forced clean encoding of that same span.

### `replay_semantic_carrier`

Re-encodes the poisoned run's exact visible transcript without any flashed page and asks the identical final question.

### `greedy_non_control_token`

Uses greedy decoding while masking model control tokens so a small model cannot terminate the fixed-width experimental carrier early. Surface formatting is not evaluated.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Semantic experiment runner | Page input token IDs are retained exactly and replayable | Sampling or normalization that changes replay IDs |
| Cache surgery | Removing a page preserves the appended page-conditioned tokens | Re-encoding or compacting the carried KV |
| Accuracy analysis | Correctness is scored only on the final answer, not the notes | Searching the scratchpad itself for the target answer |
| Causal control | No runtime choice uses relevance labels, target strings, or hop depth | Selecting, stopping, or retaining pages based on evaluation metadata |
