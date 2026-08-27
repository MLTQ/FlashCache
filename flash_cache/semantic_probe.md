# semantic_probe.py

## Purpose

Defines a query-conditioned, answer-free relevance probe for flashed cache blocks. It asks whether the single supplied record matches the query subject and scores YES against NO without exposing the requested answer.

## Components

### `contains_normalized_key`
- **Does**: Matches the complete ordered alphanumeric token sequence of a target key while ignoring case, quotation marks, Markdown, and punctuation.
- **Rationale**: Page selection should not fail because the model formats `Title (1901)` as `"Title" (1901)`.

### `make_relevance_probe_task`
- **Does**: Preserves the task blocks and target provenance while replacing the answer question with an exact-subject YES/NO question.
- **Interacts with**: `SyntheticNeedleTask` in `synthetic.py` and ordinary tokenization/cache preparation in `probing.py`.

### `make_provenance_probe_task`
- **Does**: Replaces the answer question with direct extraction of the flashed page's own event, treaty, quote, or identifier key.
- **Rationale**: Comparing a recovered page key with the query subject is grounded provenance checking, not model introspection about whether it knows the answer.

### `single_token_variant_ids`
- **Does**: Finds case and spacing variants of YES or NO that the active tokenizer represents with one token.
- **Rationale**: Summing several surface forms reduces dependence on capitalization while keeping the probe to one forward step.

### `binary_token_set_metrics`
- **Does**: Reports normalized positive/negative token-set log probabilities, log odds, and a two-class normalized probability.
- **Rationale**: Log-mean probability prevents a set with more spelling variants from winning solely by cardinality.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment runner | The relevance query contains `target_key` but never `answer` or `answer_match` | Leaking the ground-truth answer into the probe |
| Provenance selector | The extraction prompt contains neither `target_key` nor the answer; matching happens outside the model | Leaking the requested key into page-key generation |
| Key matching | Every target token must appear contiguously and in order after punctuation normalization | Fuzzy partial-title matching or answer-based matching |
| Selector ranking | Higher `semantic_yes_no_log_odds` means more relevant | Reversing score polarity |
| JSONL diagnostics | Returned metrics are JSON-serializable floats | Tensor-valued outputs |

## Notes

- This is an active model probe rather than a passive distribution statistic.
- It may still fail through instruction-following or acquiescence bias; those failures are part of the experiment.
