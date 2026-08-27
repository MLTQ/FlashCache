# run_provenance_probe_experiment.py

## Purpose

Tests answer-free page selection by extracting each flashed page's own subject key and matching it externally against the query target. This avoids readiness introspection and never shows the target key or answer to the extraction prompt.

## Components

### `parse_args`
- **Does**: Defines model, task, extraction/answer horizons, cache-position policy, prompt format, output directory, and GPU guard.

### `rollout_confidence_metrics`
- **Does**: Records mean greedy log probability, entropy, and top-token margin for winner/loser provenance generations.
- **Rationale**: Confidence is telemetry only and is not used to select the page.

### `main`
- **Does**: Exhaustively extracts provenance from every flashed page, selects the first exact target-key match, retains that page under the ordinary answer prompt, and measures end-to-end correctness.
- **Interacts with**: `make_provenance_probe_task` in `semantic_probe.py`, cache operations in `probing.py`, and task provenance in `task_families.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Answer-free selector | The model's provenance prompt contains neither target key nor answer | Leaking either value into generation |
| External controller | Candidate selection uses the complete ordered target-key token sequence after punctuation normalization | Selecting by answer correctness, fuzzy partial titles, or answer likelihood |
| End-to-end evaluation | The selected page is reinserted under the original answer prompt before decoding | Answering from the provenance prompt |
| Research record | Every candidate row is preserved, including nonmatching losers and confidence telemetry | Logging only selected candidates |
| GPU allocation | Wrong GPU fails before model loading | Removing the RTX 2070 SUPER guard |

## Notes

- Exhaustive probing is used for diagnosis; a deployable controller may stop at the first matching key.
- Matching ignores formatting punctuation but still requires every target-key token contiguously and in order.
