# run_needle_experiment.py

## Purpose

Runs the smallest exhaustive Phase 1 experiment: flash every historical block independently, measure its effect over a fixed speculative trajectory, rank by both mean Jensen-Shannon divergence and its inverse ordering, and preserve candidate-level data plus a plot.

## Components

### `parse_args`
- **Does**: Defines model, seed, block count, task family and variant, valve-specific target overrides, speculative and free-generation horizons, position policy, prompt format, output directory, and GPU guard.

### `decoded_top_tokens`
- **Does**: Converts a small top-token set into inspectable JSON records.

### `main`
- **Does**: Loads Qwen3-1.7B, prepares clean baseline/cold caches, probes all candidates, evaluates format-insensitive free-form answer correctness and answer log-probability gain, records both `JS` and `1 - JS` ranks, logs JSONL and summary JSON, and renders the influence plot.
- **Interacts with**: `synthetic.py`, `probing.py`, and `metrics.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research review | `candidates.jsonl`, `summary.json`, and `influence.png` share one seed/output directory | File or field names |
| Phase 1 decision | Results retain maximum-`js_mean` as a negative control and explicitly report maximum-`one_minus_js_mean` | Silently reversing the ranking without preserving both views |
| Utility evaluation | Answer log-probability ranks are labeled evaluation-only ground-truth measurements | Using them as the retrieval selector |
| Outcome evaluation | Free-form generations are correct when they contain the task's complete numeric or categorical match phrase | Requiring exact formatting or wording |
| Prompt control | `prompt_format=chat` uses Qwen's official non-thinking serialization | Treating raw and chat results as interchangeable |
| Task variation | Target identifier and pressure are saved in every candidate row and summary | Aggregating different targets without provenance |
| Family variation | Task family, variant, and generic target key are saved in every row and summary | Mixing answer spaces without provenance |
| GPU allocation | Wrong GPU fails before model loading | Removing the device-name guard |

## Notes

- `one_minus_js_mean` is a monotone transformation of mean JS measured in nats. It does not add information; ranking it high is exactly equivalent to ranking raw JS low.
- The inverse score is an answer-free hypothesis. Ground-truth correctness remains evaluation-only and is not used to calculate it.
