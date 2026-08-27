# run_needle_experiment.py

## Purpose

Runs the exhaustive Phase 1 single-block experiment. It flashes every historical block independently, records broad passive diagnostics, evaluates a query-conditioned semantic relevance probe, and preserves candidate-level outcomes plus legacy distribution telemetry.

## Components

### `parse_args`
- **Does**: Defines model, task, speculative/free-generation/behavior horizons, semantic-probe toggle, position policy, prompt format, output directory, and GPU guard.

### `decoded_top_tokens`
- **Does**: Converts a small top-token set into inspectable JSON records.

### `main`
- **Does**: Loads Qwen3-1.7B, prepares answer and relevance-probe caches, evaluates all candidates, records format-insensitive correctness, passive behavior telemetry, semantic YES/NO scores, legacy distribution metrics, JSONL, summary JSON, and a diagnostic plot.
- **Interacts with**: `candidate_diagnostics.py`, `semantic_probe.py`, `synthetic.py`, `probing.py`, and `metrics.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research review | `candidates.jsonl`, `summary.json`, and `influence.png` share one seed/output directory | File or field names |
| Phase 1 decision | Results retain maximum-`js_mean` as a negative control and explicitly report maximum-`one_minus_js_mean` | Silently reversing the ranking without preserving both views |
| Utility evaluation | Answer log-probability ranks are labeled evaluation-only ground-truth measurements | Using them as the retrieval selector |
| Answer-free selection | Semantic-probe ranking uses only the query target and flashed record, never the answer | Leaking `answer` or `answer_match` into the probe |
| Winner/loser telemetry | Behavioral diagnostics are recorded for every candidate with stable field names | Logging only successful candidates or silently changing formulas |
| Outcome evaluation | Free-form generations are correct when they contain the task's complete numeric or categorical match phrase | Requiring exact formatting or wording |
| Prompt control | `prompt_format=chat` uses Qwen's official non-thinking serialization | Treating raw and chat results as interchangeable |
| Task variation | Target identifier and pressure are saved in every candidate row and summary | Aggregating different targets without provenance |
| Family variation | Task family, variant, and generic target key are saved in every row and summary | Mixing answer spaces without provenance |
| GPU allocation | Wrong GPU fails before model loading | Removing the device-name guard |

## Notes

- `one_minus_js_mean` is a monotone transformation of mean JS measured in nats. It does not add information; ranking it high is exactly equivalent to ranking raw JS low.
- JS-family metrics are retained as background telemetry but are no longer the primary selector hypothesis.
- Ground-truth correctness and answer likelihood are evaluation-only and are not used by the semantic probe or passive behavior metrics.
