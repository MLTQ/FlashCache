# run_iterative_flash_experiment.py

## Purpose

Runs the sequential sentinel-token Flash Cache experiment. It scans cold pages in order, expects a period for irrelevant pages, commits clean period state after misses, and retains the first branch that begins an answer.

## Components

### `parse_args`
- **Does**: Defines the model, task family/variant, page count, private gate horizon, continuation horizon, miss-transition mode, evaluation-only relevant-first control, cache-position policy, prompt format, output directory, and GPU guard.

### `main`
- **Does**: Loads Qwen3-1.7B, builds the answer-free sentinel prompt and cold caches, derives clean chat-turn miss transitions, runs noise-tolerant iterative search, and saves visited-step JSONL plus an end-to-end summary.
- **Interacts with**: `iterative_search.py`, `probing.py`, and `task_families.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research review | `steps.jsonl` contains every visited winner/loser and `summary.json` contains the stopping outcome | Omitting rejected steps or changing filenames |
| GPU allocation | The process refuses any device whose name does not match the expected RTX 2070 SUPER guard | Removing the guard |
| Sentinel protocol | `.` encodes as exactly one token; private punctuation/control-only gate output remains a miss | Accepting a multi-token sentinel or committing noisy private output |
| Iteration modes | Inline, full chat-turn, and single-period transitions all begin with the period sentinel and commit only on the clean cache | Reusing a rejected page branch |
| End-to-end evaluation | The first meaningful visible continuation retains its candidate block throughout answer decoding | Decoding from the clean cache after a hit |
| Evaluation control | `--relevant-first-control` is labeled in the summary and never presented as a deployable search order | Mixing oracle ordering into retrieval results |

## Notes

- Inline mode is the default: `.` plus a short `NEXT PAGE:` marker avoids chat termination and minimizes positional drift.
- `baseline_no_page_greedy_text` diagnoses whether the sentinel instruction works even without a page.
- False breaks stop the search and count as failures; the runner does not peek ahead to the relevant page.
