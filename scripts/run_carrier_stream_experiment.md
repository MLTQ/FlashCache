# run_carrier_stream_experiment.py

## Purpose

Runs a no-selector, multi-page streaming experiment on Qwen3-1.7B. It compares page-conditioned carrier KV against the earlier clean-period behavior while keeping page order, generated choices, and task content identical.

## Components

### `parse_args`
- **Does**: Defines model, randomized corpus, one-page calibration or two-to-four-page chain depth, corpus passes, carrier tokens per page, sentinel/forced-sweep/warmup-then-break mode, generation horizons, position policy, output directory, and GPU guard.

### `main`
- **Does**: Builds one shuffled multi-hop corpus, measures no-page, normal full-prefill, and all-page Flash Cache baselines, runs poisoned and clean carrier conditions, and writes complete step traces plus a summary.
- **Interacts with**: `carrier_stream.py`, `multi_hop_tasks.py`, and `probing.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research review | `poisoned_steps.jsonl` and `clean_steps.jsonl` differ only in miss-state retention | Changing prompts, ordering, or decoding between conditions |
| No-oracle claim | Every physical page is streamed in order for each pass; relevance labels affect only logs and scoring | Filtering or reordering pages from ground truth |
| Full-corpus control | Every cached page is inserted at its original logical position under the ordinary question | Selecting only known relevant pages |
| Model-capability control | A normal prefill contains every shuffled page and uses no relevance filtering | Prefilling only the known chain |
| Multi-hop evaluation | The runtime algorithm never receives hop depth or relevant IDs as control inputs | Stopping after a known count of relevant pages |
| GPU allocation | Execution refuses a device whose name does not match the RTX 2070 SUPER guard | Removing the device-name check |

## Notes

- `passes` is a fixed page-flash budget, not a known logical-depth parameter.
- The ordinary baseline uses the original answer prompt; stream conditions use the period-until-sufficient-evidence prompt.
- Explicit insufficiency prose is normalized to the same period transition in both conditions, while its first-step page-conditioned state is retained only by the poisoned condition.
- If neither stream breaks naturally, both receive the same fixed end-of-budget instruction to answer. This tests retained state without revealing relevance or logical depth.
- Forced-sweep mode uses the ordinary question during accumulation, normalizes every intermediate output to a period, and waits for the shared final cue before judging an answer.
- Warmup-then-break mode carries one complete shuffled corpus pass, then permits speculative answer attempts during a second pass. The boundary depends only on corpus size, never hop depth or relevance.
- Carrier width defaults to one for the literal page-per-token proposal. Wider values allocate a larger placeholder-token scratchpad to every page in both poisoned and clean conditions.
