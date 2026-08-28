# `run_attention_shortlist_experiment.py`

## Purpose

Runs a discovery experiment for answer-free page retrieval from query attention, then separates retrieval quality from cache-composition quality with two fixed top-K reconstruction paths.

## Conditions

- **No page**: ordinary baseline without cold pages.
- **All-page query refresh**: intact independent page KV plus one query refresh, with no shortlisting.
- **Attention scan**: one query-prefix pass over all independent cold pages with attention output enabled.
- **Independent-KV shortlist**: selected intact page KV, refreshed query, and normal fixed-horizon decode.
- **Exact-text shortlist**: selected original page text is replayed in a compact prompt, then decoded through the same fixed-horizon rollout loop.
- **Full prefill all pages**: all page text is replayed as the positive correctness control.
- **Fixed-prefix control**: pages zero through K-minus-one, independent of the query.
- **Oracle relevant-plus-fill control**: all labeled relevant pages plus deterministic filler. This is an evaluation-only upper bound and never a candidate online controller.

## Ranking diagnostics

All predeclared attention metrics are reported from the same scan:

- all query tokens across all layers;
- the last configured query tokens across the last configured layers;
- the last query token across the last configured layers;
- the maximum query-token mass across the last configured layers;
- a per-token density form of each metric.

The runner records full rankings, ranks of labeled relevant pages, top-K coverage, and the distinct selected sets. Identical selected sets are decoded once and shared across metrics.

## Answer evaluation

Every reconstruction is evaluated two ways:

- free-form generation with legacy phrase-presence telemetry;
- teacher-forced likelihood over all favorite-food values present in the archive.

The likelihood evaluation teacher-forces one shared `Final answer:` cue, scores only the following answer tokens, and reports the correct value's rank, mean-token log-probability margin over the best distractor, and a probability normalized over the archive-derived answer set. The cue avoids penalizing a model merely because it prefers an explanatory opening. This is evaluation-only and never influences page selection, and is the primary signal when a response mentions the right value but explicitly declines to connect it to the query.

## Timing contract

- Cold page preparation is reported separately and excluded from online totals.
- Attention scan and shortlist reconstruction are synchronized GPU timings.
- `total_with_attention_scan_latency_ms` is the scan plus the chosen reconstruction path.
- Exact-text and full-prefill controls use the same fixed-horizon Python rollout as KV paths, avoiding early-stop timing differences from `generate`.
- This remains a screening benchmark. A surviving method requires repeated warm trials and separate time-to-first-token and steady-state decode throughput measurements.

## Example

```bash
CUDA_VISIBLE_DEVICES=<2070S_UUID> python scripts/run_attention_shortlist_experiment.py \
  --seed 68 --blocks 12 --hop-depth 2 --task-variant 4 --top-k 4 \
  --prompt-format chat --output-dir outputs/phase8/attention_shortlist_seed68
```
