# `run_kv_similarity_ranking_suite.py`

## Purpose

Tests whether position-independent similarity between already-cached query and page value vectors can replace the extra model forward used by query-attention ranking.

## Trial design

- Use the same stratified development/holdout scheduling as the attention-ranking suite.
- Build the ordinary no-page baseline and independent cold-page KV once.
- Time cached-value similarity with no model call.
- Build the normalized packed page-value index outside online timing, then time the single-metric packed scan separately.
- Record all predeclared KV similarity rankings.
- Run the fixed all-query attention ranking on the same task as a retrieval and latency reference.
- Generate a full-prefill control to identify tasks within Qwen3-1.7B's capability.
- Select one global KV metric on development labels, then report it unchanged on holdout.

## KV metrics

Four reductions are measured over all layers and the configured tail layers:

- mean of the top cosine token pairs per KV head;
- mean best page-token cosine for each query token;
- mean best query-token cosine for each page token;
- maximum token-pair cosine.

The selection objective matches the attention suite: all-required-page top-K rate, then mean recall, then reciprocal worst-relevant rank, with declared metric order as the final tie break.

## Timing contract

- Cold-page construction is separate and excluded from online scans.
- KV and attention scans are synchronized independently on the same prepared caches.
- The current KV implementation uses float32 normalization and Python page/layer loops. Its latency is a conservative prototype number; batching and half-precision kernels are future optimization work.
- The packed maximum-cosine path batches all page tokens per layer in half precision, uses segmented maxima, and synchronizes once. Its top-K set is checked against the slower float32 reference on every trial.
- Packed-index build time is reported as offline preparation and excluded from packed query latency.
- Full-prefill generation is a capability control, not part of either retrieval latency.

## Outputs

- `trials.jsonl`: complete KV scores/rankings, attention reference selection, page labels, controls, and timings.
- `summary.json`: globally selected KV metric, development/holdout coverage, attention reference coverage, and scan latency comparison.
