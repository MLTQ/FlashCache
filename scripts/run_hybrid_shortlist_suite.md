# `run_hybrid_shortlist_suite.py`

## Purpose

Evaluates a two-stage answer-free retrieval funnel: a fast packed cold-value scan selects a generous coarse set, query attention reranks only that set, and the final page IDs are reconstructed for answer decoding.

## Conditions

- **No page**: ordinary baseline.
- **Hybrid independent KV**: packed top-M, subset attention top-K, intact independent-KV query refresh.
- **Hybrid exact replay**: the same hybrid top-K original page tokens replayed compactly.
- **Packed exact replay**: direct packed-value top-K replay, without attention.
- **Attention exact replay**: all-page attention top-K replay, the stronger but more expensive reference.
- **Fixed-prefix exact replay**: query-independent first K pages.
- **Full prefill**: every page.

## Retrieval policy

- Coarse metric: globally fixed all-layer maximum value-vector cosine.
- Reranking metric: globally fixed all-query attention mass.
- Coarse M and final K are fixed before the run.
- Relevance labels and answers are used only for offline coverage/correctness reporting.

## Timing contract

- Cold-page creation and packed-index construction are offline.
- Online hybrid scan latency is packed scan plus subset attention reranking.
- Full all-page attention is timed independently on the same prepared cache.
- At 12 tiny pages, model-launch overhead can dominate subset attention; the intended benefit appears as archive size grows while M remains fixed.
- All reconstruction paths use identical fixed-horizon decoding.

## Outputs

- `trials.jsonl`: stage selections, coverage, generated answers, strict/permissive correctness, and timings.
- `summary.json`: coverage/answer aggregates on all and full-prefill-solvable tasks plus scan latency comparison.
