# `attention_shortlist.py`

## Purpose

Uses the model's own query-to-page attention as an answer-free retrieval signal. Cold page KV remains intact: the mechanism compresses only the shortlist decision, then rebuilds a small page cache for ordinary answer decoding.

## Mechanism

1. Concatenate pinned KV and every independently encoded cold page.
2. Process the existing recent query prefix once with eager attention output enabled.
3. Reduce query-to-page attention to scalar page scores without consulting answers or relevance labels.
4. Choose a fixed top-K page set.
5. Reassemble pinned KV plus those intact pages in original corpus order, refresh the ordinary query, and decode normally.

The attention sweep is a discovery pass. It does not retain generated text, query KV, or the all-page cache for final decoding. Exact-text replay over the same selected IDs is an important companion condition: it separates selection quality from limitations of independently encoded KV.

## Components

### `PageAttentionScore`

- **Does**: Stores mass and per-token density under all-query, tail-query, last-query, and max-query reductions.
- **Rationale**: Multiple predeclared reductions let the experiment determine whether useful retrieval information exists without treating correctness as an online oracle.

### `aggregate_page_attention`

- **Does**: Reduces returned layer/head/query attention tensors over declared page spans.
- **Rationale**: The pure aggregation function can be unit-tested independently from a GPU model.

### `scan_query_attention`

- **Does**: Runs one full query-prefix attention sweep over all cold pages and returns only page-level scores.
- **Optional subset**: Accepts a nonempty unique page-ID set, preserves its original corpus order, and returns scores keyed by the original IDs. This supports a cheap KV-similarity first stage followed by attention reranking.

### `rank_page_ids`

- **Does**: Selects fixed top-K IDs using a named metric and deterministic page-ID tie breaking.

### `assemble_selected_archive_cache`

- **Does**: Concatenates pinned KV and selected intact pages in their original corpus order.

### `run_attention_shortlist`

- **Does**: Refreshes the unchanged query over the selected KV pages and performs ordinary greedy decoding.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Attention runner | The model is loaded with eager attention and returns one attention tensor per layer | Switching to an implementation that returns no attention weights |
| Page scoring | Key-axis layout is pinned, all cold pages, refreshed query | Reusing stale recent KV or reordering cache segments |
| Answer-free claim | Selection receives only query attention and a fixed K | Using answer labels, answer tokens, or correctness to select pages online |
| Shortlist decoding | Selected pages retain their original RoPE positions and corpus order | Compacting positions or ranking-order concatenation |
| Holdout evaluation | A ranking metric and K are fixed before holdout tasks | Retrospectively choosing the winning metric per question |
| Hybrid reranking | Subset spans retain original page IDs even though their physical cache is compact | Renumbering subset pages from zero |

## Notes

- Attention mass can favor long pages; density metrics divide mass by page token count.
- The max-query metrics can surface a page strongly attended by one query token even if assistant-template suffix tokens dilute the average.
- The discovery sweep still attends across the full archive once. A scalable version would stream fixed-size page groups and keep only scalar scores and top-K page IDs.
- Successful exact-text replay with failed independent-KV replay would still be valuable: it would identify a cheap query-conditioned retrieval front end while isolating the remaining cache-composition problem.
