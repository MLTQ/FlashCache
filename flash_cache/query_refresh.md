# `query_refresh.py`

## Purpose

Tests a low-overhead alternative to rotating decode. Independently precomputed cold page KV remains reusable, but the stale no-page query KV is discarded and the short recent query prefix is recomputed once while attending to every cold page.

## Mechanism

1. Slice the pinned prefix from the ordinary no-page baseline cache.
2. Concatenate every independently cached page in physical corpus order at its original logical position.
3. Process the complete recent query prefix in one causal model forward over that archive cache.
4. Decode normally from the original final probe token.

Archive pages are never re-encoded online. The extra online prefill is proportional to the query length, and ordinary autoregressive token speed is unchanged after refresh.

## Components

### `QueryRefreshResult`

- **Does**: Stores the final answer, correctness, and archive/query/final cache sizes.

### `assemble_cold_archive_cache`

- **Does**: Preserves pinned KV, appends every cold page, and omits baseline recent KV that was computed without those pages.
- **Interacts with**: Token-axis cache operations in `dense_cache.py`.

### `refresh_query_prefix`

- **Does**: Processes all recent-prefix tokens in one causal forward against the assembled archive.
- **Rationale**: A batched query refresh avoids one Python/model call per query token and preserves the intended fast online path.

### `run_query_refresh`

- **Does**: Assembles pages, refreshes the query, checks cache lengths, and runs an ordinary greedy answer rollout.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Query-refresh runner | Cold pages were encoded independently using original logical positions | Silently compacting or reordering page positions |
| Online-cost analysis | Archive page KV is already available; only concatenation, query refresh, and decode are timed online | Including page encoding only in the refresh condition |
| Cache layout | Final order is pinned, every cold page, refreshed recent query | Reusing stale recent KV or moving query before pages |
| Correctness comparison | Prompt, page corpus, probe, and answer horizon match stale all-page and full-prefill controls | Changing instructions by condition |

## Notes

- This does not recreate full-prefill page KV: deeper page states remain independent and cannot attend to other pages. It gives the refreshed query tokens access to every page at every layer, which may be sufficient for cross-page integration.
