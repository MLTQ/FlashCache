# `query_capsules.py`

## Purpose

Builds compact, query-conditioned latent memory from reusable cold pages without generating visible summaries. Each page independently influences a small tail of ordinary query-token KV; raw page KV is then discarded and the compact tails are integrated by one final query refresh.

## Mechanism

For each cold page:

1. Combine pinned KV with that page alone.
2. Process the complete ordinary recent query prefix in one forward.
3. Retain only the final `capsule_width` query-token KV entries.
4. Assign retained tails non-overlapping logical positions and discard the raw page and earlier query KV.

After every page has produced a capsule, concatenate pinned KV plus all capsules, process the ordinary query once more over the compact bank, and decode from the ordinary probe token at the next logical position.

No page text is generated, no evaluation label controls retention, and decode proceeds at normal speed after the bank is built.

## Components

### `QueryCapsuleBank`

- **Does**: Carries compact capsule KV, width/count metadata, and the next logical position available for final query integration.

### `capsule_query_positions`

- **Does**: Gives each retained page-query tail a unique RoPE position span while allowing discarded earlier query tokens to overlap.
- **Rationale**: Physical cache size grows by only `pages × width`, not `pages × full_query_length`.

### `build_query_capsule_bank`

- **Does**: Flashes the full query over each independent page, slices the tail, discards raw page state, and concatenates compact capsules.
- **Interacts with**: Batched sequence refresh in `query_refresh.py` and dense cache slicing.

### `run_query_capsules`

- **Does**: Refreshes the query over the capsule bank, decodes the final answer, and reports compact cache sizes.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Capsule runner | Every page produces the same fixed width with no relevance filtering | Per-page adaptive retention based on labels or answers |
| Position layout | Retained capsule spans are unique and final query positions follow the last span | Reusing original query positions for every retained capsule |
| Context scaling | Raw page KV is absent from the integrated bank | Retaining pages alongside capsules |
| Correctness comparison | Ordinary prompt and answer horizon match all controls | Capsule-specific instructions or generated text |

## Notes

- Capsule construction currently loops over pages for experimental clarity. The page-query forwards have equal shapes for same-template pages and are candidates for cache batching if correctness is promising.
- Capsule KV is lossy: deeper retained query states encode the discarded page and earlier query tokens, but the final integration stage cannot directly attend those discarded tokens.
