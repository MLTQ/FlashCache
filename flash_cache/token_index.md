# `token_index.py`

## Purpose

Provides a minimal rare-key sidecar for cold KV pages. It indexes the original page token IDs already retained for exact replay, then ranks pages by IDF-weighted overlap with the current rewritten question.

## Components

### `ColdTokenIndex`

- **Does**: Stores immutable token-to-page postings, IDF weights, page count, and the common-token cutoff.

### `build_cold_token_index`

- **Does**: Builds postings from unique tokens per page and excludes tokens appearing above a fixed document fraction.
- **Rationale**: Chat punctuation, note templates, and very common words should not dominate rare entity/relation keys.

### `scan_query_token_overlap`

- **Does**: Adds each unique query token's IDF weight to matching pages and records overlap counts.
- **Rationale**: Exact model-token overlap reliably propagates generated entities such as `Vera` into the next retrieval step.

### `rank_token_overlap_page_ids`

- **Does**: Ranks by IDF sum, overlap count, and stable page ID.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Iterative navigation | Index uses only immutable source token IDs | Adding relevance labels or answer-derived tokens |
| Exact replay | Page IDs match cold block/text order | Reordering postings independently from the archive |
| Common-token filter | Cutoff is fixed before a query | Query-specific tuning from correctness |
| Query rewrite | Generated entity text is tokenized by the same tokenizer as pages | Mixing tokenizer vocabularies |

## Notes

- This is intentionally conventional. The experiments show that hidden KV values are not reliable rare-key retrieval vectors at archive scale.
- An inverted token index is much smaller than KV and can live beside disk/host-memory pages.
- It retrieves explicit lexical keys, while the model supplies unknown-depth semantic traversal by rewriting each next question.
