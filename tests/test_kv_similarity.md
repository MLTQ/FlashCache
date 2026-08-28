# `test_kv_similarity.py`

## Purpose

Protects cached-value slicing, cosine aggregation, declared-metric ranking, and validation without requiring a GPU model.

## Coverage

- Recent query values exclude pinned tokens.
- A page aligned with query value vectors outranks an orthogonal page under all-layer and tail-layer reductions.
- A baseline with no query values and an unknown ranking metric fail explicitly.
- Offline-packed maximum-cosine scores and ranking match the slower discovery implementation.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| KV similarity experiment | Baseline layout is pinned then query | Changing slice boundaries |
| Reproducibility | Ties resolve by ascending page ID | Unstable ranking |
