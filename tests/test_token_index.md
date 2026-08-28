# `test_token_index.py`

## Purpose

Protects offline token postings, common-token filtering, rare-key ranking, validation, and deterministic ties.

## Coverage

- A rare query token retrieves its page while a corpus-wide token is excluded.
- Empty/invalid page tensors fail explicitly.
- All-zero scores preserve page-ID order.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Navigation controller | Rare exact tokens dominate common note-template tokens | Removing IDF/common-token logic |
| Reproducibility | Equal scores resolve by ascending page ID | Unstable posting iteration order |
