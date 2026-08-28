# `test_query_capsules.py`

## Purpose

Checks the logical-position scheme that lets full per-page query forwards yield compact, non-overlapping retained capsule tails.

## Components

### Unique-tail test

- **Does**: Confirms full query spans may overlap while retained tail spans are adjacent and non-overlapping across pages.

### Invalid-layout tests

- **Does**: Rejects empty archives/queries, invalid capsule widths, and negative page indices.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `query_capsules.py` | Page `i` shifts its full query span by `i × capsule_width` | Position layouts that overlap retained tails |
| Final integration | Last capsule ends exactly where the final query begins | Gaps or overlap after the capsule bank |
