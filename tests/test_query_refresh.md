# `test_query_refresh.py`

## Purpose

Checks that query refresh assembles exactly the reusable archive state intended by the experiment and validates query tensors before model execution.

## Components

### Archive assembly test

- **Does**: Proves pinned KV and every cold page remain in physical order while stale baseline query KV is removed.

### Empty archive test

- **Does**: Rejects a query refresh with no cold pages.

### Query tensor validation test

- **Does**: Rejects malformed ID and position shapes without requiring a model.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `query_refresh.py` | Layout is pinned followed by all cold pages, never stale query KV | Retaining the baseline recent span |
| Batched refresh | Query IDs and logical positions have identical `[1, sequence]` shape | Implicit broadcasting or token-by-token fallback |
