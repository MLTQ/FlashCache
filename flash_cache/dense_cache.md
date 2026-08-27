# dense_cache.py

## Purpose

Implements token-axis cache slicing and concatenation for models whose every layer stores ordinary attention keys and values. It rejects hybrid or otherwise non-token-addressable cache layers.

## Components

### `cache_length`
- **Does**: Returns the shared KV sequence length and validates layer consistency.

### `spans_from_boundaries`
- **Does**: Converts validated cut points into adjacent half-open cache spans.

### `slice_cache`
- **Does**: Copies a half-open token span from every cache layer.
- **Interacts with**: `clone_cache` in `hybrid_cache.py`.

### `concatenate_caches`
- **Does**: Reassembles compatible cache blocks in a supplied physical order.
- **Rationale**: Exact same-order reassembly is the Phase 0 correctness gate before candidate insertion.

### `cache_tensor_error`
- **Does**: Aggregates key/value reconstruction error across all layers.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense surgery validation | Valid spans and same-order split/reassembly are bit-identical | Boundary, copy, or concatenation semantics |
| Candidate probing | Slices retain already-applied positional representation | Recomputing or remapping keys |
| Hybrid safeguards | Non-KV layers raise `UnsupportedCacheSurgery` | Allowing missing key/value tensors |

## Notes

- This module preserves the positional encoding already embedded in cached keys.
- Physical cache order and logical RoPE positions are separate concerns; position policies belong in the probing layer.
