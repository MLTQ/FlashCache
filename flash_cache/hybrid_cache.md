# hybrid_cache.py

## Purpose

Provides whole-cache operations that remain valid for hybrid models mixing attention KV with recurrent state. It intentionally refuses to treat recurrent tensors as token blocks.

## Components

### `clone_cache`
- **Does**: Creates an independent tensor-level clone of model-specific or generalized Transformers caches.
- **Interacts with**: Qwen3.5 speculative/equivalence probes.
- **Rationale**: Model forwards mutate cache objects in place, so every uncommitted branch needs a private clone.

### `require_token_block_cache`
- **Does**: Raises unless every layer in an inspection report is token-addressable.
- **Interacts with**: `inspect_cache` in `cache_inspection.py`.
- **Rationale**: Prevents a hybrid recurrent cache from being silently mislabeled as ordinary KV block surgery.

### `UnsupportedCacheSurgery`
- **Does**: Communicates a scientific incompatibility rather than a transient runtime failure.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Equivalence and probe code | Clones do not share mutable tensor storage | Returning shallow state lists |
| Phase 0 validation | Hybrid caches fail token-block compatibility loudly | Weakening the recurrent-layer guard |

