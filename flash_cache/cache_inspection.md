# cache_inspection.py

## Purpose

Describes an instantiated model cache without presuming that every layer is a token-indexed key/value cache. It supports both Transformers 5.3's model-specific Qwen3.5 list layout and the newer generalized `Cache.layers` layout. This is the Phase 0 guard against applying ordinary KV surgery to hybrid recurrent architectures.

## Components

### `inspect_cache`
- **Does**: Produces a JSON-serializable inventory of cache layer classes, state shapes, memory, and block-addressability.
- **Interacts with**: Hugging Face generalized `Cache` implementations, `Qwen3_5DynamicCache`, and `inspect_qwen35_cache.py`.
- **Rationale**: Qwen3.5 mixes full attention with Gated DeltaNet layers whose recurrent states cannot be sliced along a token axis.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `scripts/inspect_qwen35_cache.py` | A dictionary serializable by `json.dump` | Returning tensors or custom objects |
| Phase 0 tests | `supports_arbitrary_block_concatenation` is false when recurrent state is present | Changing the compatibility rule |

## Notes

- A `false` compatibility result does not mean the cache cannot be cloned or restored as a whole.
- Token-addressable attention layers may still support partial experiments, but that is not equivalent to whole-model KV block splicing.
