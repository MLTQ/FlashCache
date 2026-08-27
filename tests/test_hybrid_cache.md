# test_hybrid_cache.py

## Purpose

Checks whole-cache cloning and the hybrid-architecture safety guard without downloading or loading model weights.

## Components

### `FakeQwenCache`
- **Does**: Mimics the Transformers 5.3 Qwen3.5 cache lists with one recurrent and one full-attention layer.

### `test_clone_cache_has_independent_tensor_storage`
- **Does**: Proves mutations to a speculative clone do not leak into the baseline cache.

### `test_hybrid_report_rejects_arbitrary_token_block_surgery`
- **Does**: Proves recurrent layers trigger a loud compatibility failure.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `hybrid_cache.py` | Test fixture fields match the supported model-specific layout | Changing supported state-list names |

