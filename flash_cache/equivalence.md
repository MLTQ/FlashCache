# equivalence.py

## Purpose

Validates that normal cached decoding and whole-cache speculative restoration reproduce a reference forward closely enough for later influence measurements.

## Components

### `logit_error`
- **Does**: Reports maximum and mean absolute error between two logit vectors in float32.

### `validate_cache_equivalence`
- **Does**: Compares a full prompt forward with prefix-cache continuation, then compares two branches cloned from the same baseline cache.
- **Interacts with**: `clone_cache` in `hybrid_cache.py` and Hugging Face causal language model outputs.
- **Rationale**: Full-vs-cached error characterizes numerical implementation differences; branch-vs-branch error validates exact speculative restoration.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `scripts/validate_cache_equivalence.py` | JSON-serializable metrics and argmax agreement | Renaming report fields |
| Probe experiments | Restored branches start from independent identical state | Removing clone-based branching |

