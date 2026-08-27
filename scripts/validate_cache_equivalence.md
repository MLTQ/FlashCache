# validate_cache_equivalence.py

## Purpose

Runs the decisive Phase 0 numerical checks for an ordinary causal text model on the selected GPU. The default target is Qwen3-1.7B; the script persists both equivalence metrics and explicit block-surgery compatibility.

## Components

### `parse_args`
- **Does**: Defines model, output, GPU guard, and offline-loading options.

### `main`
- **Does**: Loads a Hugging Face causal LM in FP16, runs equivalence checks, inventories the cache, and writes JSON evidence.
- **Interacts with**: `validate_cache_equivalence`, `inspect_cache`, and `require_token_block_cache`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Phase 0 workflow | Wrong GPUs fail before weights load | Removing device-name validation |
| Later probes | Restored branches match and incompatibility is recorded, not suppressed | Omitting either result |
