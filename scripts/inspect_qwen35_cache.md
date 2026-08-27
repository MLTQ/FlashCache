# inspect_qwen35_cache.py

## Purpose

Runs the first model-backed Phase 0 check on the selected GPU and records Qwen3.5's concrete cache representation. It uses the text tokenizer directly and fails early if CUDA device filtering did not expose the intended 2070 Super.

## Components

### `parse_args`
- **Does**: Defines the model, output, GPU guard, and offline-loading options.

### `main`
- **Does**: Loads Qwen3.5 in FP16 with eager attention, tokenizes text without vision dependencies, performs a short prefill, and writes the cache inventory as JSON.
- **Interacts with**: `inspect_cache` in `flash_cache/cache_inspection.py`.
- **Rationale**: The persisted report is evidence for choosing block-KV surgery or whole-state checkpoint probing.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Phase 0 workflow | Nonzero exit on the wrong GPU | Removing the GPU assertion |
| Research notes | JSON report contains model, GPU, token count, and layer inventory | Changing report fields |
