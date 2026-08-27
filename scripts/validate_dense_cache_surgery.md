# validate_dense_cache_surgery.py

## Purpose

Runs the strict Phase 0 surgery gate for an all-attention model: split one real cache into token blocks, reassemble them unchanged, and require bit-identical tensors and probe logits.

## Components

### `parse_args`
- **Does**: Defines model, output, GPU guard, and offline-loading options.

### `main`
- **Does**: Prefills Qwen3-1.7B, slices its cache into three spans, reassembles them, probes both caches, and writes JSON evidence.
- **Interacts with**: `spans_from_boundaries`, `slice_cache`, `concatenate_caches`, `cache_tensor_error`, and `clone_cache`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Phase 1 probing | The script exits successfully only on exact reconstruction | Weakening the exactness gate |
| Experiment records | Output includes spans, tensor error, logit error, and GPU | Removing report fields |
