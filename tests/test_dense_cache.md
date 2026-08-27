# test_dense_cache.py

## Purpose

Checks token-axis cache slicing, concatenation, exact reconstruction, and storage independence without loading model weights.

## Components

### `FakeLayer` / `FakeCache`
- **Does**: Mimics the generalized Transformers cache layout with deterministic tensors.

### `test_slice_and_concatenate_reconstruct_exact_cache`
- **Does**: Proves ordered blocks reconstruct every key/value element exactly.

### `test_boundaries_form_adjacent_spans`
- **Does**: Proves cut points produce adjacent spans without a strict-zip length mismatch.

### `test_slices_do_not_share_storage_with_source`
- **Does**: Proves mutations to a sliced block do not mutate the source cache.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `dense_cache.py` | Generalized cache layers expose `keys` and `values` | Changing supported tensor attributes |
