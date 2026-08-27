"""Unit tests for token-addressable cache block operations."""

from __future__ import annotations

import torch

from flash_cache.dense_cache import (
    cache_length,
    cache_tensor_error,
    concatenate_caches,
    slice_cache,
    spans_from_boundaries,
)


class FakeLayer:
    def __init__(self, offset: int) -> None:
        values = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) + offset
        self.keys = values.clone()
        self.values = values.clone() * 2


class FakeCache:
    def __init__(self) -> None:
        self.layers = [FakeLayer(0), FakeLayer(100)]


def test_slice_and_concatenate_reconstruct_exact_cache() -> None:
    cache = FakeCache()
    blocks = [slice_cache(cache, 0, 1), slice_cache(cache, 1, 3)]
    reconstructed = concatenate_caches(blocks)

    assert cache_length(reconstructed) == 3
    assert cache_tensor_error(cache, reconstructed) == {"max_abs": 0.0, "mean_abs": 0.0}


def test_boundaries_form_adjacent_spans() -> None:
    assert spans_from_boundaries([0, 4, 9, 12]) == [(0, 4), (4, 9), (9, 12)]


def test_slices_do_not_share_storage_with_source() -> None:
    cache = FakeCache()
    original = cache.layers[0].keys.clone()
    block = slice_cache(cache, 0, 1)
    block.layers[0].keys.zero_()

    assert torch.equal(cache.layers[0].keys, original)
