"""Unit tests for cache cloning and compatibility guards without model weights."""

from __future__ import annotations

import torch

from flash_cache.hybrid_cache import UnsupportedCacheSurgery, clone_cache, require_token_block_cache


class FakeQwenCache:
    def __init__(self) -> None:
        self.layer_types = ["linear_attention", "full_attention"]
        self.transformer_layers = [1]
        self.key_cache = [None, torch.ones(1, 1, 2, 2)]
        self.value_cache = [None, torch.ones(1, 1, 2, 2) * 2]
        self.conv_states = [torch.ones(1, 2, 2), None]
        self.recurrent_states = [torch.ones(1, 2, 2) * 3, None]


def test_clone_cache_has_independent_tensor_storage() -> None:
    original = FakeQwenCache()
    cloned = clone_cache(original)

    cloned.key_cache[1].zero_()
    cloned.recurrent_states[0].zero_()

    assert torch.all(original.key_cache[1] == 1)
    assert torch.all(original.recurrent_states[0] == 3)


def test_hybrid_report_rejects_arbitrary_token_block_surgery() -> None:
    report = {
        "supports_arbitrary_block_concatenation": False,
        "recurrent_layer_count": 18,
        "layer_count": 24,
    }

    try:
        require_token_block_cache(report)
    except UnsupportedCacheSurgery as error:
        assert "18 of 24" in str(error)
    else:
        raise AssertionError("hybrid cache should be rejected")

