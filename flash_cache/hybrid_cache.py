"""Whole-state operations that are valid for hybrid attention/recurrent caches."""

from __future__ import annotations

import copy
from typing import Any

import torch


_STATE_LISTS = ("key_cache", "value_cache", "conv_states", "recurrent_states")


class UnsupportedCacheSurgery(RuntimeError):
    """Raised when an experiment requests token-block surgery on recurrent state."""


def clone_cache(cache: Any) -> Any:
    """Clone a cache so a speculative forward cannot mutate the baseline state."""
    cloned = copy.copy(cache)

    if all(hasattr(cache, name) for name in _STATE_LISTS):
        for name in _STATE_LISTS:
            values = getattr(cache, name)
            setattr(
                cloned,
                name,
                [value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) for value in values],
            )
        if hasattr(cache, "layer_types"):
            cloned.layer_types = list(cache.layer_types)
        if hasattr(cache, "transformer_layers"):
            cloned.transformer_layers = list(cache.transformer_layers)
        return cloned

    layers = getattr(cache, "layers", None)
    if layers is not None:
        cloned.layers = copy.deepcopy(layers)
        return cloned

    raise TypeError(f"Unsupported cache layout: {type(cache).__name__}")


def require_token_block_cache(cache_report: dict[str, Any]) -> None:
    """Fail loudly unless every cached layer supports token-axis block operations."""
    if cache_report["supports_arbitrary_block_concatenation"]:
        return
    recurrent = cache_report["recurrent_layer_count"]
    total = cache_report["layer_count"]
    raise UnsupportedCacheSurgery(
        f"Arbitrary token-block cache surgery is invalid: {recurrent} of {total} layers store recurrent state "
        "without a token axis. Whole-cache clone/restore remains supported."
    )

