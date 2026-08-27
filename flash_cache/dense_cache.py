"""Token-axis slicing and concatenation for all-attention Transformers caches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from flash_cache.hybrid_cache import UnsupportedCacheSurgery, clone_cache


def _layers(cache: Any) -> list[Any]:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise UnsupportedCacheSurgery(f"Cache {type(cache).__name__} has no generalized layer collection")
    for index, layer in enumerate(layers):
        if not isinstance(getattr(layer, "keys", None), torch.Tensor) or not isinstance(
            getattr(layer, "values", None), torch.Tensor
        ):
            raise UnsupportedCacheSurgery(f"Cache layer {index} is not a token-addressable KV layer")
    return list(layers)


def cache_length(cache: Any) -> int:
    """Return the shared token-axis length after checking layer consistency."""
    lengths = {int(layer.keys.shape[-2]) for layer in _layers(cache)}
    if len(lengths) != 1:
        raise ValueError(f"Cache layers have inconsistent sequence lengths: {sorted(lengths)}")
    return lengths.pop()


def spans_from_boundaries(boundaries: Sequence[int]) -> list[tuple[int, int]]:
    """Convert monotonically increasing cut points into adjacent half-open spans."""
    if len(boundaries) < 2:
        raise ValueError("At least two cache boundaries are required")
    spans = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    if any(start >= stop for start, stop in spans):
        raise ValueError(f"Cache boundaries must be strictly increasing: {list(boundaries)}")
    return spans


def slice_cache(cache: Any, start: int, stop: int) -> Any:
    """Copy a half-open token interval from every KV layer."""
    length = cache_length(cache)
    if not 0 <= start < stop <= length:
        raise ValueError(f"Invalid cache slice [{start}:{stop}] for length {length}")

    sliced = clone_cache(cache)
    for layer in _layers(sliced):
        layer.keys = layer.keys[..., start:stop, :].clone()
        layer.values = layer.values[..., start:stop, :].clone()
    return sliced


def concatenate_caches(blocks: Sequence[Any]) -> Any:
    """Concatenate compatible KV blocks in their supplied physical order."""
    if not blocks:
        raise ValueError("At least one cache block is required")
    layer_groups = [_layers(block) for block in blocks]
    layer_count = len(layer_groups[0])
    if any(len(group) != layer_count for group in layer_groups):
        raise ValueError("All cache blocks must contain the same number of layers")

    result = clone_cache(blocks[0])
    for index, result_layer in enumerate(_layers(result)):
        keys = [group[index].keys for group in layer_groups]
        values = [group[index].values for group in layer_groups]
        reference_key_shape = keys[0].shape[:2] + keys[0].shape[3:]
        reference_value_shape = values[0].shape[:2] + values[0].shape[3:]
        if any(tensor.shape[:2] + tensor.shape[3:] != reference_key_shape for tensor in keys):
            raise ValueError(f"Incompatible key shapes at layer {index}")
        if any(tensor.shape[:2] + tensor.shape[3:] != reference_value_shape for tensor in values):
            raise ValueError(f"Incompatible value shapes at layer {index}")
        result_layer.keys = torch.cat(keys, dim=-2)
        result_layer.values = torch.cat(values, dim=-2)
    return result


def cache_tensor_error(reference: Any, candidate: Any) -> dict[str, float]:
    """Measure maximum and mean absolute error over every cached key and value."""
    reference_layers = _layers(reference)
    candidate_layers = _layers(candidate)
    if len(reference_layers) != len(candidate_layers):
        raise ValueError("Caches contain different layer counts")

    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    for reference_layer, candidate_layer in zip(reference_layers, candidate_layers, strict=True):
        for reference_tensor, candidate_tensor in (
            (reference_layer.keys, candidate_layer.keys),
            (reference_layer.values, candidate_layer.values),
        ):
            if reference_tensor.shape != candidate_tensor.shape:
                raise ValueError("Caches contain different tensor shapes")
            error = (reference_tensor.float() - candidate_tensor.float()).abs()
            maximum = max(maximum, float(error.max().item()))
            absolute_sum += float(error.sum().item())
            element_count += error.numel()
    return {"max_abs": maximum, "mean_abs": absolute_sum / element_count}
