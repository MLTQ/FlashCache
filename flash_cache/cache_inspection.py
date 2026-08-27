"""Describe cache state without assuming every layer stores tokenwise KV tensors."""

from __future__ import annotations

from typing import Any

import torch


def _tensor_description(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "bytes": value.numel() * value.element_size(),
    }


def _state_description(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_description(value)
    if isinstance(value, (list, tuple)):
        return [_state_description(item) for item in value]
    return None


def inspect_cache(cache: Any) -> dict[str, Any]:
    """Return a JSON-serializable description of a Transformers cache object."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        raw_layers = [
            {
                "class": type(layer).__name__,
                "keys": getattr(layer, "keys", None),
                "values": getattr(layer, "values", None),
                "conv_states": getattr(layer, "conv_states", None),
                "recurrent_states": getattr(layer, "recurrent_states", None),
            }
            for layer in layers
        ]
    elif all(
        hasattr(cache, name)
        for name in ("layer_types", "key_cache", "value_cache", "conv_states", "recurrent_states")
    ):
        raw_layers = [
            {
                "class": cache.layer_types[index],
                "keys": cache.key_cache[index],
                "values": cache.value_cache[index],
                "conv_states": cache.conv_states[index],
                "recurrent_states": cache.recurrent_states[index],
            }
            for index in range(len(cache.layer_types))
        ]
    else:
        raise TypeError(f"Unsupported cache layout: {type(cache).__name__}")

    layer_descriptions: list[dict[str, Any]] = []
    total_bytes = 0
    token_addressable_layers = 0
    recurrent_layers = 0

    for index, layer in enumerate(raw_layers):
        keys = _tensor_description(layer["keys"])
        values = _tensor_description(layer["values"])
        conv_states = _state_description(layer["conv_states"])
        recurrent_states = _state_description(layer["recurrent_states"])

        tensors = [item for item in (keys, values) if item is not None]
        for state in (conv_states, recurrent_states):
            if isinstance(state, dict):
                tensors.append(state)
            elif isinstance(state, list):
                tensors.extend(item for item in state if isinstance(item, dict))
        total_bytes += sum(item["bytes"] for item in tensors)

        is_token_addressable = keys is not None and values is not None
        is_recurrent = conv_states is not None or recurrent_states is not None
        token_addressable_layers += int(is_token_addressable)
        recurrent_layers += int(is_recurrent)

        layer_descriptions.append(
            {
                "index": index,
                "class": layer["class"],
                "token_addressable": is_token_addressable,
                "recurrent": is_recurrent,
                "keys": keys,
                "values": values,
                "conv_states": conv_states,
                "recurrent_states": recurrent_states,
            }
        )

    try:
        sequence_length = int(cache.get_seq_length())
    except (AttributeError, TypeError, ValueError):
        sequence_length = None

    return {
        "cache_class": type(cache).__name__,
        "sequence_length": sequence_length,
        "layer_count": len(layer_descriptions),
        "token_addressable_layer_count": token_addressable_layers,
        "recurrent_layer_count": recurrent_layers,
        "supports_arbitrary_block_concatenation": recurrent_layers == 0,
        "total_bytes": total_bytes,
        "layers": layer_descriptions,
    }
