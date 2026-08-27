"""Numerical checks for ordinary decoding and whole-cache clone/restore."""

from __future__ import annotations

from typing import Any

import torch

from flash_cache.hybrid_cache import clone_cache


def logit_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Summarize elementwise absolute logit error."""
    error = (reference.float() - candidate.float()).abs()
    return {
        "max_abs": float(error.max().item()),
        "mean_abs": float(error.mean().item()),
    }


def validate_cache_equivalence(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    """Compare full prefill, cached continuation, and two restored speculative branches."""
    if input_ids.shape[-1] < 2:
        raise ValueError("Cache equivalence requires at least two tokens")

    prefix = input_ids[:, :-1]
    final_token = input_ids[:, -1:]
    full_attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        full = model(input_ids=input_ids, attention_mask=full_attention_mask, use_cache=True, return_dict=True)
        prefill = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True, return_dict=True)

        branch_a = clone_cache(prefill.past_key_values)
        branch_b = clone_cache(prefill.past_key_values)
        cached_a = model(
            input_ids=final_token,
            attention_mask=full_attention_mask,
            past_key_values=branch_a,
            use_cache=True,
            return_dict=True,
        )
        cached_b = model(
            input_ids=final_token,
            attention_mask=full_attention_mask,
            past_key_values=branch_b,
            use_cache=True,
            return_dict=True,
        )

    full_logits = full.logits[:, -1, :]
    cached_a_logits = cached_a.logits[:, -1, :]
    cached_b_logits = cached_b.logits[:, -1, :]
    return {
        "input_tokens": int(input_ids.shape[-1]),
        "full_vs_cached": logit_error(full_logits, cached_a_logits),
        "restored_branch_a_vs_b": logit_error(cached_a_logits, cached_b_logits),
        "full_argmax": int(full_logits.argmax(dim=-1).item()),
        "cached_argmax": int(cached_a_logits.argmax(dim=-1).item()),
        "argmax_matches": bool(full_logits.argmax(dim=-1).item() == cached_a_logits.argmax(dim=-1).item()),
    }

