"""Answer-free semantic relevance probes for flashed cache blocks."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any

import torch

from flash_cache.synthetic import SyntheticNeedleTask


def contains_normalized_key(generated_text: str, target_key: str) -> bool:
    """Match a complete ordered key token sequence while ignoring punctuation and formatting."""
    generated_tokens = re.findall(r"[\w]+", generated_text.casefold())
    target_tokens = re.findall(r"[\w]+", target_key.casefold())
    if not target_tokens:
        raise ValueError("Target key must contain at least one alphanumeric token")
    width = len(target_tokens)
    return any(
        generated_tokens[start : start + width] == target_tokens
        for start in range(len(generated_tokens) - width + 1)
    )


def make_relevance_probe_task(task: SyntheticNeedleTask) -> SyntheticNeedleTask:
    """Replace the answer query with an exact-subject YES/NO relevance question."""
    query = (
        "Consider only the single archived record supplied above. Does it refer to the exact "
        f'requested subject "{task.target_key}"? Reply with exactly YES or NO.'
    )
    return replace(
        task,
        query_message=query,
        recent_text=f"User: {query}\nAssistant:",
    )


def make_provenance_probe_task(task: SyntheticNeedleTask) -> SyntheticNeedleTask:
    """Replace the answer query with direct extraction of the flashed page's own key."""
    query = (
        "Read only the single archived record supplied above. Copy the exact subject key that "
        "identifies what that record is about. For a historical event or treaty, copy its full "
        "title and year. For a literary entry, copy the full quoted line. For a valve record, copy "
        "the valve identifier. Do not answer any other question and do not guess."
    )
    return replace(
        task,
        query_message=query,
        recent_text=f"User: {query}\nAssistant:",
    )


def single_token_variant_ids(tokenizer: Any, variants: tuple[str, ...]) -> tuple[int, ...]:
    """Return distinct token IDs for surface variants that encode as one token."""
    token_ids: set[int] = set()
    for variant in variants:
        encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if len(encoded) == 1:
            token_ids.add(int(encoded[0]))
    if not token_ids:
        raise ValueError(f"No single-token variants found for {variants!r}")
    return tuple(sorted(token_ids))


def binary_token_set_metrics(
    logits: torch.Tensor,
    positive_token_ids: tuple[int, ...],
    negative_token_ids: tuple[int, ...],
) -> dict[str, float]:
    """Compare normalized next-token probability mass for two surface-form sets."""
    if logits.ndim != 1:
        raise ValueError("Binary probe logits must have shape [vocabulary]")
    positive = tuple(sorted(set(positive_token_ids)))
    negative = tuple(sorted(set(negative_token_ids)))
    if not positive or not negative:
        raise ValueError("Binary probe token sets must not be empty")
    if set(positive) & set(negative):
        raise ValueError("Binary probe token sets must not overlap")

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    positive_index = torch.tensor(positive, device=logits.device)
    negative_index = torch.tensor(negative, device=logits.device)
    positive_log_prob = torch.logsumexp(log_probs[positive_index], dim=0) - math.log(len(positive))
    negative_log_prob = torch.logsumexp(log_probs[negative_index], dim=0) - math.log(len(negative))
    log_odds = positive_log_prob - negative_log_prob

    return {
        "semantic_yes_log_prob": float(positive_log_prob.item()),
        "semantic_no_log_prob": float(negative_log_prob.item()),
        "semantic_yes_no_log_odds": float(log_odds.item()),
        "semantic_yes_probability_normalized": float(torch.sigmoid(log_odds).item()),
    }
