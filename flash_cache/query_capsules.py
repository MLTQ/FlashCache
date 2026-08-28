"""Compact query-conditioned latent capsules derived from reusable cold page KV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flash_cache.dense_cache import cache_length, concatenate_caches, slice_cache
from flash_cache.probing import PreparedProbeCaches, TokenizedNeedleTask, rollout
from flash_cache.query_refresh import refresh_query_prefix
from flash_cache.synthetic import SyntheticNeedleTask, contains_answer_text


@dataclass(frozen=True)
class QueryCapsuleBank:
    """Compact cache bank plus the next free logical position for final integration."""

    cache: Any
    final_query_position_start: int
    capsule_token_count: int
    capsule_width: int


@dataclass(frozen=True)
class QueryCapsuleResult:
    """Answer and size telemetry from the two-stage latent capsule path."""

    generated_answer: str
    answer_correct: bool
    capsule_token_count: int
    capsule_width: int
    refreshed_query_token_count: int
    final_cache_token_count: int
    final_probe_position: int


def capsule_query_positions(
    archive_position_stop: int,
    query_token_count: int,
    capsule_width: int,
    page_index: int,
    device: torch.device,
) -> torch.Tensor:
    """Place each page's retained query tail in a unique logical position span."""
    if archive_position_stop < 1:
        raise ValueError("Archive position stop must be positive")
    if query_token_count < 1:
        raise ValueError("Capsule query must contain at least one token")
    if not 1 <= capsule_width <= query_token_count:
        raise ValueError("Capsule width must be between one and the query token count")
    if page_index < 0:
        raise ValueError("Page index cannot be negative")
    query_start = archive_position_stop + page_index * capsule_width
    return torch.arange(
        query_start,
        query_start + query_token_count,
        device=device,
    ).unsqueeze(0)


def build_query_capsule_bank(
    model: Any,
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    capsule_width: int,
) -> QueryCapsuleBank:
    """Encode the query against each page and retain only a unique-position KV tail."""
    if not prepared.cold_blocks:
        raise ValueError("Capsule construction requires at least one cold page")
    query_ids = tokenized_task.recent_prefix_ids
    query_token_count = int(query_ids.shape[-1])
    if not 1 <= capsule_width <= query_token_count:
        raise ValueError("Capsule width must be between one and the query token count")
    pinned_length = int(tokenized_task.pinned_ids.shape[-1])
    pinned_cache = slice_cache(prepared.baseline_cache, 0, pinned_length)
    archive_position_stop = int(tokenized_task.recent_prefix_positions[0, 0].item())
    capsules: list[Any] = []

    for page_index, page_cache in enumerate(prepared.cold_blocks):
        page_base = concatenate_caches((pinned_cache, page_cache))
        query_positions = capsule_query_positions(
            archive_position_stop,
            query_token_count,
            capsule_width,
            page_index,
            query_ids.device,
        )
        page_query_cache = refresh_query_prefix(
            model,
            page_base,
            query_ids,
            query_positions,
        )
        capsules.append(
            slice_cache(
                page_query_cache,
                cache_length(page_query_cache) - capsule_width,
                cache_length(page_query_cache),
            )
        )

    capsule_token_count = len(capsules) * capsule_width
    bank_cache = concatenate_caches((pinned_cache, *capsules))
    final_query_position_start = (
        archive_position_stop + query_token_count + (len(capsules) - 1) * capsule_width
    )
    return QueryCapsuleBank(
        cache=bank_cache,
        final_query_position_start=final_query_position_start,
        capsule_token_count=capsule_token_count,
        capsule_width=capsule_width,
    )


def run_query_capsules(
    model: Any,
    tokenizer: Any,
    source_task: SyntheticNeedleTask,
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    capsule_width: int,
    continuation_horizon: int,
) -> QueryCapsuleResult:
    """Build page capsules, integrate them with one query refresh, and decode normally."""
    if continuation_horizon < 1:
        raise ValueError("Continuation horizon must be at least one")
    bank = build_query_capsule_bank(
        model,
        tokenized_task,
        prepared,
        capsule_width,
    )
    query_token_count = int(tokenized_task.recent_prefix_ids.shape[-1])
    final_query_positions = torch.arange(
        bank.final_query_position_start,
        bank.final_query_position_start + query_token_count,
        device=tokenized_task.recent_prefix_ids.device,
    ).unsqueeze(0)
    integrated_cache = refresh_query_prefix(
        model,
        bank.cache,
        tokenized_task.recent_prefix_ids,
        final_query_positions,
    )
    final_probe_position = bank.final_query_position_start + query_token_count
    answer = rollout(
        model,
        integrated_cache,
        tokenized_task.probe_token,
        final_probe_position,
        continuation_horizon,
    )
    answer_text = tokenizer.decode(answer.tokens.tolist(), skip_special_tokens=True)
    return QueryCapsuleResult(
        generated_answer=answer_text,
        answer_correct=contains_answer_text(answer_text, source_task.answer_match),
        capsule_token_count=bank.capsule_token_count,
        capsule_width=bank.capsule_width,
        refreshed_query_token_count=query_token_count,
        final_cache_token_count=cache_length(integrated_cache),
        final_probe_position=final_probe_position,
    )
