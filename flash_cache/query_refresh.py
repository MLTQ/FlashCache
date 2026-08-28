"""Refresh short query KV over concatenated independently cached cold pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flash_cache.dense_cache import cache_length, concatenate_caches, slice_cache
from flash_cache.hybrid_cache import clone_cache
from flash_cache.probing import PreparedProbeCaches, TokenizedNeedleTask, rollout
from flash_cache.synthetic import SyntheticNeedleTask, contains_answer_text


@dataclass(frozen=True)
class QueryRefreshResult:
    """Answer and cache-size telemetry from one refreshed-query decode."""

    generated_answer: str
    answer_correct: bool
    cold_archive_token_count: int
    refreshed_query_token_count: int
    final_cache_token_count: int


def assemble_cold_archive_cache(
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
) -> Any:
    """Concatenate pinned KV and all independent cold pages, omitting stale recent KV."""
    if not prepared.cold_blocks:
        raise ValueError("Query refresh requires at least one cold page")
    pinned_length = int(tokenized_task.pinned_ids.shape[-1])
    pinned_cache = slice_cache(prepared.baseline_cache, 0, pinned_length)
    return concatenate_caches((pinned_cache, *prepared.cold_blocks))


def refresh_query_prefix(
    model: Any,
    archive_cache: Any,
    query_ids: torch.Tensor,
    query_positions: torch.Tensor,
) -> Any:
    """Process the entire recent query prefix in one causal forward over cold page KV."""
    if query_ids.ndim != 2 or query_ids.shape[0] != 1 or query_ids.shape[-1] < 1:
        raise ValueError("Query IDs must have shape [1, sequence] with a nonempty sequence")
    if query_positions.shape != query_ids.shape:
        raise ValueError("Query positions must match query ID shape")
    branch = clone_cache(archive_cache)
    attention_mask = torch.ones(
        (1, cache_length(branch) + query_ids.shape[-1]),
        dtype=torch.long,
        device=query_ids.device,
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=query_ids,
            attention_mask=attention_mask,
            position_ids=query_positions,
            past_key_values=branch,
            use_cache=True,
            return_dict=True,
        )
    return outputs.past_key_values


def run_query_refresh(
    model: Any,
    tokenizer: Any,
    source_task: SyntheticNeedleTask,
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    continuation_horizon: int,
) -> QueryRefreshResult:
    """Recompute only the recent query over cold pages, then decode the ordinary answer."""
    if continuation_horizon < 1:
        raise ValueError("Continuation horizon must be at least one")
    archive_cache = assemble_cold_archive_cache(tokenized_task, prepared)
    archive_length = cache_length(archive_cache)
    refreshed_cache = refresh_query_prefix(
        model,
        archive_cache,
        tokenized_task.recent_prefix_ids,
        tokenized_task.recent_prefix_positions,
    )
    refreshed_query_length = int(tokenized_task.recent_prefix_ids.shape[-1])
    expected_length = archive_length + refreshed_query_length
    if cache_length(refreshed_cache) != expected_length:
        raise ValueError("Refreshed cache length does not match archive plus query prefix")
    answer = rollout(
        model,
        refreshed_cache,
        tokenized_task.probe_token,
        tokenized_task.probe_position,
        continuation_horizon,
    )
    answer_text = tokenizer.decode(answer.tokens.tolist(), skip_special_tokens=True)
    return QueryRefreshResult(
        generated_answer=answer_text,
        answer_correct=contains_answer_text(answer_text, source_task.answer_match),
        cold_archive_token_count=archive_length,
        refreshed_query_token_count=refreshed_query_length,
        final_cache_token_count=expected_length,
    )
