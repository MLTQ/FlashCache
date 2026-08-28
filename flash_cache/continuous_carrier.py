"""Uninterrupted free-form decoding under rotating Flash Cache pages."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch

from flash_cache.carrier_stream import strip_flashed_page
from flash_cache.dense_cache import cache_length, cache_tensor_error, slice_cache
from flash_cache.multi_hop_tasks import MultiHopNeedleTask
from flash_cache.probing import (
    PreparedProbeCaches,
    TokenizedNeedleTask,
    advance_cache,
    flash_candidate,
    rollout,
)
from flash_cache.semantic_carrier import greedy_non_control_token
from flash_cache.synthetic import contains_answer_text

_FINAL_MARKER = "\nFinal answer to the original question:"


@dataclass(frozen=True)
class ContinuousCarrierResult:
    """One uninterrupted page-rotated transcript and its final answer."""

    steps: tuple[dict[str, Any], ...]
    visible_token_ids: tuple[int, ...]
    processed_token_ids: tuple[int, ...]
    generated_answer: str
    answer_correct: bool


@dataclass(frozen=True)
class ContinuousReplayResult:
    """Final answer after cleanly encoding the carrier's exact token sequence."""

    processed_token_ids: tuple[int, ...]
    generated_answer: str
    answer_correct: bool


def make_continuous_carrier_task(task: MultiHopNeedleTask) -> MultiHopNeedleTask:
    """Request one ongoing reasoning stream without page-level output structure."""
    query = (
        "Answer the original question while writing one continuous free-form reasoning stream. "
        "Archived information will become available invisibly as you write. Let useful facts "
        "change the same ongoing response, and state the answer whenever the evidence supports "
        "it. Do not restart, divide the response into pages, describe page boundaries, or produce "
        f"separate page notes. Original question: {task.query_message} Begin now."
    )
    return replace(
        task,
        query_message=query,
        recent_text=f"User: {query}\nAssistant:",
    )


def make_rotation_schedule(
    block_count: int,
    processed_token_count: int,
    page_window_tokens: int,
) -> tuple[int, ...]:
    """Return a deterministic round-robin page ID for every processed response token."""
    if block_count < 1:
        raise ValueError("Rotation requires at least one page")
    if processed_token_count < 1:
        raise ValueError("Rotation requires at least one processed token")
    if page_window_tokens < 1:
        raise ValueError("Page window must contain at least one token")
    return tuple(
        (token_step // page_window_tokens) % block_count
        for token_step in range(processed_token_count)
    )


def exact_replay_input_token_ids(
    probe_token: torch.Tensor,
    visible_token_ids: Sequence[int],
) -> tuple[int, ...]:
    """Prepend the original probe token to the exact visible carrier transcript."""
    if probe_token.shape != (1, 1):
        raise ValueError("Probe token must have shape [1, 1]")
    return (int(probe_token.item()), *(int(token_id) for token_id in visible_token_ids))


def _token_tensor(token_id: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([[token_id]], dtype=torch.long, device=device)


def _encode_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError(f"Text encoded to no tokens: {text!r}")
    return tuple(int(token_id) for token_id in token_ids)


def _answer_from_cache(
    model: Any,
    tokenizer: Any,
    cache: Any,
    next_position: int,
    continuation_horizon: int,
    device: torch.device,
) -> str:
    if continuation_horizon < 1:
        raise ValueError("Continuation horizon must be at least one")
    final_ids = _encode_ids(tokenizer, _FINAL_MARKER)
    branch = cache
    for offset, token_id in enumerate(final_ids[:-1]):
        branch = advance_cache(
            model,
            branch,
            _token_tensor(token_id, device),
            next_position + offset,
        ).cache
    final_rollout = rollout(
        model,
        branch,
        _token_tensor(final_ids[-1], device),
        next_position + len(final_ids) - 1,
        continuation_horizon,
    )
    return tokenizer.decode(final_rollout.tokens.tolist(), skip_special_tokens=True)


def _page_conditioned_step(
    model: Any,
    persistent_cache: Any,
    page_cache: Any,
    pinned_length: int,
    input_token: torch.Tensor,
    input_position: int,
) -> tuple[Any, torch.Tensor, dict[str, float]]:
    """Process one token under one page, strip the page, and measure its hidden-state effect."""
    active_cache = flash_candidate(persistent_cache, page_cache, pinned_length)
    active_step = advance_cache(model, active_cache, input_token, input_position)
    clean_step = advance_cache(model, persistent_cache, input_token, input_position)
    active_tail = slice_cache(
        active_step.cache,
        cache_length(active_step.cache) - 1,
        cache_length(active_step.cache),
    )
    clean_tail = slice_cache(
        clean_step.cache,
        cache_length(clean_step.cache) - 1,
        cache_length(clean_step.cache),
    )
    delta = cache_tensor_error(clean_tail, active_tail)
    retained_cache = strip_flashed_page(
        active_step.cache,
        pinned_length,
        cache_length(page_cache),
    )
    return retained_cache, active_step.logits, delta


def run_continuous_carrier(
    model: Any,
    tokenizer: Any,
    source_task: MultiHopNeedleTask,
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    carrier_token_count: int,
    page_window_tokens: int,
    continuation_horizon: int,
) -> ContinuousCarrierResult:
    """Greedily decode one response while rotating the active page on a fixed schedule."""
    if carrier_token_count < 1:
        raise ValueError("Continuous carrier must generate at least one token")
    block_count = len(prepared.cold_blocks)
    schedule = make_rotation_schedule(
        block_count,
        carrier_token_count + 1,
        page_window_tokens,
    )
    persistent_cache = prepared.baseline_cache
    pinned_length = int(tokenized_task.pinned_ids.shape[-1])
    device = tokenized_task.probe_token.device
    current_token = tokenized_task.probe_token
    current_position = tokenized_task.probe_position
    control_ids = tuple(int(token_id) for token_id in tokenizer.all_special_ids)
    relevant_ids = set(source_task.relevant_block_ids)
    visible_ids: list[int] = []
    steps: list[dict[str, Any]] = []

    for decode_step in range(carrier_token_count):
        page_id = schedule[decode_step]
        persistent_length_before = cache_length(persistent_cache)
        persistent_cache, logits, delta = _page_conditioned_step(
            model,
            persistent_cache,
            prepared.cold_blocks[page_id],
            pinned_length,
            current_token,
            current_position,
        )
        selected_token_id = greedy_non_control_token(logits, control_ids)
        visible_ids.append(selected_token_id)
        steps.append(
            {
                "decode_step": decode_step,
                "commit_only": False,
                "page_id": page_id,
                "page_window_index": decode_step // page_window_tokens,
                "page_window_offset": decode_step % page_window_tokens,
                "source_text": source_task.blocks[page_id],
                "ground_truth_relevant": page_id in relevant_ids,
                "logical_relevant_step": (
                    source_task.relevant_block_ids.index(page_id)
                    if page_id in relevant_ids
                    else None
                ),
                "input_position": current_position,
                "input_token_id": int(current_token.item()),
                "input_token_text": tokenizer.decode(
                    current_token.reshape(-1).tolist(),
                    skip_special_tokens=True,
                ),
                "selected_token_id": selected_token_id,
                "selected_token_text": tokenizer.decode(
                    [selected_token_id],
                    skip_special_tokens=True,
                ),
                "persistent_cache_length_before": persistent_length_before,
                "persistent_cache_length_after": cache_length(persistent_cache),
                "page_conditioned_token_delta_max_abs": delta["max_abs"],
                "page_conditioned_token_delta_mean_abs": delta["mean_abs"],
            }
        )
        current_token = _token_tensor(selected_token_id, device)
        current_position += 1

    final_commit_step = carrier_token_count
    final_page_id = schedule[final_commit_step]
    persistent_length_before = cache_length(persistent_cache)
    persistent_cache, _, final_delta = _page_conditioned_step(
        model,
        persistent_cache,
        prepared.cold_blocks[final_page_id],
        pinned_length,
        current_token,
        current_position,
    )
    steps.append(
        {
            "decode_step": final_commit_step,
            "commit_only": True,
            "page_id": final_page_id,
            "page_window_index": final_commit_step // page_window_tokens,
            "page_window_offset": final_commit_step % page_window_tokens,
            "source_text": source_task.blocks[final_page_id],
            "ground_truth_relevant": final_page_id in relevant_ids,
            "logical_relevant_step": (
                source_task.relevant_block_ids.index(final_page_id)
                if final_page_id in relevant_ids
                else None
            ),
            "input_position": current_position,
            "input_token_id": int(current_token.item()),
            "input_token_text": tokenizer.decode(
                current_token.reshape(-1).tolist(),
                skip_special_tokens=True,
            ),
            "selected_token_id": None,
            "selected_token_text": None,
            "persistent_cache_length_before": persistent_length_before,
            "persistent_cache_length_after": cache_length(persistent_cache),
            "page_conditioned_token_delta_max_abs": final_delta["max_abs"],
            "page_conditioned_token_delta_mean_abs": final_delta["mean_abs"],
        }
    )
    current_position += 1

    final_answer = _answer_from_cache(
        model,
        tokenizer,
        persistent_cache,
        current_position,
        continuation_horizon,
        device,
    )
    processed_ids = exact_replay_input_token_ids(
        tokenized_task.probe_token,
        visible_ids,
    )
    return ContinuousCarrierResult(
        steps=tuple(steps),
        visible_token_ids=tuple(visible_ids),
        processed_token_ids=processed_ids,
        generated_answer=final_answer,
        answer_correct=contains_answer_text(final_answer, source_task.answer_match),
    )


def replay_continuous_carrier(
    model: Any,
    tokenizer: Any,
    source_task: MultiHopNeedleTask,
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    processed_token_ids: Sequence[int],
    continuation_horizon: int,
) -> ContinuousReplayResult:
    """Process the exact uninterrupted token sequence without pages, then answer."""
    expected_probe_id = int(tokenized_task.probe_token.item())
    replay_ids = tuple(int(token_id) for token_id in processed_token_ids)
    if not replay_ids or replay_ids[0] != expected_probe_id:
        raise ValueError("Replay transcript must begin with the original probe token")
    cache = prepared.baseline_cache
    device = tokenized_task.probe_token.device
    for offset, token_id in enumerate(replay_ids):
        cache = advance_cache(
            model,
            cache,
            _token_tensor(token_id, device),
            tokenized_task.probe_position + offset,
        ).cache
    final_answer = _answer_from_cache(
        model,
        tokenizer,
        cache,
        tokenized_task.probe_position + len(replay_ids),
        continuation_horizon,
        device,
    )
    return ContinuousReplayResult(
        processed_token_ids=replay_ids,
        generated_answer=final_answer,
        answer_correct=contains_answer_text(final_answer, source_task.answer_match),
    )
