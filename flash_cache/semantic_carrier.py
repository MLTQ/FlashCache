"""Page-conditioned semantic token accumulation for Flash Cache experiments."""

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
from flash_cache.synthetic import contains_answer_text

_PAGE_MARKER = "\nPage note:"
_FINAL_MARKER = "\nFinal answer to the original question:"


@dataclass(frozen=True)
class SemanticCarrierResult:
    """A page-conditioned transcript and the answer produced from its retained KV."""

    steps: tuple[dict[str, Any], ...]
    page_input_token_ids: tuple[tuple[int, ...], ...]
    generated_answer: str
    answer_correct: bool


@dataclass(frozen=True)
class SemanticReplayResult:
    """An answer produced after cleanly encoding a fixed visible transcript."""

    replayed_token_ids: tuple[int, ...]
    generated_answer: str
    answer_correct: bool


def make_semantic_carrier_task(task: MultiHopNeedleTask) -> MultiHopNeedleTask:
    """Ask the model to externalize each flashed page as a terse factual note."""
    query = (
        "Archived pages will become available one at a time. For each newly available page, "
        "write one terse factual scratchpad note that preserves its exact names, entities, "
        "preferences, and relationships. Treat every page alike and do not decide whether to "
        "keep or discard it. Do not answer the original question until you receive the Final "
        f"answer signal. Original question: {task.query_message} The first page is available now; "
        "write its factual note."
    )
    return replace(
        task,
        query_message=query,
        recent_text=f"User: {query}\nAssistant:",
    )


def greedy_non_control_token(logits: torch.Tensor, control_token_ids: Sequence[int]) -> int:
    """Select the highest-logit token after excluding model control tokens."""
    if logits.ndim != 1:
        raise ValueError("Carrier logits must be one-dimensional")
    masked = logits.clone()
    valid_control_ids = sorted(
        {int(token_id) for token_id in control_token_ids if 0 <= int(token_id) < logits.shape[0]}
    )
    if valid_control_ids:
        masked[valid_control_ids] = -torch.inf
    selected = int(masked.argmax().item())
    if not torch.isfinite(masked[selected]):
        raise ValueError("Control-token mask excluded the entire vocabulary")
    return selected


def flatten_page_input_token_ids(
    page_input_token_ids: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Flatten a page transcript without altering token order or multiplicity."""
    return tuple(int(token_id) for page in page_input_token_ids for token_id in page)


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
) -> str:
    if continuation_horizon < 1:
        raise ValueError("Continuation horizon must be at least one")
    final_ids = _encode_ids(tokenizer, _FINAL_MARKER)
    device = next(model.parameters()).device
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


def _generate_note_tokens(
    model: Any,
    cache: Any,
    prefix_ids: Sequence[int],
    start_position: int,
    note_token_count: int,
    control_ids: Sequence[int],
    device: torch.device,
) -> tuple[int, ...]:
    """Greedily propose a fixed-width note from one private cache branch."""
    branch = cache
    logits: torch.Tensor | None = None
    cursor = start_position
    for token_id in prefix_ids:
        step = advance_cache(model, branch, _token_tensor(int(token_id), device), cursor)
        branch = step.cache
        logits = step.logits
        cursor += 1
    if logits is None:
        raise ValueError("A note-generation prefix must contain at least one token")

    generated_ids: list[int] = []
    for _ in range(note_token_count):
        selected_token_id = greedy_non_control_token(logits, control_ids)
        generated_ids.append(selected_token_id)
        step = advance_cache(
            model,
            branch,
            _token_tensor(selected_token_id, device),
            cursor,
        )
        branch = step.cache
        logits = step.logits
        cursor += 1
    return tuple(generated_ids)


def run_semantic_carrier(
    model: Any,
    tokenizer: Any,
    source_task: MultiHopNeedleTask,
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    page_order: tuple[int, ...],
    note_token_count: int,
    continuation_horizon: int,
    note_selection_mode: str = "sequential",
) -> SemanticCarrierResult:
    """Generate notes under flashed pages, remove pages, and retain the note-token KV."""
    if not page_order:
        raise ValueError("Page order must contain at least one page")
    if note_token_count < 1:
        raise ValueError("A semantic note must contain at least one generated token")
    if note_selection_mode not in {"sequential", "isolated"}:
        raise ValueError(f"Unknown note selection mode: {note_selection_mode}")
    block_count = len(prepared.cold_blocks)
    if any(page_id < 0 or page_id >= block_count for page_id in page_order):
        raise ValueError("Page order contains an out-of-range block ID")

    persistent_cache = prepared.baseline_cache
    pinned_length = int(tokenized_task.pinned_ids.shape[-1])
    device = tokenized_task.probe_token.device
    current_position = tokenized_task.probe_position
    marker_ids = _encode_ids(tokenizer, _PAGE_MARKER)
    control_ids = tuple(int(token_id) for token_id in tokenizer.all_special_ids)
    relevant_ids = set(source_task.relevant_block_ids)
    page_inputs: list[tuple[int, ...]] = []
    steps: list[dict[str, Any]] = []

    for stream_step, page_id in enumerate(page_order):
        page_cache = prepared.cold_blocks[page_id]
        persistent_length_before = cache_length(persistent_cache)
        active_cache = flash_candidate(persistent_cache, page_cache, pinned_length)
        clean_reference_cache = persistent_cache
        prefix_ids = (
            (int(tokenized_task.probe_token.item()),) if stream_step == 0 else marker_ids
        )
        if note_selection_mode == "sequential":
            selection_cache = active_cache
            selection_prefix_ids = prefix_ids
            selection_position = current_position
        else:
            selection_cache = flash_candidate(
                prepared.baseline_cache,
                page_cache,
                pinned_length,
            )
            selection_prefix_ids = (int(tokenized_task.probe_token.item()),)
            selection_position = tokenized_task.probe_position
        generated_ids = _generate_note_tokens(
            model,
            selection_cache,
            selection_prefix_ids,
            selection_position,
            note_token_count,
            control_ids,
            device,
        )

        input_ids = [*prefix_ids, *generated_ids]
        cursor = current_position
        for token_id in input_ids:
            active_step = advance_cache(
                model,
                active_cache,
                _token_tensor(token_id, device),
                cursor,
            )
            clean_step = advance_cache(
                model,
                clean_reference_cache,
                _token_tensor(token_id, device),
                cursor,
            )
            active_cache = active_step.cache
            clean_reference_cache = clean_step.cache
            cursor += 1

        appended_length = len(input_ids)
        active_tail = slice_cache(
            active_cache,
            cache_length(active_cache) - appended_length,
            cache_length(active_cache),
        )
        clean_tail = slice_cache(
            clean_reference_cache,
            cache_length(clean_reference_cache) - appended_length,
            cache_length(clean_reference_cache),
        )
        page_conditioned_delta = cache_tensor_error(clean_tail, active_tail)
        persistent_cache = strip_flashed_page(
            active_cache,
            pinned_length,
            cache_length(page_cache),
        )
        page_inputs.append(tuple(input_ids))
        steps.append(
            {
                "stream_step": stream_step,
                "page_id": page_id,
                "source_text": source_task.blocks[page_id],
                "ground_truth_relevant": page_id in relevant_ids,
                "logical_relevant_step": (
                    source_task.relevant_block_ids.index(page_id)
                    if page_id in relevant_ids
                    else None
                ),
                "input_position_start": current_position,
                "input_position_stop": cursor,
                "prefix_token_ids": list(prefix_ids),
                "input_token_ids": input_ids,
                "generated_note_token_ids": list(generated_ids),
                "generated_note": tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ),
                "note_selection_mode": note_selection_mode,
                "persistent_cache_length_before": persistent_length_before,
                "persistent_cache_length_after": cache_length(persistent_cache),
                "page_conditioned_token_delta_max_abs": page_conditioned_delta["max_abs"],
                "page_conditioned_token_delta_mean_abs": page_conditioned_delta["mean_abs"],
            }
        )
        current_position = cursor

    final_answer = _answer_from_cache(
        model,
        tokenizer,
        persistent_cache,
        current_position,
        continuation_horizon,
    )
    return SemanticCarrierResult(
        steps=tuple(steps),
        page_input_token_ids=tuple(page_inputs),
        generated_answer=final_answer,
        answer_correct=contains_answer_text(final_answer, source_task.answer_match),
    )


def replay_semantic_carrier(
    model: Any,
    tokenizer: Any,
    source_task: MultiHopNeedleTask,
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    page_input_token_ids: Sequence[Sequence[int]],
    continuation_horizon: int,
) -> SemanticReplayResult:
    """Encode an exact carrier transcript without pages, then ask the same final question."""
    replayed_ids = flatten_page_input_token_ids(page_input_token_ids)
    if not replayed_ids:
        raise ValueError("A clean replay requires at least one visible input token")
    cache = prepared.baseline_cache
    device = tokenized_task.probe_token.device
    for offset, token_id in enumerate(replayed_ids):
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
        tokenized_task.probe_position + len(replayed_ids),
        continuation_horizon,
    )
    return SemanticReplayResult(
        replayed_token_ids=replayed_ids,
        generated_answer=final_answer,
        answer_correct=contains_answer_text(final_answer, source_task.answer_match),
    )
