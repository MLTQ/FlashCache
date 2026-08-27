"""Persistent page-conditioned token state for streaming Flash Cache pages."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from flash_cache.dense_cache import (
    cache_length,
    cache_tensor_error,
    concatenate_caches,
    slice_cache,
)
from flash_cache.multi_hop_tasks import MultiHopNeedleTask
from flash_cache.probing import (
    PreparedProbeCaches,
    TokenizedNeedleTask,
    advance_cache,
    flash_candidate,
    rollout,
)
from flash_cache.synthetic import contains_answer_text

_WAIT_PHRASES = (
    "insufficient",
    "do not know",
    "don't know",
    "cannot answer",
    "can't answer",
    "not enough",
    "no answer",
    "no information",
    "not available",
    "does not contain",
    "do not contain",
    "not contain",
    "does not include",
    "no mention",
    "no direct mention",
    "does not provide",
    "without additional information",
    "without additional details",
    "not possible to determine",
    "cannot determine",
    "unable to determine",
    "need more",
    "unable to answer",
)


@dataclass(frozen=True)
class CarrierStreamResult:
    """Per-flash telemetry and any response that breaks the period stream."""

    steps: tuple[dict[str, Any], ...]
    break_page_id: int | None
    break_page_relevant: bool | None
    generated_answer: str | None
    answer_correct: bool
    answer_source: str | None


def make_carrier_stream_task(task: MultiHopNeedleTask) -> MultiHopNeedleTask:
    """Replace the ordinary question with instructions for one persistent page stream."""
    query = (
        "Archived pages will become available one at a time while you produce one continuous "
        f"response. Question: {task.query_message} Retain useful connections from every page seen "
        "so far. If the accumulated evidence is insufficient, output exactly one period character "
        '(".") and nothing else. As soon as the accumulated evidence supports the answer, answer '
        "the question immediately. Do not guess."
    )
    return replace(
        task,
        query_message=query,
        recent_text=f"User: {query}\nAssistant:",
    )


def strip_flashed_page(
    advanced_cache: Any,
    pinned_length: int,
    flashed_length: int,
) -> Any:
    """Remove only the flashed span while preserving newly appended page-conditioned KV."""
    total_length = cache_length(advanced_cache)
    flash_stop = pinned_length + flashed_length
    if pinned_length < 1 or flashed_length < 1 or flash_stop >= total_length:
        raise ValueError("Flash span must leave nonempty pinned and carried cache regions")
    return concatenate_caches(
        (
            slice_cache(advanced_cache, 0, pinned_length),
            slice_cache(advanced_cache, flash_stop, total_length),
        )
    )


def _step_metrics(logits: torch.Tensor, sentinel_token_id: int) -> dict[str, float | int]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probabilities = log_probs.exp()
    top_two = log_probs.topk(2)
    sentinel_log_prob = log_probs[sentinel_token_id]
    return {
        "greedy_token_id": int(top_two.indices[0].item()),
        "greedy_log_prob": float(top_two.values[0].item()),
        "top1_log_prob_margin": float((top_two.values[0] - top_two.values[1]).item()),
        "distribution_entropy": float((-(probabilities * log_probs).sum()).item()),
        "sentinel_log_prob": float(sentinel_log_prob.item()),
        "sentinel_probability": float(sentinel_log_prob.exp().item()),
    }


def classify_carrier_gate(generated_text: str) -> tuple[bool, str]:
    """Treat punctuation and explicit insufficiency prose as waits, not attempted answers."""
    normalized = " ".join(generated_text.casefold().split())
    if any(phrase in normalized for phrase in _WAIT_PHRASES):
        return False, "explicit_wait"
    visible_characters = [character for character in normalized if not character.isspace()]
    if visible_characters and all(character == "." for character in visible_characters):
        return False, "sentinel"
    if not any(character.isalnum() for character in normalized):
        return False, "no_content"
    return True, "answer_attempt"


def visible_tokens_before_control(token_ids: torch.Tensor, tokenizer: Any) -> tuple[list[int], str]:
    """Decode only tokens before the first model control token in a speculative response."""
    special_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
    visible_ids: list[int] = []
    for token_id in token_ids.tolist():
        token_id = int(token_id)
        if token_id in special_ids:
            break
        visible_ids.append(token_id)
    return visible_ids, tokenizer.decode(visible_ids, skip_special_tokens=True)


def run_carrier_stream(
    model: Any,
    tokenizer: Any,
    source_task: MultiHopNeedleTask,
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    page_order: tuple[int, ...],
    sentinel_token_id: int,
    continuation_horizon: int,
    carry_page_state: bool,
    break_after_steps: int | None,
    carrier_tokens_per_page: int,
) -> CarrierStreamResult:
    """Cycle pages per token and optionally retain each page-conditioned token KV on misses."""
    if not page_order:
        raise ValueError("Page order must contain at least one flash")
    if continuation_horizon < 1:
        raise ValueError("Continuation horizon must be at least one")
    if carrier_tokens_per_page < 1:
        raise ValueError("Carrier tokens per page must be at least one")
    block_count = len(prepared.cold_blocks)
    if any(page_id < 0 or page_id >= block_count for page_id in page_order):
        raise ValueError("Page order contains an out-of-range block ID")

    persistent_cache = prepared.baseline_cache
    current_token = tokenized_task.probe_token
    current_position = tokenized_task.probe_position
    pinned_length = int(tokenized_task.pinned_ids.shape[-1])
    relevant_ids = set(source_task.relevant_block_ids)
    steps: list[dict[str, Any]] = []

    for stream_step, page_id in enumerate(page_order):
        page_cache = prepared.cold_blocks[page_id]
        persistent_length_before = cache_length(persistent_cache)
        active_cache = flash_candidate(persistent_cache, page_cache, pinned_length)
        active_step = advance_cache(
            model,
            active_cache,
            current_token,
            current_position,
        )
        clean_reference_step = advance_cache(
            model,
            persistent_cache,
            current_token,
            current_position,
        )
        active_token_cache = slice_cache(
            active_step.cache,
            cache_length(active_step.cache) - 1,
            cache_length(active_step.cache),
        )
        clean_token_cache = slice_cache(
            clean_reference_step.cache,
            cache_length(clean_reference_step.cache) - 1,
            cache_length(clean_reference_step.cache),
        )
        page_conditioned_token_delta = cache_tensor_error(
            clean_token_cache,
            active_token_cache,
        )
        metrics = _step_metrics(active_step.logits, sentinel_token_id)
        selected_token_id = int(metrics["greedy_token_id"])
        may_break = break_after_steps is not None and stream_step >= break_after_steps
        if may_break:
            speculative = rollout(
                model,
                active_cache,
                current_token,
                current_position,
                continuation_horizon,
            )
            speculative_text = tokenizer.decode(speculative.tokens.tolist())
            decision_token_ids, speculative_visible_text = visible_tokens_before_control(
                speculative.tokens,
                tokenizer,
            )
            broke_sentinel, gate_decision = classify_carrier_gate(speculative_visible_text)
        else:
            speculative_text = tokenizer.decode([selected_token_id])
            speculative_visible_text = tokenizer.decode(
                [selected_token_id],
                skip_special_tokens=True,
            )
            decision_token_ids = [selected_token_id]
            broke_sentinel = False
            gate_decision = "forced_carrier"
        row: dict[str, Any] = {
            "stream_step": stream_step,
            "page_id": page_id,
            "source_text": source_task.blocks[page_id],
            "ground_truth_relevant": page_id in relevant_ids,
            "logical_relevant_step": (
                source_task.relevant_block_ids.index(page_id) if page_id in relevant_ids else None
            ),
            "input_position": current_position,
            "input_token_text": tokenizer.decode(current_token.reshape(-1).tolist()),
            "selected_token_text": tokenizer.decode([selected_token_id]),
            "speculative_text": speculative_text,
            "speculative_visible_text": speculative_visible_text,
            "decision_token_ids": decision_token_ids,
            "gate_decision": gate_decision,
            "break_enabled": may_break,
            "persistent_cache_length_before": persistent_length_before,
            "carry_page_state": carry_page_state,
            "carrier_tokens_per_page": carrier_tokens_per_page,
            "page_conditioned_token_delta_max_abs": page_conditioned_token_delta["max_abs"],
            "page_conditioned_token_delta_mean_abs": page_conditioned_token_delta["mean_abs"],
            **metrics,
        }

        if broke_sentinel:
            answer_correct = contains_answer_text(speculative_text, source_task.answer_match)
            row["broke_sentinel"] = True
            row["generated_answer"] = speculative_text
            row["answer_correct"] = answer_correct
            steps.append(row)
            return CarrierStreamResult(
                steps=tuple(steps),
                break_page_id=page_id,
                break_page_relevant=page_id in relevant_ids,
                generated_answer=speculative_text,
                answer_correct=answer_correct,
                answer_source="stream_break",
            )

        if carry_page_state:
            carried_active_cache = active_step.cache
            for carrier_offset in range(1, carrier_tokens_per_page):
                carried_active_cache = advance_cache(
                    model,
                    carried_active_cache,
                    torch.tensor(
                        [[sentinel_token_id]],
                        dtype=torch.long,
                        device=current_token.device,
                    ),
                    current_position + carrier_offset,
                ).cache
            persistent_cache = strip_flashed_page(
                carried_active_cache,
                pinned_length,
                cache_length(page_cache),
            )
        else:
            persistent_cache = clean_reference_step.cache
            clean_current_token = torch.tensor(
                [[sentinel_token_id]],
                dtype=torch.long,
                device=current_token.device,
            )
            for carrier_offset in range(1, carrier_tokens_per_page):
                persistent_cache = advance_cache(
                    model,
                    persistent_cache,
                    clean_current_token,
                    current_position + carrier_offset,
                ).cache
        row["broke_sentinel"] = False
        row["persistent_cache_length_after"] = cache_length(persistent_cache)
        steps.append(row)
        current_token = torch.tensor([[sentinel_token_id]], dtype=torch.long, device=current_token.device)
        current_position += carrier_tokens_per_page

    final_prompt_ids = tokenizer(
        "\nAll archived pages have now been inspected. Give the final answer to the original "
        "question now.\nAnswer:",
        add_special_tokens=False,
    )["input_ids"]
    if not final_prompt_ids:
        raise ValueError("Final answer transition must not be empty")
    transition = torch.tensor(final_prompt_ids, dtype=torch.long, device=current_token.device)
    tokens_to_commit = torch.cat((current_token.reshape(-1), transition[:-1]))
    for offset, token_id in enumerate(tokens_to_commit):
        persistent_cache = advance_cache(
            model,
            persistent_cache,
            token_id.reshape(1, 1),
            current_position + offset,
        ).cache
    final_position = current_position + int(tokens_to_commit.shape[0])
    final_rollout = rollout(
        model,
        persistent_cache,
        transition[-1].reshape(1, 1),
        final_position,
        continuation_horizon,
    )
    final_answer = tokenizer.decode(final_rollout.tokens.tolist())
    final_answer_correct = contains_answer_text(final_answer, source_task.answer_match)
    return CarrierStreamResult(
        steps=tuple(steps),
        break_page_id=None,
        break_page_relevant=None,
        generated_answer=final_answer,
        answer_correct=final_answer_correct,
        answer_source="end_of_budget",
    )
