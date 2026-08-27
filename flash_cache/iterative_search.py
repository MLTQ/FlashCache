"""Sequential sentinel-token search across cold Flash Cache blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from os.path import commonprefix
from typing import Any

import torch

from flash_cache.probing import (
    PreparedProbeCaches,
    TokenizedNeedleTask,
    advance_cache,
    flash_candidate,
    rollout,
)
from flash_cache.synthetic import SyntheticNeedleTask, contains_answer_text

_NEGATIVE_GATE_PHRASES = (
    "does not contain",
    "doesn't contain",
    "do not contain",
    "not contain",
    "does not provide",
    "not provided",
    "not found",
    "cannot answer",
    "can't answer",
    "insufficient information",
    "not enough information",
    "no answer",
)


@dataclass(frozen=True)
class IterativeSearchResult:
    """Visited-page telemetry and the first branch that broke the sentinel pattern."""

    steps: tuple[dict[str, Any], ...]
    selected_candidate_id: int | None
    selected_ground_truth_relevant: bool | None
    generated_continuation: str | None
    answer_correct: bool


def make_sentinel_search_task(task: SyntheticNeedleTask) -> SyntheticNeedleTask:
    """Replace the answer prompt with a period-until-ready search protocol."""
    query = (
        "We will inspect one archived page at a time. The current page is available in memory. "
        f"Question: {task.query_message} "
        "If the current page contains the exact answer, answer the question now. Otherwise output "
        'exactly one period character (".") and nothing else. Do not guess. After a miss, the '
        'controller may append a "NEXT PAGE:" marker; when it does, apply this same rule again to '
        "the newly available current page."
    )
    return replace(
        task,
        query_message=query,
        recent_text=f"User: {query}\nAssistant:",
    )


def make_chat_miss_transition_ids(
    tokenizer: Any,
    task: SyntheticNeedleTask,
    device: torch.device,
) -> torch.Tensor:
    """Build the token suffix for period, closed assistant turn, and fresh page prompt."""
    placeholder = "<<<FLASH_CACHE_HISTORY_BLOCKS>>>"
    initial_messages = [
        {"role": "system", "content": task.system_message},
        {
            "role": "user",
            "content": f"Archived engineering records:\n{placeholder}\n{task.query_message}",
        },
    ]
    initial_rendered = tokenizer.apply_chat_template(
        initial_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    followup_rendered = tokenizer.apply_chat_template(
        [
            *initial_messages,
            {"role": "assistant", "content": "."},
            {
                "role": "user",
                "content": "Next archived page. Apply the same rule to the new current page.",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    shared_prefix = commonprefix((initial_rendered, followup_rendered))
    if not shared_prefix.endswith("<|im_start|>assistant\n"):
        raise ValueError("Chat renderings did not share the expected assistant-start boundary")
    suffix_text = followup_rendered[len(shared_prefix) :]
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    if not suffix_ids:
        raise ValueError("Chat miss transition must not be empty")
    return torch.tensor(suffix_ids, dtype=torch.long, device=device)


def make_inline_miss_transition_ids(tokenizer: Any, device: torch.device) -> torch.Tensor:
    """Build the compact period-plus-next-page controller transition."""
    sentinel_ids = tokenizer(".", add_special_tokens=False)["input_ids"]
    marker_ids = tokenizer("\nNEXT PAGE:", add_special_tokens=False)["input_ids"]
    if len(sentinel_ids) != 1 or not marker_ids:
        raise ValueError("Inline miss transition requires one sentinel and a nonempty marker")
    transition_ids = [*sentinel_ids, *marker_ids]
    return torch.tensor(transition_ids, dtype=torch.long, device=device)


def _sentinel_step_metrics(logits: torch.Tensor, sentinel_token_id: int) -> dict[str, float | int]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum()
    top_two = log_probs.topk(2)
    greedy_token_id = int(top_two.indices[0].item())
    sentinel_log_prob = log_probs[sentinel_token_id]
    non_sentinel_log_probs = log_probs.clone()
    non_sentinel_log_probs[sentinel_token_id] = -torch.inf
    best_non_sentinel_log_prob, best_non_sentinel_id = non_sentinel_log_probs.max(dim=-1)
    return {
        "greedy_token_id": greedy_token_id,
        "greedy_log_prob": float(top_two.values[0].item()),
        "top1_log_prob_margin": float((top_two.values[0] - top_two.values[1]).item()),
        "distribution_entropy": float(entropy.item()),
        "sentinel_log_prob": float(sentinel_log_prob.item()),
        "sentinel_probability": float(sentinel_log_prob.exp().item()),
        "sentinel_log_odds_vs_best_non_sentinel": float(
            (sentinel_log_prob - best_non_sentinel_log_prob).item()
        ),
        "best_non_sentinel_token_id": int(best_non_sentinel_id.item()),
        "best_non_sentinel_log_prob": float(best_non_sentinel_log_prob.item()),
    }


def classify_gate_tokens(
    token_ids: torch.Tensor,
    tokenizer: Any,
    sentinel_token_id: int,
) -> tuple[bool, str, int | None]:
    """Ignore formatting/control tokens until the gate reaches sentinel or visible content."""
    visible_pieces = [
        tokenizer.decode([int(token_id)], skip_special_tokens=True)
        for token_id in token_ids.tolist()
    ]
    normalized_visible = " ".join("".join(visible_pieces).casefold().split())
    if any(phrase in normalized_visible for phrase in _NEGATIVE_GATE_PHRASES):
        return False, "negative_content", None
    for index, token_id in enumerate(token_ids.tolist()):
        token_id = int(token_id)
        if token_id == sentinel_token_id:
            return False, "sentinel", index
        visible_text = tokenizer.decode([token_id], skip_special_tokens=True)
        if "." in visible_text and not any(character.isalnum() for character in visible_text):
            return False, "sentinel_surface", index
        if any(character.isalnum() for character in visible_text):
            return True, "content", index
    return False, "no_signal", None


def run_iterative_flash_search(
    model: Any,
    tokenizer: Any,
    source_task: SyntheticNeedleTask,
    tokenized_search_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    sentinel_token_id: int,
    gate_horizon: int,
    continuation_horizon: int,
    miss_transition_ids: torch.Tensor | None = None,
    candidate_order: tuple[int, ...] | None = None,
) -> IterativeSearchResult:
    """Flash pages sequentially, commit clean sentinels on misses, and retain the first hit."""
    if gate_horizon < 1 or continuation_horizon < 1:
        raise ValueError("Gate and continuation horizons must be at least one")
    block_count = len(prepared.cold_blocks)
    order = candidate_order or tuple(range(block_count))
    if sorted(order) != list(range(block_count)):
        raise ValueError("Candidate order must be a permutation of all block IDs")

    clean_cache = prepared.baseline_cache
    current_token = tokenized_search_task.probe_token
    current_position = tokenized_search_task.probe_position
    pinned_length = int(tokenized_search_task.pinned_ids.shape[-1])
    steps: list[dict[str, Any]] = []

    for search_step, block_id in enumerate(order):
        active_cache = flash_candidate(
            clean_cache,
            prepared.cold_blocks[block_id],
            pinned_length,
        )
        candidate_gate = rollout(
            model,
            active_cache,
            current_token,
            current_position,
            gate_horizon,
        )
        step_metrics = _sentinel_step_metrics(candidate_gate.logits[0], sentinel_token_id)
        greedy_token_id = int(step_metrics["greedy_token_id"])
        gate_text = tokenizer.decode(candidate_gate.tokens.tolist())
        gate_visible_text = tokenizer.decode(
            candidate_gate.tokens.tolist(),
            skip_special_tokens=True,
        )
        broke_sentinel, gate_decision, gate_decision_token_index = classify_gate_tokens(
            candidate_gate.tokens,
            tokenizer,
            sentinel_token_id,
        )
        row: dict[str, Any] = {
            "search_step": search_step,
            "candidate_block_id": block_id,
            "source_text": source_task.blocks[block_id],
            "ground_truth_relevant": block_id == source_task.relevant_block_id,
            "input_position": current_position,
            "greedy_token_text": tokenizer.decode([greedy_token_id]),
            "best_non_sentinel_token_text": tokenizer.decode(
                [int(step_metrics["best_non_sentinel_token_id"])]
            ),
            "first_token_broke_exact_sentinel": greedy_token_id != sentinel_token_id,
            "gate_token_ids": [int(token_id) for token_id in candidate_gate.tokens.tolist()],
            "gate_text": gate_text,
            "gate_visible_text": gate_visible_text,
            "gate_decision": gate_decision,
            "gate_decision_token_index": gate_decision_token_index,
            "broke_sentinel": broke_sentinel,
            **step_metrics,
        }

        if broke_sentinel:
            continuation = rollout(
                model,
                active_cache,
                current_token,
                current_position,
                continuation_horizon,
            )
            generated_text = tokenizer.decode(continuation.tokens.tolist())
            answer_correct = contains_answer_text(generated_text, source_task.answer_match)
            row["generated_continuation"] = generated_text
            row["answer_correct"] = answer_correct
            steps.append(row)
            return IterativeSearchResult(
                steps=tuple(steps),
                selected_candidate_id=block_id,
                selected_ground_truth_relevant=block_id == source_task.relevant_block_id,
                generated_continuation=generated_text,
                answer_correct=answer_correct,
            )

        steps.append(row)
        if miss_transition_ids is None:
            transition = torch.tensor(
                [sentinel_token_id],
                dtype=torch.long,
                device=current_token.device,
            )
        else:
            transition = miss_transition_ids
            if transition.ndim != 1 or transition.numel() < 1:
                raise ValueError("Miss transition IDs must be a nonempty one-dimensional tensor")
            if int(transition[0].item()) != sentinel_token_id:
                raise ValueError("Miss transition must begin with the sentinel token")

        tokens_to_commit = torch.cat((current_token.reshape(-1), transition[:-1]))
        for offset, token_id in enumerate(tokens_to_commit):
            clean_step = advance_cache(
                model,
                clean_cache,
                token_id.reshape(1, 1),
                current_position + offset,
            )
            clean_cache = clean_step.cache
        current_token = transition[-1].reshape(1, 1)
        current_position += int(transition.shape[0])

    return IterativeSearchResult(
        steps=tuple(steps),
        selected_candidate_id=None,
        selected_ground_truth_relevant=None,
        generated_continuation=None,
        answer_correct=False,
    )
