"""Late-insertion cache preparation and speculative rollout for one needle task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flash_cache.dense_cache import cache_length, concatenate_caches, slice_cache
from flash_cache.hybrid_cache import clone_cache
from flash_cache.synthetic import SyntheticNeedleTask


@dataclass(frozen=True)
class TokenizedNeedleTask:
    """Token IDs and original logical positions for all task sections."""

    pinned_ids: torch.Tensor
    block_ids: tuple[torch.Tensor, ...]
    block_positions: tuple[torch.Tensor, ...]
    recent_prefix_ids: torch.Tensor
    recent_prefix_positions: torch.Tensor
    probe_token: torch.Tensor
    probe_position: int


@dataclass(frozen=True)
class Rollout:
    """Speculative logits and the shared tokens selected from them."""

    logits: torch.Tensor
    tokens: torch.Tensor


@dataclass(frozen=True)
class PreparedProbeCaches:
    """Baseline, cold blocks, and the positions actually used to encode each block."""

    baseline_cache: Any
    cold_blocks: tuple[Any, ...]
    effective_block_positions: tuple[torch.Tensor, ...]


def _encode(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


def tokenize_task(
    tokenizer: Any,
    task: SyntheticNeedleTask,
    device: torch.device,
    prompt_format: str = "raw",
) -> TokenizedNeedleTask:
    """Tokenize independent sections and assign their positions in the full logical history."""
    if prompt_format == "raw":
        pinned_text = task.pinned_text
        recent_text = task.recent_text
    elif prompt_format == "chat":
        placeholder = "<<<FLASH_CACHE_HISTORY_BLOCKS>>>"
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": task.system_message},
                {
                    "role": "user",
                    "content": f"Archived engineering records:\n{placeholder}\n{task.query_message}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if rendered.count(placeholder) != 1:
            raise ValueError("Chat template did not preserve the history placeholder exactly once")
        pinned_text, recent_text = rendered.split(placeholder)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")

    pinned_ids = _encode(tokenizer, pinned_text, device)
    block_ids = tuple(_encode(tokenizer, block, device) for block in task.blocks)
    recent_ids = _encode(tokenizer, recent_text, device)
    if recent_ids.shape[-1] < 2:
        raise ValueError("Recent query must contain a prefix and a final probe token")

    pinned_length = int(pinned_ids.shape[-1])
    cursor = pinned_length
    block_positions: list[torch.Tensor] = []
    for ids in block_ids:
        length = int(ids.shape[-1])
        block_positions.append(torch.arange(cursor, cursor + length, device=device).unsqueeze(0))
        cursor += length

    recent_length = int(recent_ids.shape[-1])
    recent_positions = torch.arange(cursor, cursor + recent_length, device=device).unsqueeze(0)
    return TokenizedNeedleTask(
        pinned_ids=pinned_ids,
        block_ids=block_ids,
        block_positions=tuple(block_positions),
        recent_prefix_ids=recent_ids[:, :-1],
        recent_prefix_positions=recent_positions[:, :-1],
        probe_token=recent_ids[:, -1:],
        probe_position=int(recent_positions[0, -1].item()),
    )


def _prefill(model: Any, input_ids: torch.Tensor, position_ids: torch.Tensor) -> Any:
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        ).past_key_values


def prepare_probe_caches(
    model: Any,
    task: TokenizedNeedleTask,
    position_policy: str = "original",
) -> PreparedProbeCaches:
    """Build a clean baseline cache and independently precomputed cold candidate blocks."""
    device = task.pinned_ids.device
    pinned_length = int(task.pinned_ids.shape[-1])
    pinned_positions = torch.arange(pinned_length, device=device).unsqueeze(0)

    baseline_ids = torch.cat((task.pinned_ids, task.recent_prefix_ids), dim=-1)
    baseline_positions = torch.cat((pinned_positions, task.recent_prefix_positions), dim=-1)
    baseline_prefill = _prefill(model, baseline_ids, baseline_positions)
    pinned_cache = slice_cache(baseline_prefill, 0, pinned_length)
    recent_cache = slice_cache(baseline_prefill, pinned_length, int(baseline_ids.shape[-1]))
    baseline_cache = concatenate_caches((pinned_cache, recent_cache))

    if position_policy == "original":
        effective_positions = task.block_positions
    elif position_policy == "hot_slot":
        recent_start = int(task.recent_prefix_positions[0, 0].item())
        effective_positions = tuple(
            torch.arange(recent_start - ids.shape[-1], recent_start, device=device).unsqueeze(0)
            for ids in task.block_ids
        )
    else:
        raise ValueError(f"Unknown position policy: {position_policy}")

    cold_blocks: list[Any] = []
    for block_ids, block_positions in zip(task.block_ids, effective_positions, strict=True):
        candidate_ids = torch.cat((task.pinned_ids, block_ids), dim=-1)
        candidate_positions = torch.cat((pinned_positions, block_positions), dim=-1)
        candidate_prefill = _prefill(model, candidate_ids, candidate_positions)
        cold_blocks.append(slice_cache(candidate_prefill, pinned_length, int(candidate_ids.shape[-1])))
    return PreparedProbeCaches(
        baseline_cache=baseline_cache,
        cold_blocks=tuple(cold_blocks),
        effective_block_positions=effective_positions,
    )


def flash_candidate(baseline_cache: Any, candidate_cache: Any, pinned_length: int) -> Any:
    """Insert one cold KV block between pinned and already-cached recent state."""
    baseline_length = cache_length(baseline_cache)
    pinned = slice_cache(baseline_cache, 0, pinned_length)
    recent = slice_cache(baseline_cache, pinned_length, baseline_length)
    return concatenate_caches((pinned, candidate_cache, recent))


def rollout(
    model: Any,
    cache: Any,
    probe_token: torch.Tensor,
    probe_position: int,
    horizon: int,
    forced_tokens: torch.Tensor | None = None,
) -> Rollout:
    """Run an uncommitted greedy or forced-token rollout from a private cache branch."""
    if horizon < 1:
        raise ValueError("Speculative horizon must be at least one")
    if forced_tokens is not None and forced_tokens.shape != (horizon,):
        raise ValueError("Forced token count must equal the speculative horizon")

    branch = clone_cache(cache)
    current_token = probe_token
    logits: list[torch.Tensor] = []
    selected_tokens: list[torch.Tensor] = []

    for step in range(horizon):
        attention_mask = torch.ones(
            (1, cache_length(branch) + 1), dtype=torch.long, device=current_token.device
        )
        position_ids = torch.tensor([[probe_position + step]], dtype=torch.long, device=current_token.device)
        with torch.inference_mode():
            outputs = model(
                input_ids=current_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=branch,
                use_cache=True,
                return_dict=True,
            )
        branch = outputs.past_key_values
        step_logits = outputs.logits[0, -1, :]
        logits.append(step_logits)
        selected = forced_tokens[step] if forced_tokens is not None else step_logits.argmax(dim=-1)
        selected_tokens.append(selected)
        current_token = selected.reshape(1, 1)

    return Rollout(logits=torch.stack(logits), tokens=torch.stack(selected_tokens))
