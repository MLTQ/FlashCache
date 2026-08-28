"""Rank intact cold KV pages from query-prefix attention and decode over a shortlist."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from flash_cache.dense_cache import cache_length, concatenate_caches, slice_cache
from flash_cache.hybrid_cache import clone_cache
from flash_cache.probing import PreparedProbeCaches, TokenizedNeedleTask, rollout
from flash_cache.query_refresh import refresh_query_prefix
from flash_cache.synthetic import SyntheticNeedleTask, contains_answer_text


ATTENTION_METRICS = (
    "all_query_mass",
    "tail_query_mass",
    "last_query_mass",
    "max_query_mass",
    "all_query_density",
    "tail_query_density",
    "last_query_density",
    "max_query_density",
)


@dataclass(frozen=True)
class PageAttentionScore:
    """Several answer-free attention aggregations for one cold page."""

    page_id: int
    token_count: int
    all_query_mass: float
    tail_query_mass: float
    last_query_mass: float
    max_query_mass: float
    all_query_density: float
    tail_query_density: float
    last_query_density: float
    max_query_density: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class AttentionScan:
    """Page scores and size telemetry from one query-attention sweep."""

    scores: tuple[PageAttentionScore, ...]
    archive_token_count: int
    query_token_count: int
    layer_count: int
    tail_layer_count: int
    tail_query_token_count: int


@dataclass(frozen=True)
class AttentionShortlistResult:
    """Answer and cache telemetry after rebuilding a fixed page shortlist."""

    selected_page_ids: tuple[int, ...]
    generated_answer: str
    answer_correct: bool
    selected_page_token_count: int
    final_cache_token_count: int


def cold_page_spans(
    tokenized_task: TokenizedNeedleTask,
    page_ids: Sequence[int] | None = None,
) -> tuple[tuple[int, int, int], ...]:
    """Return page IDs and half-open physical spans in pinned-plus-pages cache order."""
    cursor = int(tokenized_task.pinned_ids.shape[-1])
    spans: list[tuple[int, int, int]] = []
    selected = tuple(range(len(tokenized_task.block_ids))) if page_ids is None else tuple(page_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Page IDs must be a nonempty unique sequence")
    if min(selected) < 0 or max(selected) >= len(tokenized_task.block_ids):
        raise ValueError("Page ID is out of range")
    for page_id in sorted(selected):
        stop = cursor + int(tokenized_task.block_ids[page_id].shape[-1])
        spans.append((page_id, cursor, stop))
        cursor = stop
    return tuple(spans)


def aggregate_page_attention(
    attentions: Sequence[torch.Tensor | None],
    page_spans: Sequence[tuple[int, int, int]],
    archive_token_count: int,
    query_token_count: int,
    tail_layer_count: int = 4,
    tail_query_token_count: int = 4,
) -> tuple[PageAttentionScore, ...]:
    """Aggregate query-to-page attention without consulting page labels or answers."""
    if not attentions or any(attention is None for attention in attentions):
        raise ValueError("Model did not return attention tensors for every layer")
    if archive_token_count < 1 or query_token_count < 1:
        raise ValueError("Archive and query token counts must be positive")
    if tail_layer_count < 1 or tail_query_token_count < 1:
        raise ValueError("Tail layer and query widths must be positive")

    tensors = [attention for attention in attentions if attention is not None]
    expected_key_count = archive_token_count + query_token_count
    for layer_index, attention in enumerate(tensors):
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError(f"Unexpected attention shape at layer {layer_index}: {attention.shape}")
        if attention.shape[-2] != query_token_count:
            raise ValueError("Attention query axis does not match the refreshed query length")
        if attention.shape[-1] != expected_key_count:
            raise ValueError("Attention key axis does not match archive plus refreshed query")

    stacked = torch.stack([attention[0].float() for attention in tensors], dim=0)
    selected_layers = stacked[-min(tail_layer_count, len(tensors)) :]
    selected_query = selected_layers[
        ..., -min(tail_query_token_count, query_token_count) :, :archive_token_count
    ]
    all_query = stacked[..., :archive_token_count]

    scores: list[PageAttentionScore] = []
    for page_id, start, stop in page_spans:
        if not 0 <= start < stop <= archive_token_count:
            raise ValueError(f"Invalid page span [{start}:{stop}] for archive length {archive_token_count}")
        token_count = stop - start
        all_mass = float(all_query[..., start:stop].sum(dim=-1).mean().item())
        tail_mass = float(selected_query[..., start:stop].sum(dim=-1).mean().item())
        last_mass = float(selected_layers[..., -1, start:stop].sum(dim=-1).mean().item())
        per_query_mass = selected_layers[..., start:stop].sum(dim=-1).mean(dim=(0, 1))
        max_mass = float(per_query_mass.max().item())
        scores.append(
            PageAttentionScore(
                page_id=page_id,
                token_count=token_count,
                all_query_mass=all_mass,
                tail_query_mass=tail_mass,
                last_query_mass=last_mass,
                max_query_mass=max_mass,
                all_query_density=all_mass / token_count,
                tail_query_density=tail_mass / token_count,
                last_query_density=last_mass / token_count,
                max_query_density=max_mass / token_count,
            )
        )
    return tuple(scores)


def rank_page_ids(
    scores: Sequence[PageAttentionScore],
    metric: str,
    top_k: int,
) -> tuple[int, ...]:
    """Rank page IDs by a declared attention metric with deterministic tie breaking."""
    if metric not in ATTENTION_METRICS:
        raise ValueError(f"Unknown attention metric: {metric}")
    if not 1 <= top_k <= len(scores):
        raise ValueError("Top-K must be between one and the number of page scores")
    ranked = sorted(scores, key=lambda score: (-float(getattr(score, metric)), score.page_id))
    return tuple(score.page_id for score in ranked[:top_k])


def scan_query_attention(
    model: Any,
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    tail_layer_count: int = 4,
    tail_query_token_count: int = 4,
    selected_page_ids: Sequence[int] | None = None,
) -> AttentionScan:
    """Refresh the query over every intact cold page and retain only page-level scores."""
    pinned_length = int(tokenized_task.pinned_ids.shape[-1])
    pinned_cache = slice_cache(prepared.baseline_cache, 0, pinned_length)
    selected = (
        tuple(range(len(prepared.cold_blocks)))
        if selected_page_ids is None
        else tuple(selected_page_ids)
    )
    spans = cold_page_spans(tokenized_task, selected)
    ordered_blocks = tuple(prepared.cold_blocks[page_id] for page_id in sorted(selected))
    archive_cache = concatenate_caches((pinned_cache, *ordered_blocks))
    archive_token_count = cache_length(archive_cache)
    query_ids = tokenized_task.recent_prefix_ids
    query_positions = tokenized_task.recent_prefix_positions
    query_token_count = int(query_ids.shape[-1])
    branch = clone_cache(archive_cache)
    attention_mask = torch.ones(
        (1, archive_token_count + query_token_count),
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
            output_attentions=True,
            return_dict=True,
        )
    attentions = outputs.attentions
    if attentions is None:
        raise ValueError("Model returned no attention tensors")
    scores = aggregate_page_attention(
        attentions,
        spans,
        archive_token_count,
        query_token_count,
        tail_layer_count=tail_layer_count,
        tail_query_token_count=tail_query_token_count,
    )
    return AttentionScan(
        scores=scores,
        archive_token_count=archive_token_count,
        query_token_count=query_token_count,
        layer_count=len(attentions),
        tail_layer_count=min(tail_layer_count, len(attentions)),
        tail_query_token_count=min(tail_query_token_count, query_token_count),
    )


def assemble_selected_archive_cache(
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    selected_page_ids: Sequence[int],
) -> Any:
    """Assemble pinned KV and selected intact pages in original corpus order."""
    selected = tuple(selected_page_ids)
    if not selected:
        raise ValueError("At least one selected page is required")
    if len(set(selected)) != len(selected):
        raise ValueError("Selected page IDs must be unique")
    if min(selected) < 0 or max(selected) >= len(prepared.cold_blocks):
        raise ValueError("Selected page ID is out of range")
    pinned_length = int(tokenized_task.pinned_ids.shape[-1])
    pinned_cache = slice_cache(prepared.baseline_cache, 0, pinned_length)
    ordered_blocks = tuple(prepared.cold_blocks[page_id] for page_id in sorted(selected))
    return concatenate_caches((pinned_cache, *ordered_blocks))


def run_attention_shortlist(
    model: Any,
    tokenizer: Any,
    source_task: SyntheticNeedleTask,
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    selected_page_ids: Sequence[int],
    continuation_horizon: int,
) -> AttentionShortlistResult:
    """Refresh the ordinary query over selected intact page KV, then decode normally."""
    if continuation_horizon < 1:
        raise ValueError("Continuation horizon must be at least one")
    archive_cache = assemble_selected_archive_cache(tokenized_task, prepared, selected_page_ids)
    refreshed_cache = refresh_query_prefix(
        model,
        archive_cache,
        tokenized_task.recent_prefix_ids,
        tokenized_task.recent_prefix_positions,
    )
    answer = rollout(
        model,
        refreshed_cache,
        tokenized_task.probe_token,
        tokenized_task.probe_position,
        continuation_horizon,
    )
    answer_text = tokenizer.decode(answer.tokens.tolist(), skip_special_tokens=True)
    selected = tuple(selected_page_ids)
    selected_token_count = sum(
        int(tokenized_task.block_ids[page_id].shape[-1]) for page_id in selected
    )
    return AttentionShortlistResult(
        selected_page_ids=selected,
        generated_answer=answer_text,
        answer_correct=contains_answer_text(answer_text, source_task.answer_match),
        selected_page_token_count=selected_token_count,
        final_cache_token_count=cache_length(refreshed_cache),
    )
