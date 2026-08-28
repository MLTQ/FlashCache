"""Rank cold pages by position-independent similarity of cached value vectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from flash_cache.dense_cache import cache_length, slice_cache
from flash_cache.probing import PreparedProbeCaches, TokenizedNeedleTask


KV_SIMILARITY_METRICS = (
    "all_top_pair_cosine",
    "tail_top_pair_cosine",
    "all_query_max_cosine",
    "tail_query_max_cosine",
    "all_page_max_cosine",
    "tail_page_max_cosine",
    "all_max_cosine",
    "tail_max_cosine",
)


@dataclass(frozen=True)
class KVSimilarityScore:
    """Position-independent query/page value-vector similarity for one cold page."""

    page_id: int
    token_count: int
    all_top_pair_cosine: float
    tail_top_pair_cosine: float
    all_query_max_cosine: float
    tail_query_max_cosine: float
    all_page_max_cosine: float
    tail_page_max_cosine: float
    all_max_cosine: float
    tail_max_cosine: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class KVSimilarityScan:
    """Page scores and dimensions from one cached-vector scan."""

    scores: tuple[KVSimilarityScore, ...]
    query_token_count: int
    layer_count: int
    tail_layer_count: int
    top_pair_count: int


@dataclass(frozen=True)
class PackedColdValueIndex:
    """Offline-packed, normalized page values and page lengths for every model layer."""

    normalized_layer_values: tuple[torch.Tensor, ...]
    page_lengths: torch.Tensor
    page_token_counts: tuple[int, ...]


@dataclass(frozen=True)
class PackedValueMaxScan:
    """Fast all-layer maximum-cosine page scores from a packed cold index."""

    scores: tuple[float, ...]
    query_token_count: int
    layer_count: int
    page_count: int


def _cache_layers(cache: Any) -> list[Any]:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise ValueError("Cache has no layer collection")
    for layer_index, layer in enumerate(layers):
        if not isinstance(getattr(layer, "values", None), torch.Tensor):
            raise ValueError(f"Cache layer {layer_index} has no value tensor")
    return list(layers)


def _layer_similarity_metrics(
    query_values: torch.Tensor,
    page_values: torch.Tensor,
    top_pair_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if query_values.ndim != 4 or page_values.ndim != 4:
        raise ValueError("Cache values must have shape [batch, heads, tokens, dimensions]")
    if query_values.shape[0] != 1 or page_values.shape[0] != 1:
        raise ValueError("KV similarity currently supports batch size one")
    if query_values.shape[:2] + query_values.shape[3:] != page_values.shape[:2] + page_values.shape[3:]:
        raise ValueError("Query and page value tensors have incompatible shapes")
    if top_pair_count < 1:
        raise ValueError("Top-pair count must be positive")
    query = functional.normalize(query_values[0].float(), dim=-1)
    page = functional.normalize(page_values[0].float(), dim=-1)
    cosine = torch.einsum("hqd,hpd->hqp", query, page)
    flat = cosine.flatten(start_dim=-2)
    effective_top_k = min(top_pair_count, int(flat.shape[-1]))
    top_pair = flat.topk(effective_top_k, dim=-1).values.mean(dim=-1).mean()
    query_max = cosine.max(dim=-1).values.mean()
    page_max = cosine.max(dim=-2).values.mean()
    maximum = cosine.max()
    return top_pair, query_max, page_max, maximum


def _reduce_layer_metrics(
    metrics: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[float, float, float, float]:
    if not metrics:
        raise ValueError("At least one layer metric is required")
    columns = tuple(zip(*metrics, strict=True))
    return tuple(float(torch.stack(column).mean().item()) for column in columns)  # type: ignore[return-value]


def scan_kv_value_similarity(
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    tail_layer_count: int = 4,
    top_pair_count: int = 4,
) -> KVSimilarityScan:
    """Compare cached no-page query values with every independent cold page."""
    if tail_layer_count < 1 or top_pair_count < 1:
        raise ValueError("Tail-layer and top-pair counts must be positive")
    pinned_length = int(tokenized_task.pinned_ids.shape[-1])
    baseline_length = cache_length(prepared.baseline_cache)
    if baseline_length <= pinned_length:
        raise ValueError("Baseline cache contains no recent query values")
    query_cache = slice_cache(prepared.baseline_cache, pinned_length, baseline_length)
    query_layers = _cache_layers(query_cache)
    tail_count = min(tail_layer_count, len(query_layers))
    scores: list[KVSimilarityScore] = []

    for page_id, page_cache in enumerate(prepared.cold_blocks):
        page_layers = _cache_layers(page_cache)
        if len(page_layers) != len(query_layers):
            raise ValueError("Query and page caches have different layer counts")
        layer_metrics = [
            _layer_similarity_metrics(query_layer.values, page_layer.values, top_pair_count)
            for query_layer, page_layer in zip(query_layers, page_layers, strict=True)
        ]
        all_metrics = _reduce_layer_metrics(layer_metrics)
        tail_metrics = _reduce_layer_metrics(layer_metrics[-tail_count:])
        scores.append(
            KVSimilarityScore(
                page_id=page_id,
                token_count=cache_length(page_cache),
                all_top_pair_cosine=all_metrics[0],
                tail_top_pair_cosine=tail_metrics[0],
                all_query_max_cosine=all_metrics[1],
                tail_query_max_cosine=tail_metrics[1],
                all_page_max_cosine=all_metrics[2],
                tail_page_max_cosine=tail_metrics[2],
                all_max_cosine=all_metrics[3],
                tail_max_cosine=tail_metrics[3],
            )
        )
    return KVSimilarityScan(
        scores=tuple(scores),
        query_token_count=baseline_length - pinned_length,
        layer_count=len(query_layers),
        tail_layer_count=tail_count,
        top_pair_count=top_pair_count,
    )


def build_packed_cold_value_index(
    prepared: PreparedProbeCaches,
    dtype: torch.dtype = torch.float16,
) -> PackedColdValueIndex:
    """Pack and normalize cold-page values once, outside the online query path."""
    if not prepared.cold_blocks:
        raise ValueError("Packed index requires at least one cold page")
    page_layer_groups = [_cache_layers(page) for page in prepared.cold_blocks]
    layer_count = len(page_layer_groups[0])
    if layer_count < 1 or any(len(group) != layer_count for group in page_layer_groups):
        raise ValueError("Cold pages must have the same positive layer count")
    page_token_counts = tuple(cache_length(page) for page in prepared.cold_blocks)
    device = page_layer_groups[0][0].values.device
    packed_layers: list[torch.Tensor] = []
    for layer_index in range(layer_count):
        concatenated = torch.cat(
            [group[layer_index].values for group in page_layer_groups],
            dim=-2,
        )
        packed_layers.append(functional.normalize(concatenated.float(), dim=-1).to(dtype))
    return PackedColdValueIndex(
        normalized_layer_values=tuple(packed_layers),
        page_lengths=torch.tensor(page_token_counts, dtype=torch.long, device=device),
        page_token_counts=page_token_counts,
    )


def scan_packed_value_max_similarity(
    tokenized_task: TokenizedNeedleTask,
    prepared: PreparedProbeCaches,
    packed_index: PackedColdValueIndex,
) -> PackedValueMaxScan:
    """Score every packed page with one all-layer maximum-cosine reduction."""
    pinned_length = int(tokenized_task.pinned_ids.shape[-1])
    baseline_length = cache_length(prepared.baseline_cache)
    if baseline_length <= pinned_length:
        raise ValueError("Baseline cache contains no recent query values")
    query_layers = _cache_layers(prepared.baseline_cache)
    if len(query_layers) != len(packed_index.normalized_layer_values):
        raise ValueError("Packed index and query cache have different layer counts")
    if packed_index.page_lengths.numel() != len(packed_index.page_token_counts):
        raise ValueError("Packed index page lengths are inconsistent")

    accumulated = torch.zeros(
        len(packed_index.page_token_counts),
        dtype=torch.float32,
        device=packed_index.page_lengths.device,
    )
    for query_layer, packed_values in zip(
        query_layers,
        packed_index.normalized_layer_values,
        strict=True,
    ):
        query_values = query_layer.values[..., pinned_length:baseline_length, :]
        query = functional.normalize(query_values.float(), dim=-1).to(packed_values.dtype)
        cosine = torch.einsum("bhqd,bhpd->hqp", query, packed_values)
        token_maximum = cosine.amax(dim=(0, 1))
        page_maximum = torch.segment_reduce(
            token_maximum,
            reduce="max",
            lengths=packed_index.page_lengths,
        )
        accumulated += page_maximum.float()
    scores = accumulated / len(query_layers)
    return PackedValueMaxScan(
        scores=tuple(float(score) for score in scores.tolist()),
        query_token_count=baseline_length - pinned_length,
        layer_count=len(query_layers),
        page_count=len(packed_index.page_token_counts),
    )


def rank_packed_value_page_ids(scan: PackedValueMaxScan, top_k: int) -> tuple[int, ...]:
    """Rank fast packed-value scores with deterministic page-ID tie breaking."""
    if not 1 <= top_k <= len(scan.scores):
        raise ValueError("Top-K must be between one and the number of packed page scores")
    ranked = sorted(enumerate(scan.scores), key=lambda item: (-item[1], item[0]))
    return tuple(page_id for page_id, _ in ranked[:top_k])


def rank_kv_similarity_page_ids(
    scores: Sequence[KVSimilarityScore],
    metric: str,
    top_k: int,
) -> tuple[int, ...]:
    """Rank page IDs by a declared cached-value metric with stable tie breaking."""
    if metric not in KV_SIMILARITY_METRICS:
        raise ValueError(f"Unknown KV similarity metric: {metric}")
    if not 1 <= top_k <= len(scores):
        raise ValueError("Top-K must be between one and the number of page scores")
    ranked = sorted(scores, key=lambda score: (-float(getattr(score, metric)), score.page_id))
    return tuple(score.page_id for score in ranked[:top_k])
