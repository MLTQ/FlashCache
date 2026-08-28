"""Build and query a compact IDF-weighted inverted index over cold-page token IDs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ColdTokenIndex:
    """Offline token postings and IDF weights for immutable cold pages."""

    page_count: int
    postings: dict[int, tuple[int, ...]]
    inverse_document_frequency: dict[int, float]
    max_document_fraction: float


@dataclass(frozen=True)
class TokenOverlapScore:
    """Rare-token overlap diagnostics for one page."""

    page_id: int
    idf_score: float
    matched_query_token_count: int

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def build_cold_token_index(
    page_token_ids: Sequence[torch.Tensor],
    max_document_fraction: float = 0.5,
) -> ColdTokenIndex:
    """Create token-to-page postings, excluding tokens that occur in too many pages."""
    if not page_token_ids:
        raise ValueError("Token index requires at least one page")
    if not 0.0 < max_document_fraction <= 1.0:
        raise ValueError("Maximum document fraction must be in (0, 1]")
    page_count = len(page_token_ids)
    mutable_postings: dict[int, list[int]] = {}
    for page_id, ids in enumerate(page_token_ids):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[-1] < 1:
            raise ValueError("Every page token tensor must have shape [1, nonempty sequence]")
        for token_id in set(int(value) for value in ids[0].tolist()):
            mutable_postings.setdefault(token_id, []).append(page_id)
    maximum_documents = max(1, math.floor(page_count * max_document_fraction))
    postings = {
        token_id: tuple(page_ids)
        for token_id, page_ids in mutable_postings.items()
        if len(page_ids) <= maximum_documents
    }
    idf = {
        token_id: math.log((page_count + 1) / (len(page_ids) + 1)) + 1.0
        for token_id, page_ids in postings.items()
    }
    return ColdTokenIndex(
        page_count=page_count,
        postings=postings,
        inverse_document_frequency=idf,
        max_document_fraction=max_document_fraction,
    )


def scan_query_token_overlap(
    query_token_ids: torch.Tensor,
    index: ColdTokenIndex,
) -> tuple[TokenOverlapScore, ...]:
    """Score pages by unique query-token IDF overlap without inspecting page labels or answers."""
    if query_token_ids.ndim != 2 or query_token_ids.shape[0] != 1:
        raise ValueError("Query token IDs must have shape [1, sequence]")
    scores = [0.0] * index.page_count
    match_counts = [0] * index.page_count
    for token_id in set(int(value) for value in query_token_ids[0].tolist()):
        page_ids = index.postings.get(token_id, ())
        weight = index.inverse_document_frequency.get(token_id, 0.0)
        for page_id in page_ids:
            scores[page_id] += weight
            match_counts[page_id] += 1
    return tuple(
        TokenOverlapScore(
            page_id=page_id,
            idf_score=scores[page_id],
            matched_query_token_count=match_counts[page_id],
        )
        for page_id in range(index.page_count)
    )


def rank_token_overlap_page_ids(
    scores: Sequence[TokenOverlapScore],
    top_k: int,
) -> tuple[int, ...]:
    """Rank pages by IDF score, then overlap count, then stable page ID."""
    if not 1 <= top_k <= len(scores):
        raise ValueError("Top-K must be between one and the number of page scores")
    ranked = sorted(
        scores,
        key=lambda score: (-score.idf_score, -score.matched_query_token_count, score.page_id),
    )
    return tuple(score.page_id for score in ranked[:top_k])
