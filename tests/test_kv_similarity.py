"""Unit tests for position-independent cached-value page similarity."""

from types import SimpleNamespace

import pytest
import torch

from flash_cache.kv_similarity import (
    build_packed_cold_value_index,
    rank_kv_similarity_page_ids,
    rank_packed_value_page_ids,
    scan_kv_value_similarity,
    scan_packed_value_max_similarity,
)


class _FakeLayer:
    def __init__(self, vectors: list[list[float]]) -> None:
        tensor = torch.tensor(vectors, dtype=torch.float32).reshape(1, 1, len(vectors), -1)
        self.values = tensor.clone()
        self.keys = tensor.clone()


class _FakeCache:
    def __init__(self, layer_vectors: list[list[list[float]]]) -> None:
        self.layers = [_FakeLayer(vectors) for vectors in layer_vectors]


def test_value_similarity_ranks_semantically_aligned_page() -> None:
    # Baseline is pinned token followed by two query tokens in each layer.
    baseline = _FakeCache(
        [
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
        ]
    )
    aligned = _FakeCache(
        [
            [[1.0, 0.0], [0.9, 0.1]],
            [[1.0, 0.0], [0.8, 0.2]],
        ]
    )
    orthogonal = _FakeCache(
        [
            [[0.0, 1.0], [0.1, 0.9]],
            [[0.0, 1.0], [0.2, 0.8]],
        ]
    )
    tokenized = SimpleNamespace(pinned_ids=torch.tensor([[1]]))
    prepared = SimpleNamespace(baseline_cache=baseline, cold_blocks=(orthogonal, aligned))

    scan = scan_kv_value_similarity(tokenized, prepared, tail_layer_count=1, top_pair_count=2)

    assert scan.query_token_count == 2
    assert scan.layer_count == 2
    assert rank_kv_similarity_page_ids(scan.scores, "all_top_pair_cosine", 1) == (1,)
    assert rank_kv_similarity_page_ids(scan.scores, "tail_query_max_cosine", 1) == (1,)


def test_value_similarity_validates_metric_and_baseline_query() -> None:
    tokenized = SimpleNamespace(pinned_ids=torch.tensor([[1]]))
    prepared = SimpleNamespace(
        baseline_cache=_FakeCache([[[1.0, 0.0]]]),
        cold_blocks=(_FakeCache([[[1.0, 0.0]]]),),
    )

    with pytest.raises(ValueError, match="no recent query"):
        scan_kv_value_similarity(tokenized, prepared)
    score_scan = SimpleNamespace(scores=[])
    with pytest.raises(ValueError, match="Unknown"):
        rank_kv_similarity_page_ids(score_scan.scores, "missing", 1)


def test_packed_max_scan_matches_reference_ranking() -> None:
    baseline = _FakeCache(
        [
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
        ]
    )
    pages = (
        _FakeCache([[[0.0, 1.0]], [[0.0, 1.0]]]),
        _FakeCache([[[1.0, 0.0]], [[1.0, 0.0]]]),
    )
    tokenized = SimpleNamespace(pinned_ids=torch.tensor([[1]]))
    prepared = SimpleNamespace(baseline_cache=baseline, cold_blocks=pages)

    reference = scan_kv_value_similarity(tokenized, prepared)
    packed_index = build_packed_cold_value_index(prepared)
    packed = scan_packed_value_max_similarity(tokenized, prepared, packed_index)

    assert packed.scores[0] == pytest.approx(reference.scores[0].all_max_cosine, abs=1e-3)
    assert packed.scores[1] == pytest.approx(reference.scores[1].all_max_cosine, abs=1e-3)
    assert rank_packed_value_page_ids(packed, 1) == (1,)
