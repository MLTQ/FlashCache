"""Unit tests for query-attention page aggregation and shortlist assembly."""

from types import SimpleNamespace

import pytest
import torch

from flash_cache.attention_shortlist import (
    aggregate_page_attention,
    assemble_selected_archive_cache,
    cold_page_spans,
    rank_page_ids,
)
from flash_cache.dense_cache import cache_length


class _FakeLayer:
    def __init__(self, values: list[float]) -> None:
        tensor = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1)
        self.keys = tensor.clone()
        self.values = tensor.clone() + 100


class _FakeCache:
    def __init__(self, values: list[float]) -> None:
        self.layers = [_FakeLayer(values)]


def test_cold_page_spans_follow_pinned_and_variable_page_lengths() -> None:
    tokenized = SimpleNamespace(
        pinned_ids=torch.tensor([[1, 2]]),
        block_ids=(torch.tensor([[3]]), torch.tensor([[4, 5, 6]])),
    )

    assert cold_page_spans(tokenized) == ((0, 2, 3), (1, 3, 6))
    assert cold_page_spans(tokenized, (1,)) == ((1, 2, 5),)
    with pytest.raises(ValueError, match="unique"):
        cold_page_spans(tokenized, (1, 1))


def test_attention_aggregation_separates_last_and_max_query_signals() -> None:
    # Shape is [batch, heads, query=2, keys=archive(5)+query(2)].
    first = torch.zeros((1, 1, 2, 7), dtype=torch.float32)
    second = torch.zeros((1, 1, 2, 7), dtype=torch.float32)
    # Page 0 is span [1:3], page 1 is [3:5]. Page 0 dominates query 0;
    # page 1 dominates the final query token in the selected last layer.
    first[0, 0, 0, 1:3] = torch.tensor([0.30, 0.20])
    first[0, 0, 1, 3:5] = torch.tensor([0.10, 0.10])
    second[0, 0, 0, 1:3] = torch.tensor([0.40, 0.20])
    second[0, 0, 1, 3:5] = torch.tensor([0.35, 0.25])

    scores = aggregate_page_attention(
        (first, second),
        ((0, 1, 3), (1, 3, 5)),
        archive_token_count=5,
        query_token_count=2,
        tail_layer_count=1,
        tail_query_token_count=1,
    )

    assert scores[0].tail_query_mass == pytest.approx(0.0)
    assert scores[1].tail_query_mass == pytest.approx(0.6)
    assert scores[0].max_query_mass == pytest.approx(0.6)
    assert scores[1].max_query_mass == pytest.approx(0.6)
    assert rank_page_ids(scores, "last_query_mass", 1) == (1,)
    assert rank_page_ids(scores, "max_query_mass", 2) == (0, 1)


def test_attention_aggregation_rejects_missing_or_misaligned_tensors() -> None:
    with pytest.raises(ValueError, match="every layer"):
        aggregate_page_attention((None,), ((0, 0, 1),), 1, 1)
    with pytest.raises(ValueError, match="query axis"):
        aggregate_page_attention(
            (torch.zeros((1, 1, 2, 4)),),
            ((0, 0, 1),),
            archive_token_count=2,
            query_token_count=1,
        )


def test_selected_archive_uses_corpus_order_and_validates_ids() -> None:
    tokenized = SimpleNamespace(pinned_ids=torch.tensor([[1, 2]]))
    prepared = SimpleNamespace(
        baseline_cache=_FakeCache([10, 11, 90]),
        cold_blocks=(_FakeCache([20]), _FakeCache([30, 31]), _FakeCache([40])),
    )

    selected = assemble_selected_archive_cache(tokenized, prepared, (2, 0))

    assert cache_length(selected) == 4
    assert selected.layers[0].keys.reshape(-1).tolist() == [10, 11, 20, 40]
    with pytest.raises(ValueError, match="unique"):
        assemble_selected_archive_cache(tokenized, prepared, (1, 1))
    with pytest.raises(ValueError, match="out of range"):
        assemble_selected_archive_cache(tokenized, prepared, (3,))
