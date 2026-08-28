"""Unit tests for query-refresh archive assembly and validation."""

from types import SimpleNamespace

import pytest
import torch

from flash_cache.dense_cache import cache_length
from flash_cache.query_refresh import assemble_cold_archive_cache, refresh_query_prefix


class _FakeLayer:
    def __init__(self, values: list[float]) -> None:
        tensor = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1)
        self.keys = tensor.clone()
        self.values = tensor.clone() + 100


class _FakeCache:
    def __init__(self, values: list[float]) -> None:
        self.layers = [_FakeLayer(values)]


def test_archive_assembly_keeps_pinned_and_pages_but_drops_stale_query() -> None:
    tokenized = SimpleNamespace(pinned_ids=torch.tensor([[1, 2]]))
    prepared = SimpleNamespace(
        baseline_cache=_FakeCache([10, 11, 90, 91, 92]),
        cold_blocks=(_FakeCache([20, 21]), _FakeCache([30, 31, 32])),
    )

    archive = assemble_cold_archive_cache(tokenized, prepared)

    assert cache_length(archive) == 7
    assert archive.layers[0].keys.reshape(-1).tolist() == [10, 11, 20, 21, 30, 31, 32]
    assert archive.layers[0].values.reshape(-1).tolist() == [
        110,
        111,
        120,
        121,
        130,
        131,
        132,
    ]


def test_archive_assembly_requires_a_cold_page() -> None:
    tokenized = SimpleNamespace(pinned_ids=torch.tensor([[1]]))
    prepared = SimpleNamespace(baseline_cache=_FakeCache([10, 90]), cold_blocks=())

    with pytest.raises(ValueError, match="at least one"):
        assemble_cold_archive_cache(tokenized, prepared)


def test_query_refresh_validates_ids_and_position_shapes_before_model_access() -> None:
    cache = _FakeCache([1])
    with pytest.raises(ValueError, match="Query IDs"):
        refresh_query_prefix(None, cache, torch.tensor([1, 2]), torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="positions"):
        refresh_query_prefix(
            None,
            cache,
            torch.tensor([[1, 2]]),
            torch.tensor([[1]]),
        )
