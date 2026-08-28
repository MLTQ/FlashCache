"""Unit tests for latent query-capsule positions and validation."""

import pytest
import torch

from flash_cache.query_capsules import capsule_query_positions


def test_capsule_query_tails_receive_unique_adjacent_position_spans() -> None:
    page_zero = capsule_query_positions(100, 10, 3, 0, torch.device("cpu"))
    page_one = capsule_query_positions(100, 10, 3, 1, torch.device("cpu"))

    assert page_zero.tolist() == [list(range(100, 110))]
    assert page_one.tolist() == [list(range(103, 113))]
    assert page_zero[0, -3:].tolist() == [107, 108, 109]
    assert page_one[0, -3:].tolist() == [110, 111, 112]


@pytest.mark.parametrize(
    ("archive_stop", "query_count", "width", "page_index"),
    ((0, 4, 1, 0), (10, 0, 1, 0), (10, 4, 0, 0), (10, 4, 5, 0), (10, 4, 1, -1)),
)
def test_capsule_positions_reject_invalid_layouts(
    archive_stop: int,
    query_count: int,
    width: int,
    page_index: int,
) -> None:
    with pytest.raises(ValueError):
        capsule_query_positions(
            archive_stop,
            query_count,
            width,
            page_index,
            torch.device("cpu"),
        )
