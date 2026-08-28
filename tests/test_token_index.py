"""Unit tests for the answer-free cold-page token sidecar."""

import pytest
import torch

from flash_cache.token_index import (
    build_cold_token_index,
    rank_token_overlap_page_ids,
    scan_query_token_overlap,
)


def test_rare_query_token_outweighs_common_page_token() -> None:
    pages = (
        torch.tensor([[1, 2, 9]]),
        torch.tensor([[1, 3, 4]]),
        torch.tensor([[1, 5, 6]]),
        torch.tensor([[1, 7, 8]]),
    )
    index = build_cold_token_index(pages, max_document_fraction=0.75)

    scores = scan_query_token_overlap(torch.tensor([[1, 2]]), index)

    assert 1 not in index.postings
    assert rank_token_overlap_page_ids(scores, 1) == (0,)
    assert scores[0].matched_query_token_count == 1


def test_token_index_validates_shapes_and_stable_zero_score_ties() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_cold_token_index(())
    with pytest.raises(ValueError, match="shape"):
        build_cold_token_index((torch.tensor([1, 2]),))
    index = build_cold_token_index((torch.tensor([[1]]), torch.tensor([[2]])))
    scores = scan_query_token_overlap(torch.tensor([[9]]), index)
    assert rank_token_overlap_page_ids(scores, 2) == (0, 1)
