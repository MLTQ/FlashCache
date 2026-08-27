"""Unit tests for page-conditioned semantic carrier helpers."""

import pytest
import torch

from flash_cache.multi_hop_tasks import make_multi_hop_task
from flash_cache.semantic_carrier import (
    flatten_page_input_token_ids,
    greedy_non_control_token,
    make_semantic_carrier_task,
    run_semantic_carrier,
)


def test_semantic_prompt_preserves_corpus_without_answer_leakage() -> None:
    task = make_multi_hop_task(seed=7, hop_depth=3, variant=2)

    semantic_task = make_semantic_carrier_task(task)

    assert semantic_task.blocks == task.blocks
    assert task.query_message in semantic_task.query_message
    assert task.answer not in semantic_task.query_message
    assert "Treat every page alike" in semantic_task.query_message
    assert "Final answer" in semantic_task.query_message


def test_greedy_token_masks_control_tokens_without_format_constraints() -> None:
    logits = torch.tensor([0.0, 8.0, 7.0, 6.0])

    selected = greedy_non_control_token(logits, control_token_ids=(1, 9))

    assert selected == 2


def test_greedy_token_rejects_invalid_shape_or_fully_masked_vocabulary() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        greedy_non_control_token(torch.zeros((1, 3)), control_token_ids=())
    with pytest.raises(ValueError, match="entire vocabulary"):
        greedy_non_control_token(torch.zeros(2), control_token_ids=(0, 1))


def test_flatten_preserves_exact_page_token_order_and_duplicates() -> None:
    page_tokens = ((9, 2, 2), (4, 7), (4,))

    assert flatten_page_input_token_ids(page_tokens) == (9, 2, 2, 4, 7, 4)


def test_carrier_rejects_unknown_note_selection_mode_before_cache_access() -> None:
    with pytest.raises(ValueError, match="Unknown note selection mode"):
        run_semantic_carrier(
            model=None,
            tokenizer=None,
            source_task=None,
            tokenized_task=None,
            prepared=None,
            page_order=(0,),
            note_token_count=1,
            continuation_horizon=1,
            note_selection_mode="oracle",
        )
