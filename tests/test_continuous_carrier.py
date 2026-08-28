"""Unit tests for uninterrupted rotating-page carrier contracts."""

import pytest
import torch

from flash_cache.continuous_carrier import (
    exact_replay_input_token_ids,
    make_continuous_carrier_task,
    make_rotation_schedule,
)
from flash_cache.multi_hop_tasks import make_multi_hop_task


def test_continuous_prompt_forbids_page_summaries_without_answer_leakage() -> None:
    task = make_multi_hop_task(seed=8, hop_depth=3, variant=3)

    carrier_task = make_continuous_carrier_task(task)

    assert carrier_task.blocks == task.blocks
    assert task.query_message in carrier_task.query_message
    assert task.answer not in carrier_task.query_message
    assert "continuous free-form reasoning stream" in carrier_task.query_message
    assert "state the answer whenever" in carrier_task.query_message
    assert "separate page notes" in carrier_task.query_message
    assert "Page note:" not in carrier_task.query_message
    assert "until you receive" not in carrier_task.query_message


def test_rotation_schedule_supports_per_token_and_short_windows() -> None:
    assert make_rotation_schedule(3, 8, 1) == (0, 1, 2, 0, 1, 2, 0, 1)
    assert make_rotation_schedule(3, 8, 2) == (0, 0, 1, 1, 2, 2, 0, 0)


@pytest.mark.parametrize(
    ("block_count", "processed_count", "window"),
    ((0, 1, 1), (2, 0, 1), (2, 1, 0)),
)
def test_rotation_schedule_rejects_empty_dimensions(
    block_count: int,
    processed_count: int,
    window: int,
) -> None:
    with pytest.raises(ValueError):
        make_rotation_schedule(block_count, processed_count, window)


def test_exact_replay_prepends_probe_without_changing_visible_tokens() -> None:
    probe = torch.tensor([[9]])

    assert exact_replay_input_token_ids(probe, (4, 4, 7)) == (9, 4, 4, 7)


def test_exact_replay_rejects_non_scalar_probe() -> None:
    with pytest.raises(ValueError, match="shape"):
        exact_replay_input_token_ids(torch.tensor([9]), (4,))
