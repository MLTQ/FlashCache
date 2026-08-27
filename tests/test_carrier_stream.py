"""Unit tests for page-conditioned carrier cache surgery."""

import torch

from flash_cache.carrier_stream import (
    classify_carrier_gate,
    make_carrier_stream_task,
    strip_flashed_page,
    visible_tokens_before_control,
)
from flash_cache.dense_cache import cache_length
from flash_cache.multi_hop_tasks import make_multi_hop_task


class _FakeLayer:
    def __init__(self) -> None:
        values = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7, 1)
        self.keys = values.clone()
        self.values = values.clone() + 10


class _FakeCache:
    def __init__(self) -> None:
        self.layers = [_FakeLayer()]


class _FakeTokenizer:
    all_special_ids = [9]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        vocabulary = {1: ".", 2: "Answer", 9: "<end>", 3: "junk"}
        return "".join(vocabulary[token_id] for token_id in token_ids)


def test_strip_flash_keeps_pinned_recent_and_appended_carrier() -> None:
    advanced = _FakeCache()

    carried = strip_flashed_page(advanced, pinned_length=2, flashed_length=2)

    assert cache_length(carried) == 5
    assert carried.layers[0].keys.reshape(-1).tolist() == [0, 1, 4, 5, 6]
    assert carried.layers[0].values.reshape(-1).tolist() == [10, 11, 14, 15, 16]


def test_carrier_prompt_preserves_blocks_and_hides_answer() -> None:
    task = make_multi_hop_task(seed=4, hop_depth=3, variant=1)

    stream_task = make_carrier_stream_task(task)

    assert stream_task.blocks == task.blocks
    assert task.query_message in stream_task.query_message
    assert task.answer not in stream_task.query_message
    assert "accumulated evidence" in stream_task.query_message


def test_gate_normalizes_wait_prose_but_keeps_answer_attempts() -> None:
    assert classify_carrier_gate("...") == (False, "sentinel")
    assert classify_carrier_gate("The accumulated evidence is insufficient.") == (
        False,
        "explicit_wait",
    )
    assert classify_carrier_gate(
        "The evidence does not provide sufficient information to determine it."
    ) == (False, "explicit_wait")
    assert classify_carrier_gate("The archive does not include those details.") == (
        False,
        "explicit_wait",
    )
    assert classify_carrier_gate("There is no direct mention of that preference.") == (
        False,
        "explicit_wait",
    )
    assert classify_carrier_gate("Her favorite food is tacos.") == (True, "answer_attempt")


def test_gate_ignores_fixed_rollout_tokens_after_control_token() -> None:
    visible_ids, visible_text = visible_tokens_before_control(
        torch.tensor([1, 9, 3]),
        _FakeTokenizer(),
    )

    assert visible_ids == [1]
    assert visible_text == "."
