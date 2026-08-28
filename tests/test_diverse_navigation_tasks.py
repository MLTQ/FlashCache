"""Unit tests for diverse unknown-depth navigation archives."""

import pytest

from flash_cache.diverse_navigation_tasks import (
    DIVERSE_NAVIGATION_FAMILIES,
    make_diverse_navigation_task,
)
from flash_cache.synthetic import contains_answer_text


@pytest.mark.parametrize("task_family", DIVERSE_NAVIGATION_FAMILIES)
def test_diverse_navigation_task_scales_and_preserves_provenance(task_family: str) -> None:
    task = make_diverse_navigation_task(
        seed=41,
        block_count=128,
        hop_depth=4,
        task_family=task_family,
        variant=2,
    )

    assert len(task.blocks) == 128
    assert len(set(task.blocks)) == 128
    assert len(task.relevant_block_ids) == 4
    assert len(set(task.relevant_block_ids)) == 4
    assert task.answer not in task.query_message
    assert sum(contains_answer_text(block, task.answer_match) for block in task.blocks) == 1
    if task_family == "history_person":
        assert sum("closing address" in block for block in task.blocks) > 100
    elif task_family == "book_quote":
        assert sum("spoken by" in block for block in task.blocks) > 100
    else:
        assert sum("signed at" in block for block in task.blocks) > 100


def test_diverse_navigation_task_is_reproducible() -> None:
    first = make_diverse_navigation_task(11, hop_depth=3, task_family="book_quote", variant=1)
    repeated = make_diverse_navigation_task(11, hop_depth=3, task_family="book_quote", variant=1)
    reshuffled = make_diverse_navigation_task(12, hop_depth=3, task_family="book_quote", variant=1)

    assert first == repeated
    assert first.blocks != reshuffled.blocks
    assert first.answer == reshuffled.answer


def test_diverse_navigation_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        make_diverse_navigation_task(1, task_family="food")
    with pytest.raises(ValueError, match="Hop depth"):
        make_diverse_navigation_task(1, hop_depth=0)
    with pytest.raises(ValueError, match="Variant"):
        make_diverse_navigation_task(1, variant=-1)


def test_post_freeze_variant_pool_is_disjoint_from_legacy_mapping() -> None:
    legacy_answers = {
        make_diverse_navigation_task(1, task_family="history_person", variant=index).answer
        for index in range(6)
    }
    held_out_answers = {
        make_diverse_navigation_task(1, task_family="history_person", variant=index).answer
        for index in range(6, 10)
    }

    assert legacy_answers.isdisjoint(held_out_answers)
