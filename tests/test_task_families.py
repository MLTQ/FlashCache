"""Unit tests for numeric and categorical synthetic task families."""

import pytest

from flash_cache.synthetic import contains_answer_text
from flash_cache.task_families import TASK_FAMILIES, make_experiment_task


@pytest.mark.parametrize("task_family", TASK_FAMILIES)
def test_each_family_is_reproducible_with_one_answer_bearing_block(task_family: str) -> None:
    first = make_experiment_task(seed=31, block_count=12, task_family=task_family, variant=2)
    second = make_experiment_task(seed=31, block_count=12, task_family=task_family, variant=2)

    assert first == second
    assert first.task_family == task_family
    assert len(first.blocks) == 12
    assert first.target_key in first.query_message
    assert sum(first.answer_match in block for block in first.blocks) == 1
    assert first.answer_match in first.blocks[first.relevant_block_id]


def test_categorical_answer_matching_ignores_format_and_case() -> None:
    assert contains_answer_text("The speaker was **MARA   VENN**.", "Mara Venn")
    assert not contains_answer_text("The speaker was Mara Vennick.", "Mara Venn")
