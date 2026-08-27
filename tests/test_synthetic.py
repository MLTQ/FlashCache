"""Unit tests for deterministic synthetic task generation."""

from flash_cache.synthetic import contains_answer_text, contains_answer_value, make_needle_task


def test_task_has_one_relevant_record_and_is_reproducible() -> None:
    first = make_needle_task(seed=19, block_count=8)
    second = make_needle_task(seed=19, block_count=8)

    assert first == second
    assert len(first.blocks) == 8
    assert sum("X-17" in block for block in first.blocks) == 1
    assert "413 psi" in first.blocks[first.relevant_block_id]
    assert "X-17" in first.query_message
    assert first.system_message in first.pinned_text


def test_answer_scoring_ignores_format_but_not_other_numbers() -> None:
    assert contains_answer_value("Based on the record, it is 413 PSI.", "413 psi")
    assert contains_answer_value("413", "413 psi")
    assert not contains_answer_value("The answer is 1413 psi.", "413 psi")
    assert not contains_answer_value("The answer is 425 psi.", "413 psi")
    assert contains_answer_text("The answer is **Elara Voss**.", "elara voss")


def test_task_supports_a_custom_target_without_duplicate_answer_values() -> None:
    task = make_needle_task(
        seed=23,
        block_count=12,
        target_identifier="R-42",
        target_pressure=250,
    )

    assert task.target_identifier == "R-42"
    assert task.target_pressure == 250
    assert task.answer == "250 psi"
    assert sum("R-42" in block for block in task.blocks) == 1
    assert sum("250 psi" in block for block in task.blocks) == 1
    assert "R-42" in task.query_message
