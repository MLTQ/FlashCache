"""Unit tests for deterministic multi-page relationship chains."""

from flash_cache.multi_hop_tasks import make_multi_hop_task
from flash_cache.synthetic import contains_answer_text


def test_multi_hop_task_requires_every_chain_record() -> None:
    task = make_multi_hop_task(seed=19, block_count=12, hop_depth=4, variant=0)

    assert len(task.blocks) == 12
    assert len(task.relevant_block_ids) == 4
    assert len(set(task.relevant_block_ids)) == 4
    assert task.answer not in task.query_message
    assert sum(contains_answer_text(block, task.answer_match) for block in task.blocks) == 1
    assert "Shirly" in task.blocks[task.relevant_block_ids[0]]
    assert "tacos" in task.blocks[task.relevant_block_ids[-1]]


def test_multi_hop_task_is_reproducible_but_seed_changes_placement() -> None:
    first = make_multi_hop_task(seed=21, hop_depth=3, variant=2)
    repeated = make_multi_hop_task(seed=21, hop_depth=3, variant=2)
    reshuffled = make_multi_hop_task(seed=22, hop_depth=3, variant=2)

    assert first == repeated
    assert first.blocks != reshuffled.blocks
    assert first.answer == reshuffled.answer


def test_depth_one_is_a_direct_carrier_retention_calibration() -> None:
    task = make_multi_hop_task(seed=8, block_count=4, hop_depth=1, variant=0)

    assert task.query_message == "What is Shirly's favorite food?"
    assert len(task.relevant_block_ids) == 1
    assert task.answer in task.blocks[task.relevant_block_ids[0]]


def test_large_archive_extends_distractors_without_duplicate_records() -> None:
    task = make_multi_hop_task(seed=31, block_count=128, hop_depth=2, variant=1)

    assert len(task.blocks) == 128
    assert len(set(task.blocks)) == 128
    assert any("FillerPerson" in block for block in task.blocks)
    assert sum(contains_answer_text(block, task.answer_match) for block in task.blocks) == 1
