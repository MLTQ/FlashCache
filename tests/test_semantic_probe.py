"""Unit tests for answer-free semantic relevance probes."""

import torch

from flash_cache.semantic_probe import (
    binary_token_set_metrics,
    contains_normalized_key,
    make_provenance_probe_task,
    make_relevance_probe_task,
    single_token_variant_ids,
)
from flash_cache.task_families import make_experiment_task


class _ToyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        vocabulary = {"YES": [1], "Yes": [2], "NO": [3], "No": [4], "not one": [5, 6]}
        return {"input_ids": vocabulary[text]}


def test_relevance_probe_preserves_blocks_without_leaking_answer() -> None:
    task = make_experiment_task(seed=9, task_family="book_quote", variant=2)
    probe = make_relevance_probe_task(task)

    assert probe.blocks == task.blocks
    assert probe.target_key in probe.query_message
    assert task.answer not in probe.query_message
    assert "YES or NO" in probe.query_message


def test_provenance_probe_leaks_neither_target_key_nor_answer() -> None:
    task = make_experiment_task(seed=11, task_family="history_place", variant=4)
    probe = make_provenance_probe_task(task)

    assert probe.blocks == task.blocks
    assert task.target_key not in probe.query_message
    assert task.answer not in probe.query_message
    assert "subject key" in probe.query_message


def test_normalized_key_matching_ignores_punctuation_but_not_missing_tokens() -> None:
    assert contains_normalized_key('"Red Orchard Pact" (1901)', "Red Orchard Pact (1901)")
    assert not contains_normalized_key("Red Orchard Pact", "Red Orchard Pact (1901)")


def test_binary_probe_prefers_the_higher_probability_token_set() -> None:
    tokenizer = _ToyTokenizer()
    yes_ids = single_token_variant_ids(tokenizer, ("YES", "Yes", "not one"))
    no_ids = single_token_variant_ids(tokenizer, ("NO", "No"))
    logits = torch.tensor([0.0, 4.0, 3.0, 0.0, -1.0, -2.0, -2.0])

    metrics = binary_token_set_metrics(logits, yes_ids, no_ids)

    assert yes_ids == (1, 2)
    assert metrics["semantic_yes_no_log_odds"] > 0
    assert metrics["semantic_yes_probability_normalized"] > 0.5
