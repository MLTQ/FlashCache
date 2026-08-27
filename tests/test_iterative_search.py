"""Unit tests for iterative sentinel-search prompt construction."""

import torch

from flash_cache.iterative_search import classify_gate_tokens, make_sentinel_search_task
from flash_cache.task_families import make_experiment_task


def test_sentinel_search_prompt_preserves_question_without_answer_leakage() -> None:
    task = make_experiment_task(seed=17, task_family="history_person", variant=3)
    search_task = make_sentinel_search_task(task)

    assert task.query_message in search_task.query_message
    assert task.answer not in search_task.query_message
    assert 'period character (".")' in search_task.query_message
    assert search_task.blocks == task.blocks


class _GateTokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        vocabulary = {
            1: ".",
            2: "",
            3: "**",
            4: "Answer",
            5: "The",
            6: " current",
            7: " page",
            8: " does",
            9: " not",
            10: " contain",
            11: ".\n",
        }
        text = "".join(vocabulary[token_id] for token_id in token_ids)
        return text if skip_special_tokens or 2 not in token_ids else "<special>"


def test_gate_ignores_formatting_but_stops_at_first_sentinel_or_content() -> None:
    tokenizer = _GateTokenizer()

    miss = classify_gate_tokens(torch.tensor([2, 3, 1, 4]), tokenizer, sentinel_token_id=1)
    hit = classify_gate_tokens(torch.tensor([2, 3, 4, 1]), tokenizer, sentinel_token_id=1)
    negative = classify_gate_tokens(
        torch.tensor([2, 5, 6, 7, 8, 9, 10]),
        tokenizer,
        sentinel_token_id=1,
    )
    merged_sentinel = classify_gate_tokens(
        torch.tensor([2, 11, 4]),
        tokenizer,
        sentinel_token_id=1,
    )

    assert miss == (False, "sentinel", 2)
    assert hit == (True, "content", 2)
    assert negative == (False, "negative_content", None)
    assert merged_sentinel == (False, "sentinel_surface", 1)
