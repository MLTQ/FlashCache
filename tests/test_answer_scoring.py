"""Unit tests for format-independent canonical answer likelihood summaries."""

import pytest

from flash_cache.answer_scoring import (
    AnswerChoiceScore,
    contains_asserted_answer,
    extract_archive_answer_choices,
    summarize_answer_choice_scores,
)
from flash_cache.diverse_navigation_tasks import make_diverse_navigation_task
from flash_cache.multi_hop_tasks import make_multi_hop_task


def test_archive_answer_choices_include_expected_and_unique_distractors() -> None:
    task = make_multi_hop_task(seed=68, block_count=12, hop_depth=2, variant=4)

    choices = extract_archive_answer_choices(task)

    assert choices[0] == "mushroom pie"
    assert len(choices) == len({choice.casefold() for choice in choices})
    assert len(choices) >= 3


def test_answer_summary_uses_mean_log_prob_for_rank_margin_and_probability() -> None:
    scores = (
        AnswerChoiceScore("mushroom pie", 2, -1.0, -0.5),
        AnswerChoiceScore("plum tart", 2, -2.0, -1.0),
        AnswerChoiceScore("bean stew", 2, -3.0, -1.5),
    )

    summary = summarize_answer_choice_scores(scores, "MUSHROOM PIE")

    assert summary["expected_answer_rank"] == 1
    assert summary["expected_margin_over_best_incorrect"] == pytest.approx(0.5)
    assert 0.5 < summary["expected_restricted_choice_probability"] < 1.0
    assert summary["ranking"] == ["mushroom pie", "plum tart", "bean stew"]


def test_answer_summary_requires_multiple_choices_and_exactly_one_expected() -> None:
    only = (AnswerChoiceScore("mushroom pie", 2, -1.0, -0.5),)
    with pytest.raises(ValueError, match="At least two"):
        summarize_answer_choice_scores(only, "mushroom pie")
    with pytest.raises(ValueError, match="exactly once"):
        summarize_answer_choice_scores(
            (
                AnswerChoiceScore("plum tart", 2, -1.0, -0.5),
                AnswerChoiceScore("bean stew", 2, -2.0, -1.0),
            ),
            "mushroom pie",
        )


def test_asserted_answer_rejects_incidental_multi_hop_fact_mentions() -> None:
    task = make_multi_hop_task(seed=68, block_count=12, hop_depth=2, variant=4)

    assert contains_asserted_answer(
        task,
        "Based on the records, your wife's favorite food is **mushroom pie**.",
    )
    assert contains_asserted_answer(task, "Final answer: mushroom pie")
    assert contains_asserted_answer(task, "Mushroom pie.")
    assert not contains_asserted_answer(
        task,
        "There is no direct mention of your wife's favorite food. "
        "The only food fact says Vera's favorite food is mushroom pie.",
    )


def test_asserted_answer_accepts_direct_subject_fact() -> None:
    task = make_multi_hop_task(seed=75, block_count=4, hop_depth=1, variant=4)

    assert contains_asserted_answer(task, "Vera's favorite food is mushroom pie.")


def test_asserted_answer_accepts_diverse_domain_statements() -> None:
    history = make_diverse_navigation_task(
        1,
        block_count=8,
        hop_depth=2,
        task_family="history_person",
    )
    quotation = make_diverse_navigation_task(
        2,
        block_count=8,
        hop_depth=2,
        task_family="book_quote",
    )
    place = make_diverse_navigation_task(
        3,
        block_count=8,
        hop_depth=2,
        task_family="history_place",
    )

    assert contains_asserted_answer(
        history,
        f"The closing address was delivered by {history.answer}.",
    )
    assert contains_asserted_answer(
        quotation,
        f"The quotation was spoken by {quotation.answer}.",
    )
    assert contains_asserted_answer(
        place,
        f"The treaty was signed at {place.answer}.",
    )
