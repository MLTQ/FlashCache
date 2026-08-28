"""Unit tests for answer-free iterative navigation prompts and parsing."""

import pytest

from flash_cache.iterative_navigation import (
    NAVIGATION_SYSTEM_MESSAGE,
    canonicalize_lookup_entities,
    make_navigation_repair_user_message,
    make_navigation_user_message,
    navigation_decision_needs_target_repair,
    parse_navigation_decision,
    replace_task_question,
)
from flash_cache.multi_hop_tasks import make_multi_hop_task


def test_navigation_parser_accepts_labels_and_format_fallbacks() -> None:
    assert parse_navigation_decision(
        "LOOKUP: What is Vera's favorite food?"
    ).content == "What is Vera's favorite food?"
    assert parse_navigation_decision("ANSWER: mushroom pie").kind == "answer"
    assert parse_navigation_decision("What is Silas's favorite food?").kind == "lookup"
    assert parse_navigation_decision("Mushroom pie.").kind == "answer"
    assert parse_navigation_decision("   ").kind == "invalid"


def test_navigation_parser_accepts_grounded_mislabeled_final_fact_only() -> None:
    grounded = parse_navigation_decision(
        "LOOKUP: Vera's favorite food is mushroom pie.",
        current_question="The user's wife is Vera. Vera's favorite food is not listed.",
    )
    ungrounded = parse_navigation_decision(
        "LOOKUP: Vera's favorite food is mushroom pie.",
        current_question="What is my wife's favorite food?",
    )
    negative = parse_navigation_decision(
        "LOOKUP: Vera's favorite food is not listed.",
        current_question="The user's wife is Vera.",
    )

    assert grounded.kind == "answer"
    assert ungrounded.kind == "lookup"
    assert negative.kind == "lookup"


def test_navigation_message_requires_question_and_pages() -> None:
    message = make_navigation_user_message(
        "What is my wife's favorite food?",
        ("Relationship note: The user's wife is Vera.\n",),
    )

    assert "Current question" in message
    assert "wife is Vera" in message
    with pytest.raises(ValueError, match="question"):
        make_navigation_user_message("", ("note",))
    with pytest.raises(ValueError, match="page"):
        make_navigation_user_message("Question?", ())


def test_rewritten_task_preserves_archive_and_changes_only_query() -> None:
    task = make_multi_hop_task(seed=68, block_count=12, hop_depth=2, variant=4)

    rewritten = replace_task_question(task, "What is Vera's favorite food?")

    assert rewritten.blocks == task.blocks
    assert rewritten.pinned_text == task.pinned_text
    assert rewritten.query_message == "What is Vera's favorite food?"
    assert rewritten.recent_text.endswith("What is Vera's favorite food?\nAssistant:")


def test_lookup_entity_canonicalization_uses_only_selected_source_values() -> None:
    corrected = canonicalize_lookup_entities(
        "What is Shirley's closest friend's favorite food?",
        ("Relationship note: The user's wife is Shirly.\n",),
    )
    unchanged = canonicalize_lookup_entities(
        "What is Shirley's closest friend's favorite food?",
        ("Relationship note: The user's wife is Vera.\n",),
    )

    assert corrected == "What is Shirly's closest friend's favorite food?"
    assert unchanged == "What is Shirley's closest friend's favorite food?"


def test_navigation_repair_requires_previous_output_and_forbids_repeat() -> None:
    message = make_navigation_repair_user_message(
        "What is Niko's favorite food?",
        ("Preference note: Niko's favorite food is saffron rice.\n",),
        "LOOKUP: What is Niko's favorite food?",
    )

    assert "made no progress" in message
    assert "Do not repeat" in message
    assert "who, where, or what" in message
    with pytest.raises(ValueError, match="Repeated output"):
        make_navigation_repair_user_message("Question?", ("note",), "")


def test_navigation_instruction_preserves_targets_across_domains() -> None:
    assert "Never answer with an intermediate" in NAVIGATION_SYSTEM_MESSAGE
    assert "Where was the Ashen Compact signed?" in NAVIGATION_SYSTEM_MESSAGE
    assert "Who says the line" in NAVIGATION_SYSTEM_MESSAGE


def test_target_repair_detects_quoted_answer_to_who_question() -> None:
    quoted = parse_navigation_decision('ANSWER: "A quiet river."')
    person = parse_navigation_decision("ANSWER: Mara Venn")

    assert navigation_decision_needs_target_repair(
        quoted,
        "In The Glass Harbor, who says Sera's selected line?",
    )
    assert not navigation_decision_needs_target_repair(
        person,
        "Who delivered the closing address?",
    )
    assert not navigation_decision_needs_target_repair(
        quoted,
        "What is Sera's selected line?",
    )
