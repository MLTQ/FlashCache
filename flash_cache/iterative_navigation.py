"""Prompt and parse answer-free model decisions for iterative archive navigation."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from flash_cache.synthetic import SyntheticNeedleTask


NAVIGATION_SYSTEM_MESSAGE = (
    "Navigate a fictional archive one fact at a time using only the retrieved notes. Preserve "
    "exactly what the current question ultimately asks (who, where, or what). Never answer with "
    "an intermediate person, event, treaty, or quotation. If a note resolves a nested phrase, "
    "substitute its right-hand value into the full question and reply LOOKUP. If it directly "
    "states the requested final value, reply ANSWER with that value. Do not guess or add facts. "
    "Use one line: LOOKUP: <full rewritten question> or ANSWER: <final value>. Examples: "
    "Question 'What is my manager's neighbor's favorite color?' plus 'The user's manager is "
    "Dana' becomes 'LOOKUP: What is Dana's neighbor's favorite color?'. Question 'Where was "
    "Vera's featured treaty signed?' plus 'Vera's featured treaty is the Ashen Compact' becomes "
    "'LOOKUP: Where was the Ashen Compact signed?'. That question plus 'The Ashen Compact was "
    "signed at Marrow Bay' becomes 'ANSWER: Marrow Bay'. Question 'Who says Sera's selected "
    "line?' plus 'Sera's selected line is \"A quiet river\"' becomes 'LOOKUP: Who says the line "
    "\"A quiet river\"?'."
)


@dataclass(frozen=True)
class NavigationDecision:
    """One model-selected next action that never consults the expected answer."""

    kind: str
    content: str
    raw_text: str


def make_navigation_user_message(current_question: str, page_texts: tuple[str, ...]) -> str:
    """Render the current question and selected exact source notes for one navigation step."""
    question = " ".join(current_question.strip().split())
    if not question:
        raise ValueError("Current question must not be empty")
    if not page_texts:
        raise ValueError("At least one retrieved page is required")
    notes = "".join(f"- {text.strip()}\n" for text in page_texts)
    return f"Current question: {question}\nRetrieved notes:\n{notes}Next action:"


def make_navigation_repair_user_message(
    current_question: str,
    page_texts: tuple[str, ...],
    repeated_output: str,
) -> str:
    """Ask for one different action after a no-progress repeated lookup."""
    previous = " ".join(repeated_output.strip().split())
    if not previous:
        raise ValueError("Repeated output must not be empty")
    return (
        make_navigation_user_message(current_question, page_texts)
        + f"\nYour previous response made no progress: {previous}\n"
        + "Re-read the note and preserve whether the question asks who, where, or what. Do not "
        + "answer with an intermediate value. If the note states the requested final value, reply "
        + "ANSWER: <exact right-hand value>. Otherwise substitute the note's right-hand value into "
        + "the full question and reply with a different LOOKUP. Do not repeat."
    )


def navigation_decision_needs_target_repair(
    decision: NavigationDecision,
    current_question: str,
) -> bool:
    """Detect an answer-free type mismatch such as returning a quote to a who-question."""
    if decision.kind != "answer":
        return False
    question = " ".join(current_question.casefold().split())
    content = decision.content.strip()
    asks_who = bool(re.match(r"^(?:in\s+[^,]+,\s*)?who\b", question))
    quoted_intermediate = bool(re.match(r"^[\"“].+[\"”][.!?]?$", content))
    return asks_who and quoted_intermediate


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def canonicalize_lookup_entities(
    lookup_text: str,
    page_texts: tuple[str, ...],
) -> str:
    """Snap one-edit generated entity spellings to exact values in retrieved source notes."""
    source_values: list[str] = []
    for page_text in page_texts:
        source_values.extend(
            match.group(1).strip()
            for match in re.finditer(r"\bis\s+([^\n.]+)", page_text, re.IGNORECASE)
        )
    candidate_words = {
        word
        for value in source_values
        for word in re.findall(r"[A-Za-z][A-Za-z-]+", value)
        if len(word) >= 4
    }
    result = lookup_text
    for generated_word in set(re.findall(r"[A-Za-z][A-Za-z-]+", lookup_text)):
        if any(generated_word.casefold() == candidate.casefold() for candidate in candidate_words):
            continue
        matches = [
            candidate
            for candidate in candidate_words
            if generated_word[0].casefold() == candidate[0].casefold()
            and _edit_distance(generated_word.casefold(), candidate.casefold()) <= 1
        ]
        if len(matches) == 1:
            result = re.sub(rf"\b{re.escape(generated_word)}\b", matches[0], result)
    return result


def parse_navigation_decision(
    generated_text: str,
    current_question: str | None = None,
) -> NavigationDecision:
    """Parse labels leniently, falling back to questions as lookups and prose as answers."""
    raw = generated_text.strip()
    if not raw:
        return NavigationDecision(kind="invalid", content="", raw_text=generated_text)
    labeled = re.search(r"\b(LOOKUP|ANSWER)\s*:\s*([^\n]+)", raw, re.IGNORECASE)
    if labeled is not None:
        kind = labeled.group(1).casefold()
        content = labeled.group(2).strip().strip("*_`")
        if kind == "lookup" and current_question is not None and "?" not in content:
            normalized_content = " ".join(content.casefold().split())
            negative = re.search(
                r"\b(?:not|unknown|missing|cannot|can't|no\s+information)\b",
                normalized_content,
            )
            final_fact = re.search(
                r"\b([a-z][\w-]*)'s\s+favorite\s+food\s+is\s+(.+)",
                normalized_content,
            )
            if (
                negative is None
                and final_fact is not None
                and re.search(
                    rf"\b{re.escape(final_fact.group(1))}\b",
                    current_question.casefold(),
                )
            ):
                kind = "answer"
        return NavigationDecision(
            kind=kind if content else "invalid",
            content=content,
            raw_text=generated_text,
        )
    question = re.search(r"(What\s+is\s+[^\n?]+\?)", raw, re.IGNORECASE)
    if question is not None:
        return NavigationDecision(
            kind="lookup",
            content=" ".join(question.group(1).split()),
            raw_text=generated_text,
        )
    first_line = raw.splitlines()[0].strip().strip("*_`")
    return NavigationDecision(
        kind="answer" if first_line else "invalid",
        content=first_line,
        raw_text=generated_text,
    )


def replace_task_question(
    task: SyntheticNeedleTask,
    current_question: str,
) -> SyntheticNeedleTask:
    """Preserve pinned/archive sections while replacing only the recent user question."""
    question = " ".join(current_question.strip().split())
    if not question:
        raise ValueError("Current question must not be empty")
    return replace(
        task,
        query_message=question,
        recent_text=f"User: {question}\nAssistant:",
    )
