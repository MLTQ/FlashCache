"""Deterministic synthetic needle tasks with exact block-level provenance."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticNeedleTask:
    """Text sections and ground truth for one single-block retrieval trial."""

    seed: int
    task_family: str
    target_key: str
    target_identifier: str | None
    target_pressure: int | None
    answer_match: str
    system_message: str
    pinned_text: str
    blocks: tuple[str, ...]
    query_message: str
    recent_text: str
    relevant_block_id: int
    answer: str


def make_needle_task(
    seed: int,
    block_count: int = 12,
    target_identifier: str = "X-17",
    target_pressure: int = 413,
) -> SyntheticNeedleTask:
    """Create one shuffled valve-rating task with a single exact target record."""
    if block_count < 3:
        raise ValueError("A needle task requires at least three blocks")
    if not target_identifier.strip():
        raise ValueError("Target identifier must not be empty")
    if target_pressure <= 0:
        raise ValueError("Target pressure must be positive")

    rng = random.Random(seed)
    distractor_pressures = [175, 225, 250, 275, 300, 325, 350, 375, 425, 450, 475, 500, 525, 550, 575]
    distractor_pressures = [pressure for pressure in distractor_pressures if pressure != target_pressure]
    if block_count - 1 > len(distractor_pressures):
        raise ValueError("Not enough unique distractor pressures for the requested block count")
    rng.shuffle(distractor_pressures)
    records = [
        f"Archive record: valve D-{index:02d} has a maximum safe pressure of {pressure} psi.\n"
        for index, pressure in enumerate(distractor_pressures[: block_count - 1])
    ]
    relevant = (
        f"Archive record: valve {target_identifier} has a maximum safe pressure of "
        f"{target_pressure} psi.\n"
    )
    insertion_index = rng.randrange(block_count)
    records.insert(insertion_index, relevant)

    system_message = (
        "Answer using the archived engineering records. "
        "Match the complete valve identifier and ignore records for other valves."
    )
    query_message = (
        f"What is the maximum safe pressure for valve {target_identifier}? "
        "Reply with the number and unit."
    )
    return SyntheticNeedleTask(
        seed=seed,
        task_family="valve_pressure",
        target_key=target_identifier,
        target_identifier=target_identifier,
        target_pressure=target_pressure,
        answer_match=str(target_pressure),
        system_message=system_message,
        pinned_text=f"System: {system_message}\n",
        blocks=tuple(records),
        query_message=query_message,
        recent_text=f"User: {query_message}\nAssistant:",
        relevant_block_id=insertion_index,
        answer=f"{target_pressure} psi",
    )


def contains_answer_text(generated_text: str, answer_match: str) -> bool:
    """Match a complete expected answer phrase without requiring output formatting."""
    normalized_answer = " ".join(answer_match.casefold().split())
    if not normalized_answer:
        raise ValueError("Answer match text must not be empty")
    pattern = re.escape(normalized_answer).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){pattern}(?!\w)", generated_text.casefold()) is not None


def contains_answer_value(generated_text: str, answer: str) -> bool:
    """Accept free-form responses containing the answer's numeric value."""
    match = re.search(r"\d+(?:\.\d+)?", answer)
    if match is None:
        raise ValueError(f"Answer contains no numeric value: {answer!r}")
    return contains_answer_text(generated_text, match.group(0))
