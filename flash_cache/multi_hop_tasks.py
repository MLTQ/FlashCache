"""Deterministic relational chains for multi-page Flash Cache experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass

from flash_cache.synthetic import SyntheticNeedleTask

_RELATIONS = ("closest friend", "neighbor")

_CHAINS = (
    (("Shirly", "Amina", "Oren"), "tacos"),
    (("Rowan", "Elara", "Niko"), "saffron rice"),
    (("Maren", "Ivo", "Celia"), "dumplings"),
    (("Nella", "Tomas", "Rhea"), "lentil stew"),
    (("Vera", "Silas", "Jonas"), "mushroom pie"),
    (("Lina", "Corin", "Mira"), "sesame noodles"),
)

_DISTRACTORS = (
    "Relationship note: The user's brother is Calder.\n",
    "Relationship note: The user's cousin is Brina.\n",
    "Relationship note: Calder's closest friend is Lysa.\n",
    "Relationship note: Brina's closest friend is Evren.\n",
    "Relationship note: Lysa's neighbor is Dario.\n",
    "Relationship note: Evren's neighbor is Iona.\n",
    "Relationship note: Dario's closest friend is Nila.\n",
    "Relationship note: Iona's neighbor is Bram.\n",
    "Preference note: Calder's favorite food is apple pie.\n",
    "Preference note: Brina's favorite food is tomato soup.\n",
    "Preference note: Lysa's favorite food is roast squash.\n",
    "Preference note: Evren's favorite food is barley bread.\n",
    "Preference note: Dario's favorite food is lemon cake.\n",
    "Preference note: Iona's favorite food is bean stew.\n",
    "Preference note: Nila's favorite food is herb rice.\n",
    "Preference note: Bram's favorite food is plum tart.\n",
)


@dataclass(frozen=True)
class MultiHopNeedleTask(SyntheticNeedleTask):
    """A compatible needle task whose answer requires several source blocks."""

    relevant_block_ids: tuple[int, ...]
    hop_depth: int


def make_multi_hop_task(
    seed: int,
    block_count: int = 12,
    hop_depth: int = 2,
    variant: int = 0,
) -> MultiHopNeedleTask:
    """Create a shuffled relationship chain with no answer-bearing single block."""
    if not 1 <= hop_depth <= 4:
        raise ValueError("Hop depth must be between one and four source blocks")
    if block_count < hop_depth + 2:
        raise ValueError("A multi-hop task requires at least two distractor blocks")
    if block_count - hop_depth > len(_DISTRACTORS):
        raise ValueError("Block count exceeds the available unique distractors")

    names, answer = _CHAINS[variant % len(_CHAINS)]
    if hop_depth == 1:
        final_subject = names[0]
        relevant_records = [
            f"Preference note: {final_subject}'s favorite food is {answer}.\n"
        ]
    else:
        relevant_records = [f"Relationship note: The user's wife is {names[0]}.\n"]
        for relation_index in range(hop_depth - 2):
            relevant_records.append(
                f"Relationship note: {names[relation_index]}'s "
                f"{_RELATIONS[relation_index]} is {names[relation_index + 1]}.\n"
            )
        final_subject = names[hop_depth - 2]
        relevant_records.append(
            f"Preference note: {final_subject}'s favorite food is {answer}.\n"
        )

    rng = random.Random(seed)
    distractors = list(_DISTRACTORS)
    rng.shuffle(distractors)
    labeled_records: list[tuple[str, int | None]] = [
        (record, logical_step) for logical_step, record in enumerate(relevant_records)
    ]
    labeled_records.extend(
        (record, None) for record in distractors[: block_count - hop_depth]
    )
    rng.shuffle(labeled_records)
    blocks = tuple(record for record, _ in labeled_records)
    physical_by_logical = {
        logical_step: physical_index
        for physical_index, (_, logical_step) in enumerate(labeled_records)
        if logical_step is not None
    }
    relevant_block_ids = tuple(physical_by_logical[index] for index in range(hop_depth))

    if hop_depth == 1:
        relation_path = [final_subject, "favorite food"]
        query_message = f"What is {final_subject}'s favorite food?"
    else:
        relation_path = ["wife", *_RELATIONS[: hop_depth - 2], "favorite food"]
        possessive_path = "my wife's"
        for relation in _RELATIONS[: hop_depth - 2]:
            possessive_path += f" {relation}'s"
        query_message = f"What is {possessive_path} favorite food?"
    system_message = (
        "Answer using the supplied fictional personal archive. Facts may need to be combined "
        "across several separate notes. Do not assume unstated relationships."
    )
    return MultiHopNeedleTask(
        seed=seed,
        task_family="multi_hop_preference",
        target_key=" -> ".join(relation_path),
        target_identifier=None,
        target_pressure=None,
        answer_match=answer,
        system_message=system_message,
        pinned_text=f"System: {system_message}\n",
        blocks=blocks,
        query_message=query_message,
        recent_text=f"User: {query_message}\nAssistant:",
        relevant_block_id=relevant_block_ids[-1],
        answer=answer,
        relevant_block_ids=relevant_block_ids,
        hop_depth=hop_depth,
    )
