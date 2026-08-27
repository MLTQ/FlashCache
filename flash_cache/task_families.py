"""Controlled synthetic task families with numeric and categorical answers."""

from __future__ import annotations

import random
from collections.abc import Sequence

from flash_cache.synthetic import SyntheticNeedleTask, make_needle_task

TASK_FAMILIES = ("valve_pressure", "history_person", "book_quote", "history_place")

_HISTORY_PEOPLE = (
    ("Northbridge Accord of 1847", "Elara Voss"),
    ("Kestrel Assembly of 1912", "Tomas Vale"),
    ("Redhaven Congress of 1764", "Mira Solen"),
    ("Orchard Gate Convention of 1889", "Jonas Pell"),
    ("Ashcombe Summit of 1831", "Nila Hart"),
    ("Westmere Council of 1904", "Corin Ames"),
    ("Silver Fen Congress of 1788", "Lysa Dorn"),
    ("Bracken Charter Meeting of 1866", "Oren Pike"),
    ("Highwater Assembly of 1921", "Vera Quill"),
    ("Dunlow Accord of 1809", "Silas Wren"),
    ("Eastbarrow Convention of 1875", "Iona March"),
    ("Greyhaven Council of 1933", "Calder Noll"),
    ("Pinecross Summit of 1822", "Rhea Moss"),
    ("Larkfield Congress of 1897", "Evren Cole"),
    ("Stonewake Assembly of 1771", "Maren Holt"),
    ("Foxmere Accord of 1858", "Dario Wynn"),
)

_BOOK_QUOTES = (
    ("The tide remembers every borrowed name.", "Mara Venn"),
    ("A locked door is only a promise facing inward.", "Ilan Roake"),
    ("Carry the lantern low when the fog begins to listen.", "Sera Quill"),
    ("We counted the bells, but forgot to count the echoes.", "Tomas Grey"),
    ("No map admits how lonely its borders are.", "Nella Ward"),
    ("The harbor keeps what the mountain refuses.", "Orin Vale"),
    ("Winter wrote its answer beneath the blue paint.", "Celia Moss"),
    ("Every honest compass trembles once.", "Bram Sayer"),
    ("Ask the window which side of the wall is free.", "Mira Penn"),
    ("The smallest key had the longest shadow.", "Elias Thorn"),
    ("We left at dawn because midnight knew our names.", "Rhea Voss"),
    ("A quiet river can still disagree with the sea.", "Jonas Reed"),
    ("The red cup was empty before the feast began.", "Lina Crow"),
    ("Memory is a room that moves its own furniture.", "Silas Hart"),
    ("Three gulls turned when the false bell rang.", "Ada Fen"),
    ("Do not trust a staircase that arrives before you do.", "Corin Pike"),
)

_HISTORY_PLACES = (
    ("Treaty of Larkspur (1816)", "Stonebridge Hall"),
    ("Ashen Coast Compact (1872)", "Marrow Bay"),
    ("Northwind Articles (1794)", "Kestrel House"),
    ("Red Orchard Pact (1901)", "Vale Abbey"),
    ("Greywater Settlement (1838)", "Dunmere Keep"),
    ("Silver Road Convention (1884)", "Foxglove Court"),
    ("High Fen Agreement (1769)", "Cinder Hall"),
    ("Westbarrow Charter (1927)", "Orchid Station"),
    ("Pinegate Accord (1855)", "Bracken Manor"),
    ("Blue Lantern Compact (1807)", "Rookery House"),
    ("Eastmere Articles (1893)", "Willow Bastion"),
    ("Stonewake Pact (1778)", "Harbor Court"),
    ("Foxrun Settlement (1861)", "Juniper Hall"),
    ("Kestrel Shore Treaty (1914)", "Amber Lodge"),
    ("Dunlow Convention (1829)", "Mossgate Abbey"),
    ("White River Charter (1843)", "Sable House"),
)


def _assemble_categorical_task(
    *,
    seed: int,
    block_count: int,
    task_family: str,
    variant: int,
    entries: Sequence[tuple[str, str]],
    system_message: str,
    record_template: str,
    query_template: str,
) -> SyntheticNeedleTask:
    if block_count < 3:
        raise ValueError("A needle task requires at least three blocks")
    if block_count > len(entries):
        raise ValueError("Block count exceeds the available unique records")

    target_index = variant % len(entries)
    target_key, answer = entries[target_index]
    relevant = record_template.format(key=target_key, answer=answer)
    distractors = [
        record_template.format(key=key, answer=value)
        for index, (key, value) in enumerate(entries)
        if index != target_index
    ]
    rng = random.Random(seed)
    rng.shuffle(distractors)
    records = distractors[: block_count - 1]
    insertion_index = rng.randrange(block_count)
    records.insert(insertion_index, relevant)
    query_message = query_template.format(key=target_key)
    return SyntheticNeedleTask(
        seed=seed,
        task_family=task_family,
        target_key=target_key,
        target_identifier=None,
        target_pressure=None,
        answer_match=answer,
        system_message=system_message,
        pinned_text=f"System: {system_message}\n",
        blocks=tuple(records),
        query_message=query_message,
        recent_text=f"User: {query_message}\nAssistant:",
        relevant_block_id=insertion_index,
        answer=answer,
    )


def make_experiment_task(
    seed: int,
    block_count: int = 12,
    task_family: str = "valve_pressure",
    variant: int = 0,
    target_identifier: str = "X-17",
    target_pressure: int = 413,
) -> SyntheticNeedleTask:
    """Dispatch to a deterministic task family with one answer-bearing block."""
    if task_family == "valve_pressure":
        return make_needle_task(seed, block_count, target_identifier, target_pressure)
    if task_family == "history_person":
        return _assemble_categorical_task(
            seed=seed,
            block_count=block_count,
            task_family=task_family,
            variant=variant,
            entries=_HISTORY_PEOPLE,
            system_message=(
                "Answer using the supplied fictional historical archive. Match the complete event "
                "title and year; ignore records for other events."
            ),
            record_template=(
                "Historical archive: at the {key}, the closing address was delivered by {answer}.\n"
            ),
            query_template="Who delivered the closing address at the {key}?",
        )
    if task_family == "book_quote":
        return _assemble_categorical_task(
            seed=seed,
            block_count=block_count,
            task_family=task_family,
            variant=variant,
            entries=_BOOK_QUOTES,
            system_message=(
                "Answer using the supplied fictional literary index for the novel The Glass Harbor. "
                "Match the complete quoted line and ignore entries for other lines."
            ),
            record_template=(
                'Literary index: in The Glass Harbor, the line "{key}" is spoken by {answer}.\n'
            ),
            query_template='In The Glass Harbor, who says the line "{key}"?',
        )
    if task_family == "history_place":
        return _assemble_categorical_task(
            seed=seed,
            block_count=block_count,
            task_family=task_family,
            variant=variant,
            entries=_HISTORY_PLACES,
            system_message=(
                "Answer using the supplied fictional treaty archive. Match the complete treaty name "
                "and year; ignore records for other treaties."
            ),
            record_template="Treaty archive: the {key} was signed at {answer}.\n",
            query_template="Where was the {key} signed?",
        )
    raise ValueError(f"Unknown task family: {task_family}")
