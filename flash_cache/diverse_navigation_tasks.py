"""Deterministic multi-hop archives with names, quotations, and places as answers."""

from __future__ import annotations

import random
from dataclasses import dataclass

from flash_cache.synthetic import SyntheticNeedleTask


DIVERSE_NAVIGATION_FAMILIES = ("history_person", "book_quote", "history_place")

_HISTORY_VARIANTS = (
    ("Northbridge Accord of 1847", "Elara Voss", "Mira Penn", "Tomas Grey", "Vault Lark"),
    ("Kestrel Assembly of 1912", "Tomas Vale", "Nella Ward", "Ilan Roake", "Vault Cinder"),
    ("Redhaven Congress of 1764", "Mira Solen", "Corin Pike", "Rhea Voss", "Vault Juniper"),
    ("Orchard Gate Convention of 1889", "Jonas Pell", "Ada Fen", "Silas Hart", "Vault Amber"),
    ("Ashcombe Summit of 1831", "Nila Hart", "Orin Vale", "Celia Moss", "Vault Sable"),
    ("Westmere Council of 1904", "Corin Ames", "Lina Crow", "Mara Venn", "Vault Cobalt"),
    ("Silver Fen Congress of 1788", "Lysa Dorn", "Elias Thorn", "Rhea Moss", "Vault Ochre"),
    ("Bracken Charter Meeting of 1866", "Oren Pike", "Iona March", "Bram Sayer", "Vault Indigo"),
)

_QUOTE_VARIANTS = (
    ("The tide remembers every borrowed name.", "Mara Venn", "Sera Quill", "Bram Sayer", "Room Cobalt"),
    ("A locked door is only a promise facing inward.", "Ilan Roake", "Nella Ward", "Elias Thorn", "Room Saffron"),
    ("Carry the lantern low when the fog begins to listen.", "Sera Quill", "Mira Penn", "Jonas Reed", "Room Juniper"),
    ("We counted the bells, but forgot to count the echoes.", "Tomas Grey", "Celia Moss", "Ada Fen", "Room Indigo"),
    ("No map admits how lonely its borders are.", "Nella Ward", "Orin Vale", "Mira Solen", "Room Sable"),
    ("The harbor keeps what the mountain refuses.", "Orin Vale", "Lina Crow", "Calder Noll", "Room Amber"),
    ("Winter wrote its answer beneath the blue paint.", "Celia Moss", "Rhea Voss", "Jonas Reed", "Room Ochre"),
    ("Every honest compass trembles once.", "Bram Sayer", "Iona March", "Evren Cole", "Room Jade"),
)

_PLACE_VARIANTS = (
    ("Treaty of Larkspur (1816)", "Stonebridge Hall", "Rhea Moss", "Oren Pike", "Shelf Cobalt"),
    ("Ashen Coast Compact (1872)", "Marrow Bay", "Vera Quill", "Silas Wren", "Shelf Juniper"),
    ("Northwind Articles (1794)", "Kestrel House", "Iona March", "Calder Noll", "Shelf Amber"),
    ("Red Orchard Pact (1901)", "Vale Abbey", "Evren Cole", "Maren Holt", "Shelf Indigo"),
    ("Greywater Settlement (1838)", "Dunmere Keep", "Lysa Dorn", "Mara Venn", "Shelf Sable"),
    ("Silver Road Convention (1884)", "Foxglove Court", "Celia Moss", "Oren Pike", "Shelf Cobalt"),
    ("High Fen Agreement (1769)", "Cinder Hall", "Tomas Vale", "Rhea Voss", "Shelf Ochre"),
    ("Westbarrow Charter (1927)", "Orchid Station", "Nella Ward", "Calder Noll", "Shelf Jade"),
)


def _variant_entry(entries: tuple[tuple[str, ...], ...], variant: int) -> tuple[str, ...]:
    """Preserve variants 0–5 from earlier phases and reserve 6+ for unseen content."""
    if variant < 0:
        raise ValueError("Variant cannot be negative")
    if variant < 6:
        return entries[variant % 4]
    return entries[4 + (variant - 6) % (len(entries) - 4)]


@dataclass(frozen=True)
class DiverseNavigationTask(SyntheticNeedleTask):
    """A synthetic archive task with relevant pages ordered by logical hop."""

    relevant_block_ids: tuple[int, ...]
    hop_depth: int


def _history_chain(variant: int, hop_depth: int) -> tuple[list[str], str, str, str]:
    event, answer, curator, archivist, vault = _variant_entry(_HISTORY_VARIANTS, variant)
    final = f"Historical archive: at the {event}, the closing address was delivered by {answer}.\n"
    if hop_depth == 1:
        return [final], f"Who delivered the closing address at the {event}?", answer, event
    featured = f"Cross-reference: {curator}'s featured event is the {event}.\n"
    if hop_depth == 2:
        question = f"Who delivered the closing address at {curator}'s featured event?"
        return [featured, final], question, answer, curator
    if hop_depth == 3:
        lead = f"Registry note: {vault}'s lead curator is {curator}.\n"
        question = f"Who delivered the closing address at {vault}'s lead curator's featured event?"
        return [lead, featured, final], question, answer, vault
    first = f"Registry note: {vault}'s east-wing archivist is {archivist}.\n"
    mentor = f"Mentorship note: {archivist}'s mentor is {curator}.\n"
    question = (
        f"Who delivered the closing address at {vault}'s east-wing archivist's mentor's "
        "featured event?"
    )
    return [first, mentor, featured, final], question, answer, vault


def _quote_chain(variant: int, hop_depth: int) -> tuple[list[str], str, str, str]:
    quote, answer, reviewer, editor, room = _variant_entry(_QUOTE_VARIANTS, variant)
    final = f'Literary index: in The Glass Harbor, the line "{quote}" is spoken by {answer}.\n'
    if hop_depth == 1:
        return [final], f'In The Glass Harbor, who says the line "{quote}"?', answer, quote
    selected = f'Reading list: {reviewer}\'s selected line is "{quote}".\n'
    if hop_depth == 2:
        return [selected, final], f"In The Glass Harbor, who says {reviewer}'s selected line?", answer, reviewer
    if hop_depth == 3:
        lead = f"Registry note: {room}'s lead reviewer is {reviewer}.\n"
        question = f"In The Glass Harbor, who says {room}'s lead reviewer's selected line?"
        return [lead, selected, final], question, answer, room
    first = f"Registry note: {room}'s west-desk editor is {editor}.\n"
    mentor = f"Mentorship note: {editor}'s mentor is {reviewer}.\n"
    question = f"In The Glass Harbor, who says {room}'s west-desk editor's mentor's selected line?"
    return [first, mentor, selected, final], question, answer, room


def _place_chain(variant: int, hop_depth: int) -> tuple[list[str], str, str, str]:
    treaty, answer, cataloguer, archivist, shelf = _variant_entry(_PLACE_VARIANTS, variant)
    final = f"Treaty archive: the {treaty} was signed at {answer}.\n"
    if hop_depth == 1:
        return [final], f"Where was the {treaty} signed?", answer, treaty
    featured = f"Catalog note: {cataloguer}'s featured treaty is the {treaty}.\n"
    if hop_depth == 2:
        return [featured, final], f"Where was {cataloguer}'s featured treaty signed?", answer, cataloguer
    if hop_depth == 3:
        lead = f"Registry note: {shelf}'s lead cataloguer is {cataloguer}.\n"
        question = f"Where was {shelf}'s lead cataloguer's featured treaty signed?"
        return [lead, featured, final], question, answer, shelf
    first = f"Registry note: {shelf}'s east-wing archivist is {archivist}.\n"
    mentor = f"Mentorship note: {archivist}'s mentor is {cataloguer}.\n"
    question = f"Where was {shelf}'s east-wing archivist's mentor's featured treaty signed?"
    return [first, mentor, featured, final], question, answer, shelf


def _filler_records(task_family: str, count: int) -> list[str]:
    records: list[str] = []
    for index in range(count):
        if task_family == "history_person":
            records.append(
                f"History: at FillerEvent{index:03d}, closing address delivered by "
                f"FillerSpeaker{index:03d}.\n"
            )
        elif task_family == "book_quote":
            records.append(
                f'Literature: line "Filler quote {index:03d}." spoken by FillerReader{index:03d}.\n'
            )
        elif task_family == "history_place":
            records.append(
                f"Treaty: FillerCompact{index:03d} signed at FillerHall{index:03d}.\n"
            )
        else:
            raise ValueError(f"Unknown diverse navigation family: {task_family}")
    return records


def make_diverse_navigation_task(
    seed: int,
    block_count: int = 128,
    hop_depth: int = 2,
    task_family: str = "history_person",
    variant: int = 0,
) -> DiverseNavigationTask:
    """Create a shuffled unknown-depth chain in a non-food answer domain."""
    if task_family not in DIVERSE_NAVIGATION_FAMILIES:
        raise ValueError(f"Unknown diverse navigation family: {task_family}")
    if not 1 <= hop_depth <= 4:
        raise ValueError("Hop depth must be between one and four source blocks")
    if block_count < hop_depth + 2:
        raise ValueError("A diverse navigation task requires at least two distractor blocks")

    chain_builder = {
        "history_person": _history_chain,
        "book_quote": _quote_chain,
        "history_place": _place_chain,
    }[task_family]
    relevant_records, question, answer, target_key = chain_builder(variant, hop_depth)
    distractors = _filler_records(task_family, block_count - hop_depth)
    rng = random.Random(seed)
    rng.shuffle(distractors)
    labeled_records: list[tuple[str, int | None]] = [
        (record, logical_step) for logical_step, record in enumerate(relevant_records)
    ]
    labeled_records.extend((record, None) for record in distractors)
    rng.shuffle(labeled_records)
    blocks = tuple(record for record, _ in labeled_records)
    physical_by_logical = {
        logical_step: physical_index
        for physical_index, (_, logical_step) in enumerate(labeled_records)
        if logical_step is not None
    }
    relevant_ids = tuple(physical_by_logical[index] for index in range(hop_depth))
    system_message = (
        "Answer using only the supplied fictional archive. Facts may need to be followed across "
        "separate notes. Do not use outside knowledge or assume unstated relationships."
    )
    return DiverseNavigationTask(
        seed=seed,
        task_family=task_family,
        target_key=target_key,
        target_identifier=None,
        target_pressure=None,
        answer_match=answer,
        system_message=system_message,
        pinned_text=f"System: {system_message}\n",
        blocks=blocks,
        query_message=question,
        recent_text=f"User: {question}\nAssistant:",
        relevant_block_id=relevant_ids[-1],
        answer=answer,
        relevant_block_ids=relevant_ids,
        hop_depth=hop_depth,
    )
