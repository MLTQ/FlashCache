"""Evaluate canonical answer likelihood without imposing a generation format."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from flash_cache.probing import rollout
from flash_cache.synthetic import SyntheticNeedleTask, contains_answer_text


@dataclass(frozen=True)
class AnswerChoiceScore:
    """Teacher-forced likelihood for one candidate answer phrase."""

    answer: str
    token_count: int
    sequence_log_prob: float
    mean_token_log_prob: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def extract_archive_answer_choices(task: SyntheticNeedleTask) -> tuple[str, ...]:
    """Collect the expected answer and same-domain favorite-food distractors from page text."""
    choices = {task.answer.casefold(): task.answer}
    pattern = re.compile(r"favorite food is\s+([^\n.]+)", re.IGNORECASE)
    for block in task.blocks:
        for match in pattern.finditer(block):
            value = " ".join(match.group(1).strip().split())
            choices.setdefault(value.casefold(), value)
    return tuple(sorted(choices.values(), key=lambda value: (value.casefold() != task.answer.casefold(), value)))


def contains_asserted_answer(task: SyntheticNeedleTask, generated_text: str) -> bool:
    """Require an answer assertion rather than an incidental mention of the expected value."""
    if not contains_answer_text(generated_text, task.answer_match):
        return False
    normalized = " ".join(generated_text.casefold().split())
    answer = re.escape(" ".join(task.answer_match.casefold().split())).replace(r"\ ", r"\s+")
    decoration = r"(?:[*_`]+)?"
    strong_patterns = (
        rf"(?:final\s+)?answer\s*(?:is|:)\s*{decoration}[^.!?\n]{{0,60}}?{answer}",
        rf"therefore[^.!?\n]{{0,80}}?{answer}",
        rf"(?:your|my|the\s+user(?:'s)?)\s+wife(?:'s)?\s+favorite\s+food\s+is\s+{decoration}{answer}",
    )
    if any(re.search(pattern, normalized) for pattern in strong_patterns):
        return True

    domain_assertions = (
        rf"closing\s+address\s+was\s+delivered\s+by\s+{decoration}{answer}",
        rf"(?:line|quotation)[^.!?\n]{{0,120}}?spoken\s+by\s+{decoration}{answer}",
        rf"(?:treaty|compact|articles|pact|charter)[^.!?\n]{{0,120}}?signed\s+at\s+{decoration}{answer}",
        rf"maximum\s+safe\s+pressure[^.!?\n]{{0,80}}?{decoration}{answer}",
    )
    if any(re.search(pattern, normalized) for pattern in domain_assertions):
        return True

    direct_subject = re.search(r"what is (.+?)'s favorite food\?", task.query_message.casefold())
    if direct_subject is not None:
        subject = re.escape(direct_subject.group(1).strip())
        direct_pattern = (
            rf"{subject}'s\s+favorite\s+food\s+is\s+{decoration}{answer}"
        )
        if re.search(direct_pattern, normalized):
            return True

    stripped = re.sub(r"^[\s*_`]+|[\s*_`.!]+$", "", normalized)
    if re.fullmatch(answer, stripped):
        return True
    return False


def summarize_answer_choice_scores(
    scores: Sequence[AnswerChoiceScore],
    expected_answer: str,
) -> dict[str, Any]:
    """Summarize expected-answer rank, margins, and restricted-choice probability."""
    if len(scores) < 2:
        raise ValueError("At least two answer choices are required")
    expected_matches = [
        score for score in scores if score.answer.casefold() == expected_answer.casefold()
    ]
    if len(expected_matches) != 1:
        raise ValueError("Expected answer must occur exactly once in answer choices")
    expected = expected_matches[0]
    ranked = sorted(scores, key=lambda score: (-score.mean_token_log_prob, score.answer.casefold()))
    best_incorrect = max(
        score.mean_token_log_prob
        for score in scores
        if score.answer.casefold() != expected_answer.casefold()
    )
    maximum = max(score.mean_token_log_prob for score in scores)
    weights = [math.exp(score.mean_token_log_prob - maximum) for score in scores]
    denominator = sum(weights)
    expected_index = list(scores).index(expected)
    return {
        "expected_answer": expected.answer,
        "expected_answer_rank": ranked.index(expected) + 1,
        "choice_count": len(scores),
        "expected_sequence_log_prob": expected.sequence_log_prob,
        "expected_mean_token_log_prob": expected.mean_token_log_prob,
        "expected_margin_over_best_incorrect": expected.mean_token_log_prob - best_incorrect,
        "expected_restricted_choice_probability": weights[expected_index] / denominator,
        "ranking": [score.answer for score in ranked],
        "choices": [score.to_dict() for score in scores],
    }


def score_answer_choices(
    model: Any,
    tokenizer: Any,
    cache: Any,
    probe_token: torch.Tensor,
    probe_position: int,
    answer_choices: Sequence[str],
    expected_answer: str,
    answer_cue: str = "\nFinal answer:",
) -> dict[str, Any]:
    """Teacher-force answer phrases after one shared cue from an immutable cache state."""
    if len({choice.casefold() for choice in answer_choices}) != len(answer_choices):
        raise ValueError("Answer choices must be unique ignoring case")
    cue_ids = tokenizer(
        answer_cue,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].to(probe_token.device)
    if cue_ids.numel() < 1:
        raise ValueError("Answer cue must tokenize to a nonempty sequence")
    scores: list[AnswerChoiceScore] = []
    for choice in answer_choices:
        answer_ids = tokenizer(
            " " + choice,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].to(probe_token.device)
        if answer_ids.numel() < 1:
            raise ValueError(f"Answer choice tokenized to an empty sequence: {choice!r}")
        forced_ids = torch.cat((cue_ids, answer_ids))
        forced = rollout(
            model,
            cache,
            probe_token,
            probe_position,
            int(forced_ids.shape[0]),
            forced_tokens=forced_ids,
        )
        answer_logits = forced.logits[int(cue_ids.shape[0]) :]
        log_probs = torch.log_softmax(answer_logits.float(), dim=-1)
        selected = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
        sequence_log_prob = float(selected.sum().item())
        scores.append(
            AnswerChoiceScore(
                answer=choice,
                token_count=int(answer_ids.shape[0]),
                sequence_log_prob=sequence_log_prob,
                mean_token_log_prob=sequence_log_prob / int(answer_ids.shape[0]),
            )
        )
    summary = summarize_answer_choice_scores(scores, expected_answer)
    summary["answer_cue"] = answer_cue
    return summary
