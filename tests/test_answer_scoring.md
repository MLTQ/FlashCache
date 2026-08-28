# `test_answer_scoring.py`

## Purpose

Protects archive-derived answer-choice extraction and the evaluation-only likelihood summary used when free-form substring correctness is ambiguous.

## Coverage

- The expected answer is present first and candidate values are unique ignoring case.
- Expected rank, best-incorrect margin, and restricted-choice probability use mean token log probability.
- Summaries reject a single-choice set or a missing expected answer.
- Strict assertion scoring accepts direct and explicit answers but rejects an incidental food fact inside a multi-hop refusal.
- Diverse archive scoring accepts explicit historical-speaker, literary-speaker, and treaty-location assertions.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Phase 8 runner | Expected answer is included in extracted archive choices | Dropping the explicit expected answer seed |
| Report aggregation | Ranking is descending by mean token log probability | Switching silently to sequence sums |
