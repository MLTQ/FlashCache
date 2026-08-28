# `answer_scoring.py`

## Purpose

Measures whether a cache condition raises the model's likelihood of the correct answer phrase without requiring a particular generated format. This complements free-form substring matching, which can report a false success when the model merely mentions the answer while denying that it follows from the question.

## Components

### `extract_archive_answer_choices`

- **Does**: Extracts the expected food answer and every same-domain favorite-food distractor present in the synthetic archive.
- **Rationale**: A closed, archive-derived choice set tests semantic preference among plausible values rather than arbitrary vocabulary tokens.

### `score_answer_choices`

- **Does**: Teacher-forces one shared `Final answer:` cue followed by each canonical answer phrase from the same immutable cache state, then scores only the answer tokens.
- **Rationale**: The shared cue removes the large penalty caused by a model preferring an explanatory opening such as “Based on...” while leaving the answer comparison identical across conditions.

### `contains_asserted_answer`

- **Does**: Accepts direct answers, explicit answer markers, conclusions, query-subject favorite-food assertions, and explicit history, quotation, treaty, or pressure facts while rejecting incidental mentions of the expected value.
- **Rationale**: A response that says the relationship cannot be determined and merely repeats one page's food fact has not answered a multi-hop question.

### `summarize_answer_choice_scores`

- **Does**: Reports expected-answer rank, mean-log-probability margin over the best incorrect value, and a softmax probability restricted to the declared choices.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Synthetic preference runner | Page facts use the phrase `favorite food is ...` | Changing fact templates without updating extraction |
| Likelihood comparison | Every condition uses the identical answer cue, leading-space answer tokenization, and choice set | Condition-specific cues, formatting, or choices |
| Probability interpretation | Restricted probability is a normalized diagnostic over archive-derived choices | Describing it as unconstrained next-text probability |
| Answer-free controller | These metrics are evaluation-only | Using the expected answer score to select pages online |
| Strict generation aggregate | Assertion scoring remains separate from permissive phrase-presence telemetry | Silently replacing legacy results or treating every mention as an answer |

## Notes

- Sequence log probability is the true likelihood of the canonical token sequence but favors shorter strings when comparing different answers.
- Mean-token log probability reduces length bias and therefore drives rank, margin, and restricted-choice probability.
- The cue is evaluation scaffolding, not a retrieval instruction and not part of any answer score.
- The leading space matches an ordinary answer continuation after the cue's colon.
