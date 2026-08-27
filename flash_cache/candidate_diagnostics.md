# candidate_diagnostics.py

## Purpose

Records answer-free behavioral measurements for every flashed-cache candidate. These are exploratory winner/loser telemetry, not declared retrieval rules.

## Components

### `greedy_behavior_metrics`
- **Does**: Measures candidate confidence, entropy, top-token margin, agreement with the no-block generation, and the likelihood gain for the candidate's own proposed continuation.
- **Interacts with**: Candidate greedy rollouts and baseline forced rollouts from `probing.py`.
- **Rationale**: Scoring a candidate's own tokens under both branches compares the models along one shared path without reading the answer key.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment runner | All inputs use one shared horizon and return JSON-serializable scalars | Shape changes or tensor-valued outputs |
| Offline diagnostics | Field names remain stable across winner and loser rows | Renaming metrics without migration |

## Notes

- Self-proposed gain can reward confidently wrong continuations; it is telemetry until held-out evidence says otherwise.
- Common-prefix measurements are exact token comparisons and therefore tokenizer-dependent.
