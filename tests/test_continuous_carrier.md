# `test_continuous_carrier.py`

## Purpose

Checks the pure contracts that distinguish uninterrupted rotating decode from the earlier summarizer: prompt construction, deterministic page scheduling, and exact replay token identity.

## Components

### Prompt contract test

- **Does**: Confirms the original question and pages are preserved without answer leakage, answer emergence is permitted immediately, and separate page notes or visible boundaries are forbidden.

### Rotation tests

- **Does**: Confirms page IDs rotate every token or in fixed short windows and invalid empty dimensions fail loudly.

### Replay tests

- **Does**: Confirms exact replay processes the original probe followed by unchanged visible tokens and rejects a malformed probe.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `continuous_carrier.py` | Schedule is deterministic and relevance-blind | Random or metadata-dependent page order |
| Exact replay | Original probe is processed before every visible generated token | Omitting, duplicating, or re-tokenizing the transcript |
