# `test_semantic_carrier.py`

## Purpose

Checks the answer-free semantic-carrier prompt and the pure token-selection/transcript helpers used by both the poisoned and exact-replay paths.

## Components

### Prompt leakage test

Confirms the semantic task keeps every source page and the original question, contains generic keep-every-page instructions, and does not include the known answer.

### Greedy token tests

Confirm model control tokens are excluded without imposing a textual output format, and invalid or fully masked logits fail loudly.

### Transcript test

Confirms clean replay preserves exact token order, multiplicity, and page-to-page concatenation.

### Selection-mode validation

Confirms unsupported selection policies fail before any model or cache access; in particular, no oracle mode can silently enter the experiment.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `semantic_carrier.py` | Greedy selection masks only explicit control IDs | Filtering tokens by desired note format or target content |
| Exact-replay control | Flattening makes no semantic or textual transformation | Decode/re-tokenize replay |
