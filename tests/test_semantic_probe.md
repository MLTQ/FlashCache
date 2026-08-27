# test_semantic_probe.py

## Purpose

Verifies that the active semantic probe is answer-free, tokenizer-aware, and assigns the expected YES/NO score direction.

## Components

### `test_relevance_probe_preserves_blocks_without_leaking_answer`
- **Does**: Ensures the probe retains candidate evidence and query subject while excluding the ground-truth answer.

### `test_provenance_probe_leaks_neither_target_key_nor_answer`
- **Does**: Ensures direct page-key extraction sees neither the requested target key nor its answer.

### `test_normalized_key_matching_ignores_punctuation_but_not_missing_tokens`
- **Does**: Accepts quote/parenthesis formatting differences while rejecting incomplete keys that omit the year.

### `test_binary_probe_prefers_the_higher_probability_token_set`
- **Does**: Checks one-token surface filtering and positive log odds when YES variants receive more probability than NO variants.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `semantic_probe.py` | Query construction never includes the answer | Any answer leakage |
| Experiment ranking | Positive log odds favor YES/relevance | Token-set normalization or polarity changes |
