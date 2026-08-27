# test_metrics.py

## Purpose

Checks the numerical invariants of trajectory influence metrics using tiny deterministic logits.

## Components

### `test_identical_trajectories_have_zero_divergence`
- **Does**: Proves identical distributions yield zero KL, JS, and sequence-log-probability delta.

### `test_shifted_candidate_has_positive_divergence`
- **Does**: Proves a displaced top token yields positive divergence and a top-1 change.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `metrics.py` | Divergences obey identity and positivity invariants | Metric formulation changes |

