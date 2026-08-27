# metrics.py

## Purpose

Computes influence signals between baseline and candidate distributions along the same speculative token history. Keeping the trajectory fixed avoids confounding distribution change with different generated prefixes.

## Components

### `trajectory_influence`
- **Does**: Reports candidate-to-baseline KL, Jensen-Shannon divergence, entropy, top-k overlap, top-1 changes, and fixed-sequence log-probability delta.
- **Interacts with**: Baseline and candidate rollouts from `probing.py`.
- **Rationale**: The baseline-greedy token path gives multi-token measurements a shared conditioning sequence.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment ranking | `js_mean` is a nonnegative scalar primary score | Renaming or redefining the field |
| JSONL logging | Every return value is JSON-serializable | Returning tensors |

