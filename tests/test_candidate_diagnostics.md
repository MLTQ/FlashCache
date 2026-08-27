# test_candidate_diagnostics.py

## Purpose

Checks behavioral telemetry invariants on small deterministic logits before GPU experiments rely on the recorded fields.

## Components

### `test_behavior_metrics_capture_gain_confidence_and_prefix`
- **Does**: Verifies positive self-proposal gain, positive confidence margin, token agreement, and common-prefix length.

### `test_behavior_metrics_reject_mismatched_horizons`
- **Does**: Verifies inconsistent rollout shapes fail explicitly rather than producing misaligned metrics.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `candidate_diagnostics.py` | Shared-path gain and exact token-agreement values are deterministic | Metric formula or shape-validation changes |
