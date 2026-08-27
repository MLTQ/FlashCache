"""Unit tests for speculative influence metrics."""

from __future__ import annotations

import torch

from flash_cache.metrics import trajectory_influence


def test_identical_trajectories_have_zero_divergence() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]])
    tokens = torch.tensor([0, 1])

    metrics = trajectory_influence(logits, logits.clone(), tokens, top_k=2)

    assert abs(metrics["kl_candidate_to_baseline_mean"]) < 1e-7
    assert abs(metrics["js_mean"]) < 1e-7
    assert metrics["sequence_log_prob_delta"] == 0.0
    assert metrics["top_k_overlap_mean"] == 1.0


def test_shifted_candidate_has_positive_divergence() -> None:
    baseline = torch.tensor([[4.0, 0.0, -1.0]])
    candidate = torch.tensor([[0.0, 4.0, -1.0]])
    metrics = trajectory_influence(baseline, candidate, torch.tensor([0]), top_k=2)

    assert metrics["kl_candidate_to_baseline_mean"] > 0
    assert metrics["js_mean"] > 0
    assert metrics["top1_changed_fraction"] == 1.0

