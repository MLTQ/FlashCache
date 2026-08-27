"""Unit tests for answer-free candidate behavior diagnostics."""

import torch

from flash_cache.candidate_diagnostics import greedy_behavior_metrics


def test_behavior_metrics_capture_gain_confidence_and_prefix() -> None:
    candidate_logits = torch.tensor([[4.0, 0.0, -1.0], [0.0, 5.0, -1.0]])
    baseline_logits = torch.tensor([[0.0, 4.0, -1.0], [4.0, 0.0, -1.0]])
    candidate_tokens = torch.tensor([0, 1])
    baseline_tokens = torch.tensor([0, 2])

    metrics = greedy_behavior_metrics(
        candidate_logits,
        candidate_tokens,
        baseline_logits,
        baseline_tokens,
    )

    assert metrics["self_proposed_log_prob_gain_sum"] > 0
    assert metrics["candidate_greedy_top1_margin_mean"] > 0
    assert metrics["baseline_greedy_token_match_fraction"] == 0.5
    assert metrics["baseline_greedy_common_prefix_tokens"] == 1


def test_behavior_metrics_reject_mismatched_horizons() -> None:
    logits = torch.tensor([[1.0, 0.0]])

    try:
        greedy_behavior_metrics(logits, torch.tensor([0, 1]), logits, torch.tensor([0]))
    except ValueError as error:
        assert "Candidate token count" in str(error)
    else:
        raise AssertionError("Expected a shape validation error")
