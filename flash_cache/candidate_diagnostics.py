"""Answer-free behavioral diagnostics for candidate-cache rollouts."""

from __future__ import annotations

from typing import Any

import torch


def greedy_behavior_metrics(
    candidate_logits: torch.Tensor,
    candidate_tokens: torch.Tensor,
    baseline_on_candidate_logits: torch.Tensor,
    baseline_greedy_tokens: torch.Tensor,
) -> dict[str, Any]:
    """Describe confidence, self-proposal gain, and agreement without an answer key."""
    if candidate_logits.shape != baseline_on_candidate_logits.shape:
        raise ValueError("Candidate and baseline-on-candidate logits must have identical shapes")
    if candidate_logits.ndim != 2:
        raise ValueError("Behavior logits must have shape [horizon, vocabulary]")
    horizon = candidate_logits.shape[0]
    expected_token_shape = (horizon,)
    if candidate_tokens.shape != expected_token_shape:
        raise ValueError("Candidate token count must match the behavior horizon")
    if baseline_greedy_tokens.shape != expected_token_shape:
        raise ValueError("Baseline token count must match the behavior horizon")

    candidate_log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
    baseline_log_probs = torch.log_softmax(baseline_on_candidate_logits.float(), dim=-1)
    candidate_probs = candidate_log_probs.exp()
    token_index = candidate_tokens.unsqueeze(-1)
    candidate_selected = candidate_log_probs.gather(-1, token_index).squeeze(-1)
    baseline_selected = baseline_log_probs.gather(-1, token_index).squeeze(-1)
    candidate_entropy = -(candidate_probs * candidate_log_probs).sum(dim=-1)
    top_two = candidate_log_probs.topk(2, dim=-1).values
    top1_margin = top_two[:, 0] - top_two[:, 1]

    matches = candidate_tokens == baseline_greedy_tokens
    mismatch_positions = (~matches).nonzero(as_tuple=False)
    common_prefix_tokens = horizon if mismatch_positions.numel() == 0 else int(mismatch_positions[0, 0])
    gain = candidate_selected - baseline_selected

    return {
        "behavior_horizon": int(horizon),
        "candidate_greedy_log_prob_mean": float(candidate_selected.mean().item()),
        "candidate_greedy_log_prob_sum": float(candidate_selected.sum().item()),
        "baseline_on_candidate_log_prob_mean": float(baseline_selected.mean().item()),
        "baseline_on_candidate_log_prob_sum": float(baseline_selected.sum().item()),
        "self_proposed_log_prob_gain_mean": float(gain.mean().item()),
        "self_proposed_log_prob_gain_sum": float(gain.sum().item()),
        "candidate_greedy_entropy_mean": float(candidate_entropy.mean().item()),
        "candidate_greedy_top1_margin_mean": float(top1_margin.mean().item()),
        "baseline_greedy_token_match_fraction": float(matches.float().mean().item()),
        "baseline_greedy_common_prefix_tokens": common_prefix_tokens,
        "baseline_greedy_common_prefix_fraction": float(common_prefix_tokens / horizon),
    }
