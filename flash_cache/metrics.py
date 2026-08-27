"""Distribution and fixed-trajectory influence metrics for cache probes."""

from __future__ import annotations

from typing import Any

import torch


def trajectory_influence(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    baseline_tokens: torch.Tensor,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare distributions along a shared, baseline-greedy token trajectory."""
    if baseline_logits.shape != candidate_logits.shape:
        raise ValueError("Baseline and candidate logits must have identical shapes")
    if baseline_logits.ndim != 2:
        raise ValueError("Trajectory logits must have shape [horizon, vocabulary]")
    if baseline_tokens.shape != (baseline_logits.shape[0],):
        raise ValueError("Baseline token count must match the speculative horizon")

    baseline_log_probs = torch.log_softmax(baseline_logits.float(), dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
    baseline_probs = baseline_log_probs.exp()
    candidate_probs = candidate_log_probs.exp()

    kl_candidate_to_baseline = (candidate_probs * (candidate_log_probs - baseline_log_probs)).sum(dim=-1)
    mixture = 0.5 * (baseline_probs + candidate_probs)
    mixture_log_probs = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    js = 0.5 * (
        (baseline_probs * (baseline_log_probs - mixture_log_probs)).sum(dim=-1)
        + (candidate_probs * (candidate_log_probs - mixture_log_probs)).sum(dim=-1)
    )
    baseline_entropy = -(baseline_probs * baseline_log_probs).sum(dim=-1)
    candidate_entropy = -(candidate_probs * candidate_log_probs).sum(dim=-1)

    baseline_top = baseline_logits.topk(top_k, dim=-1).indices
    candidate_top = candidate_logits.topk(top_k, dim=-1).indices
    overlap = (baseline_top.unsqueeze(-1) == candidate_top.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1) / top_k
    top1_changed = baseline_logits.argmax(dim=-1) != candidate_logits.argmax(dim=-1)

    token_index = baseline_tokens.unsqueeze(-1)
    baseline_sequence_log_prob = baseline_log_probs.gather(-1, token_index).sum()
    candidate_sequence_log_prob = candidate_log_probs.gather(-1, token_index).sum()

    return {
        "kl_candidate_to_baseline_mean": float(kl_candidate_to_baseline.mean().item()),
        "kl_candidate_to_baseline_sum": float(kl_candidate_to_baseline.sum().item()),
        "js_mean": float(js.mean().item()),
        "js_sum": float(js.sum().item()),
        "baseline_entropy_mean": float(baseline_entropy.mean().item()),
        "candidate_entropy_mean": float(candidate_entropy.mean().item()),
        "entropy_delta_mean": float((candidate_entropy - baseline_entropy).mean().item()),
        "top_k_overlap_mean": float(overlap.mean().item()),
        "top1_changed_fraction": float(top1_changed.float().mean().item()),
        "baseline_sequence_log_prob": float(baseline_sequence_log_prob.item()),
        "candidate_sequence_log_prob": float(candidate_sequence_log_prob.item()),
        "sequence_log_prob_delta": float((candidate_sequence_log_prob - baseline_sequence_log_prob).item()),
    }

