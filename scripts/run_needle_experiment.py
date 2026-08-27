"""Run exhaustive single-block influence ranking for one synthetic needle task."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.metrics import trajectory_influence
from flash_cache.probing import flash_candidate, prepare_probe_caches, rollout, tokenize_task
from flash_cache.synthetic import contains_answer_text
from flash_cache.task_families import TASK_FAMILIES, make_experiment_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--task-family", choices=TASK_FAMILIES, default="valve_pressure")
    parser.add_argument("--task-variant", type=int, default=0)
    parser.add_argument("--target-identifier", default="X-17")
    parser.add_argument("--target-pressure", type=int, default=413)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--generation-horizon", type=int, default=20)
    parser.add_argument("--position-policy", choices=("original", "hot_slot"), default="original")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1/seed_7"))
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def decoded_top_tokens(tokenizer, logits: torch.Tensor, count: int = 5) -> list[dict[str, object]]:
    """Render a small inspectable top-token list for JSON output."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    values, indices = log_probs.topk(count)
    return [
        {"token_id": int(token_id), "text": tokenizer.decode([int(token_id)]), "log_prob": float(value)}
        for token_id, value in zip(indices.tolist(), values.tolist(), strict=True)
    ]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    gpu_name = torch.cuda.get_device_name(0)
    if args.expected_gpu.lower() not in gpu_name.lower():
        raise RuntimeError(f"Refusing to run on {gpu_name!r}; expected {args.expected_gpu!r}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).to("cuda:0")
    model.eval()

    task = make_experiment_task(
        args.seed,
        args.blocks,
        task_family=args.task_family,
        variant=args.task_variant,
        target_identifier=args.target_identifier,
        target_pressure=args.target_pressure,
    )
    tokenized = tokenize_task(tokenizer, task, torch.device("cuda:0"), prompt_format=args.prompt_format)
    prepared = prepare_probe_caches(model, tokenized, position_policy=args.position_policy)
    baseline = rollout(
        model,
        prepared.baseline_cache,
        tokenized.probe_token,
        tokenized.probe_position,
        args.horizon,
    )
    pinned_length = int(tokenized.pinned_ids.shape[-1])
    answer_tokens = tokenizer(
        " " + task.answer, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0].to("cuda:0")
    baseline_answer = rollout(
        model,
        prepared.baseline_cache,
        tokenized.probe_token,
        tokenized.probe_position,
        int(answer_tokens.shape[0]),
        forced_tokens=answer_tokens,
    )
    baseline_generation = rollout(
        model,
        prepared.baseline_cache,
        tokenized.probe_token,
        tokenized.probe_position,
        args.generation_horizon,
    )
    baseline_generated_text = tokenizer.decode(baseline_generation.tokens.tolist())

    rows: list[dict[str, object]] = []
    for block_id, candidate_cache in enumerate(prepared.cold_blocks):
        active_cache = flash_candidate(prepared.baseline_cache, candidate_cache, pinned_length)
        torch.cuda.synchronize()
        started = time.perf_counter()
        candidate = rollout(
            model,
            active_cache,
            tokenized.probe_token,
            tokenized.probe_position,
            args.horizon,
            forced_tokens=baseline.tokens,
        )
        candidate_answer = rollout(
            model,
            active_cache,
            tokenized.probe_token,
            tokenized.probe_position,
            int(answer_tokens.shape[0]),
            forced_tokens=answer_tokens,
        )
        candidate_generation = rollout(
            model,
            active_cache,
            tokenized.probe_token,
            tokenized.probe_position,
            args.generation_horizon,
        )
        generated_text = tokenizer.decode(candidate_generation.tokens.tolist())
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        metrics = trajectory_influence(baseline.logits, candidate.logits, baseline.tokens)
        answer_metrics = trajectory_influence(baseline_answer.logits, candidate_answer.logits, answer_tokens)
        one_minus_js_mean = 1.0 - float(metrics["js_mean"])
        rows.append(
            {
                "seed": task.seed,
                "model": args.model,
                "task_family": task.task_family,
                "task_variant": args.task_variant,
                "target_key": task.target_key,
                "target_identifier": task.target_identifier,
                "target_pressure": task.target_pressure,
                "candidate_block_id": block_id,
                "source_text": task.blocks[block_id],
                "token_count": int(tokenized.block_ids[block_id].shape[-1]),
                "logical_position_start": int(tokenized.block_positions[block_id][0, 0].item()),
                "effective_position_start": int(prepared.effective_block_positions[block_id][0, 0].item()),
                "ground_truth_relevant": block_id == task.relevant_block_id,
                "latency_ms": latency_ms,
                "candidate_first_top_tokens": decoded_top_tokens(tokenizer, candidate.logits[0]),
                "generated_continuation": generated_text,
                "answer_correct": contains_answer_text(generated_text, task.answer_match),
                "answer_sequence_log_prob_baseline": answer_metrics["baseline_sequence_log_prob"],
                "answer_sequence_log_prob_candidate": answer_metrics["candidate_sequence_log_prob"],
                "answer_sequence_log_prob_delta": answer_metrics["sequence_log_prob_delta"],
                "one_minus_js_mean": one_minus_js_mean,
                **metrics,
            }
        )

    ranked = sorted(rows, key=lambda row: float(row["js_mean"]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank_by_js_mean"] = rank
    ranked_by_one_minus_js = sorted(
        rows, key=lambda row: float(row["one_minus_js_mean"]), reverse=True
    )
    for rank, row in enumerate(ranked_by_one_minus_js, start=1):
        row["rank_by_one_minus_js_mean"] = rank
    ranked_by_answer = sorted(rows, key=lambda row: float(row["answer_sequence_log_prob_delta"]), reverse=True)
    for rank, row in enumerate(ranked_by_answer, start=1):
        row["rank_by_answer_log_prob_delta"] = rank
    relevant_rank = next(int(row["rank_by_js_mean"]) for row in ranked if row["ground_truth_relevant"])
    relevant_inverse_js_rank = next(
        int(row["rank_by_one_minus_js_mean"])
        for row in ranked_by_one_minus_js
        if row["ground_truth_relevant"]
    )
    relevant_answer_rank = next(
        int(row["rank_by_answer_log_prob_delta"]) for row in ranked_by_answer if row["ground_truth_relevant"]
    )
    relevant_row = next(row for row in rows if row["ground_truth_relevant"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidates.jsonl").open("w") as output:
        for row in sorted(rows, key=lambda item: int(item["candidate_block_id"])):
            output.write(json.dumps(row) + "\n")

    summary = {
        "seed": task.seed,
        "model": args.model,
        "task_family": task.task_family,
        "task_variant": args.task_variant,
        "target_key": task.target_key,
        "target_identifier": task.target_identifier,
        "target_pressure": task.target_pressure,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "block_count": args.blocks,
        "speculative_horizon": args.horizon,
        "generation_horizon": args.generation_horizon,
        "position_policy": args.position_policy,
        "prompt_format": args.prompt_format,
        "relevant_block_id": task.relevant_block_id,
        "relevant_rank_by_js_mean": relevant_rank,
        "relevant_rank_by_one_minus_js_mean": relevant_inverse_js_rank,
        "relevant_rank_by_answer_log_prob_delta": relevant_answer_rank,
        "relevant_answer_log_prob_delta": relevant_row["answer_sequence_log_prob_delta"],
        "reciprocal_rank": 1.0 / relevant_rank,
        "one_minus_js_reciprocal_rank": 1.0 / relevant_inverse_js_rank,
        "answer": task.answer,
        "baseline_continuation": tokenizer.decode(baseline.tokens.tolist()),
        "baseline_generated_continuation": baseline_generated_text,
        "baseline_answer_correct": contains_answer_text(
            baseline_generated_text, task.answer_match
        ),
        "baseline_first_top_tokens": decoded_top_tokens(tokenizer, baseline.logits[0]),
        "relevant_generated_continuation": relevant_row["generated_continuation"],
        "relevant_answer_correct": relevant_row["answer_correct"],
        "correct_candidate_ids": [
            int(row["candidate_block_id"]) for row in rows if row["answer_correct"]
        ],
        "ranking": [int(row["candidate_block_id"]) for row in ranked],
        "one_minus_js_ranking": [
            int(row["candidate_block_id"]) for row in ranked_by_one_minus_js
        ],
        "answer_delta_ranking": [int(row["candidate_block_id"]) for row in ranked_by_answer],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    ordered_rows = sorted(rows, key=lambda row: int(row["candidate_block_id"]))
    colors = ["#d62728" if row["ground_truth_relevant"] else "#4c78a8" for row in ordered_rows]
    plt.figure(figsize=(9, 4.5))
    plt.bar([int(row["candidate_block_id"]) for row in ordered_rows], [float(row["js_mean"]) for row in ordered_rows], color=colors)
    plt.xlabel("Candidate block ID")
    plt.ylabel("Mean Jensen-Shannon divergence")
    plt.title(
        f"Qwen3-1.7B influence (seed {args.seed}, k={args.horizon}, "
        f"{args.position_policy}, {args.prompt_format})"
    )
    plt.tight_layout()
    plt.savefig(args.output_dir / "influence.png", dpi=160)
    plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
