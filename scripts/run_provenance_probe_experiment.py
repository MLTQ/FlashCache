"""Select a flashed page by extracting and matching its own provenance key."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.probing import flash_candidate, prepare_probe_caches, rollout, tokenize_task
from flash_cache.semantic_probe import contains_normalized_key, make_provenance_probe_task
from flash_cache.synthetic import contains_answer_text
from flash_cache.task_families import TASK_FAMILIES, make_experiment_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--task-family", choices=TASK_FAMILIES, default="history_person")
    parser.add_argument("--task-variant", type=int, default=3)
    parser.add_argument("--target-identifier", default="X-17")
    parser.add_argument("--target-pressure", type=int, default=413)
    parser.add_argument("--provenance-horizon", type=int, default=32)
    parser.add_argument("--answer-horizon", type=int, default=40)
    parser.add_argument("--position-policy", choices=("original", "hot_slot"), default="hot_slot")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def rollout_confidence_metrics(logits: torch.Tensor, tokens: torch.Tensor) -> dict[str, float]:
    """Summarize confidence of one greedy provenance-key generation."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    selected = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    top_two = log_probs.topk(2, dim=-1).values
    return {
        "probe_greedy_log_prob_mean": float(selected.mean().item()),
        "probe_entropy_mean": float(entropy.mean().item()),
        "probe_top1_margin_mean": float((top_two[:, 0] - top_two[:, 1]).mean().item()),
    }


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
    device = torch.device("cuda:0")

    task = make_experiment_task(
        args.seed,
        args.blocks,
        task_family=args.task_family,
        variant=args.task_variant,
        target_identifier=args.target_identifier,
        target_pressure=args.target_pressure,
    )
    answer_tokenized = tokenize_task(tokenizer, task, device, prompt_format=args.prompt_format)
    answer_prepared = prepare_probe_caches(
        model,
        answer_tokenized,
        position_policy=args.position_policy,
    )
    provenance_task = make_provenance_probe_task(task)
    provenance_tokenized = tokenize_task(
        tokenizer,
        provenance_task,
        device,
        prompt_format=args.prompt_format,
    )
    provenance_prepared = prepare_probe_caches(
        model,
        provenance_tokenized,
        position_policy=args.position_policy,
    )

    baseline_answer = rollout(
        model,
        answer_prepared.baseline_cache,
        answer_tokenized.probe_token,
        answer_tokenized.probe_position,
        args.answer_horizon,
    )
    baseline_answer_text = tokenizer.decode(baseline_answer.tokens.tolist())

    rows: list[dict[str, object]] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for block_id, candidate_cache in enumerate(provenance_prepared.cold_blocks):
        active_cache = flash_candidate(
            provenance_prepared.baseline_cache,
            candidate_cache,
            int(provenance_tokenized.pinned_ids.shape[-1]),
        )
        probe = rollout(
            model,
            active_cache,
            provenance_tokenized.probe_token,
            provenance_tokenized.probe_position,
            args.provenance_horizon,
        )
        generated_key_text = tokenizer.decode(probe.tokens.tolist())
        rows.append(
            {
                "candidate_block_id": block_id,
                "source_text": task.blocks[block_id],
                "ground_truth_relevant": block_id == task.relevant_block_id,
                "generated_provenance": generated_key_text,
                "target_key_match": contains_normalized_key(generated_key_text, task.target_key),
                **rollout_confidence_metrics(probe.logits, probe.tokens),
            }
        )
    torch.cuda.synchronize()
    probe_latency_ms = (time.perf_counter() - started) * 1000.0

    matching_rows = [row for row in rows if row["target_key_match"]]
    selected_row = matching_rows[0] if matching_rows else None
    selected_answer_text = None
    selected_answer_correct = False
    if selected_row is not None:
        selected_id = int(selected_row["candidate_block_id"])
        selected_cache = flash_candidate(
            answer_prepared.baseline_cache,
            answer_prepared.cold_blocks[selected_id],
            int(answer_tokenized.pinned_ids.shape[-1]),
        )
        selected_answer = rollout(
            model,
            selected_cache,
            answer_tokenized.probe_token,
            answer_tokenized.probe_position,
            args.answer_horizon,
        )
        selected_answer_text = tokenizer.decode(selected_answer.tokens.tolist())
        selected_answer_correct = contains_answer_text(selected_answer_text, task.answer_match)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidates.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")

    summary = {
        "seed": task.seed,
        "model": args.model,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "task_family": task.task_family,
        "task_variant": args.task_variant,
        "target_key": task.target_key,
        "answer": task.answer,
        "block_count": args.blocks,
        "relevant_block_id": task.relevant_block_id,
        "position_policy": args.position_policy,
        "prompt_format": args.prompt_format,
        "provenance_horizon": args.provenance_horizon,
        "matching_candidate_ids": [int(row["candidate_block_id"]) for row in matching_rows],
        "selected_candidate_id": (
            int(selected_row["candidate_block_id"]) if selected_row is not None else None
        ),
        "selected_ground_truth_relevant": (
            bool(selected_row["ground_truth_relevant"]) if selected_row is not None else None
        ),
        "baseline_answer": baseline_answer_text,
        "baseline_answer_correct": contains_answer_text(baseline_answer_text, task.answer_match),
        "selected_answer": selected_answer_text,
        "selected_answer_correct": selected_answer_correct,
        "probe_latency_ms": probe_latency_ms,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
