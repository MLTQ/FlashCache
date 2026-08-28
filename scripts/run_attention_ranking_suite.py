"""Evaluate one globally selected query-attention ranking rule on held-out tasks."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.attention_shortlist import ATTENTION_METRICS, rank_page_ids, scan_query_attention
from flash_cache.multi_hop_tasks import make_multi_hop_task
from flash_cache.probing import prepare_probe_caches, rollout, tokenize_task
from flash_cache.synthetic import contains_answer_text


def _positive_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument("--trial-count", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--hop-depths", type=_positive_csv, default=(1, 2, 3, 4))
    parser.add_argument("--variant-count", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--tail-layers", type=int, default=4)
    parser.add_argument("--tail-query-tokens", type=int, default=4)
    parser.add_argument("--control-horizon", type=int, default=32)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.trial_count < 4:
        parser.error("--trial-count must be at least four")
    if not 1 <= args.top_k <= args.blocks:
        parser.error("--top-k must be between one and --blocks")
    if any(depth > 4 for depth in args.hop_depths):
        parser.error("hop depths must be between one and four")
    if args.variant_count < 1:
        parser.error("--variant-count must be positive")
    if args.tail_layers < 1 or args.tail_query_tokens < 1:
        parser.error("attention tail widths must be positive")
    if args.control_horizon < 1:
        parser.error("--control-horizon must be positive")
    return args


def _timed_cuda(call: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def _render_full_prompt(tokenizer: Any, task: Any, prompt_format: str) -> str:
    page_text = "".join(task.blocks)
    if prompt_format == "chat":
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": task.system_message},
                {
                    "role": "user",
                    "content": f"Archived records:\n{page_text}\n{task.query_message}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"{task.pinned_text}{page_text}{task.recent_text}"


def _full_prefill_answer(
    model: Any,
    tokenizer: Any,
    task: Any,
    prompt_format: str,
    horizon: int,
    device: torch.device,
) -> str:
    prompt = _render_full_prompt(tokenizer, task, prompt_format)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    prefix_ids = prompt_ids[:, :-1]
    with torch.inference_mode():
        outputs = model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            return_dict=True,
        )
    answer = rollout(
        model,
        outputs.past_key_values,
        prompt_ids[:, -1:],
        int(prefix_ids.shape[-1]),
        horizon,
    )
    return tokenizer.decode(answer.tokens.tolist(), skip_special_tokens=True)


def _aggregate_metric(
    rows: Sequence[dict[str, Any]],
    metric: str,
    eligible_only: bool,
) -> dict[str, Any]:
    selected_rows = [row for row in rows if row["full_prefill_correct"]] if eligible_only else list(rows)
    if not selected_rows:
        return {
            "trial_count": 0,
            "all_relevant_top_k_rate": None,
            "mean_relevant_page_recall": None,
            "mean_reciprocal_worst_relevant_rank": None,
        }
    all_relevant = 0
    recall_sum = 0.0
    reciprocal_worst_rank_sum = 0.0
    for row in selected_rows:
        result = row["metrics"][metric]
        all_relevant += int(result["all_relevant_pages_selected"])
        recall_sum += result["relevant_pages_selected"] / row["hop_depth"]
        reciprocal_worst_rank_sum += 1.0 / max(result["relevant_page_ranks"].values())
    count = len(selected_rows)
    return {
        "trial_count": count,
        "all_relevant_top_k_rate": all_relevant / count,
        "mean_relevant_page_recall": recall_sum / count,
        "mean_reciprocal_worst_relevant_rank": reciprocal_worst_rank_sum / count,
    }


def _select_global_metric(development_rows: Sequence[dict[str, Any]]) -> str:
    """Choose one metric by development coverage with declared-order tie breaking."""
    best_metric = ATTENTION_METRICS[0]
    best_objective = (-1.0, -1.0, -1.0)
    for metric in ATTENTION_METRICS:
        aggregate = _aggregate_metric(development_rows, metric, eligible_only=False)
        objective = (
            float(aggregate["all_relevant_top_k_rate"]),
            float(aggregate["mean_relevant_page_recall"]),
            float(aggregate["mean_reciprocal_worst_relevant_rank"]),
        )
        if objective > best_objective:
            best_metric = metric
            best_objective = objective
    return best_metric


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

    rows: list[dict[str, Any]] = []
    development_count = args.trial_count // 2
    for trial_index in range(args.trial_count):
        seed = args.seed_base + trial_index
        split = "development" if trial_index < development_count else "holdout"
        schedule_index = (
            trial_index if split == "development" else trial_index - development_count
        )
        hop_depth = args.hop_depths[schedule_index % len(args.hop_depths)]
        variant = schedule_index % args.variant_count
        task = make_multi_hop_task(seed, args.blocks, hop_depth, variant)
        tokenized = tokenize_task(tokenizer, task, device, prompt_format=args.prompt_format)
        prepared, cold_prepare_latency_ms = _timed_cuda(
            lambda: prepare_probe_caches(model, tokenized, position_policy="original")
        )
        scan, scan_latency_ms = _timed_cuda(
            lambda: scan_query_attention(
                model,
                tokenized,
                prepared,
                tail_layer_count=args.tail_layers,
                tail_query_token_count=args.tail_query_tokens,
            )
        )
        full_text, full_latency_ms = _timed_cuda(
            lambda: _full_prefill_answer(
                model,
                tokenizer,
                task,
                args.prompt_format,
                args.control_horizon,
                device,
            )
        )
        relevant_ids = set(task.relevant_block_ids)
        metrics: dict[str, Any] = {}
        for metric in ATTENTION_METRICS:
            ranking = rank_page_ids(scan.scores, metric, len(scan.scores))
            selected = ranking[: args.top_k]
            metrics[metric] = {
                "selected_page_ids": list(selected),
                "relevant_pages_selected": len(relevant_ids.intersection(selected)),
                "all_relevant_pages_selected": relevant_ids.issubset(selected),
                "relevant_page_ranks": {
                    str(page_id): ranking.index(page_id) + 1 for page_id in sorted(relevant_ids)
                },
            }
        fixed_prefix = set(range(args.top_k))
        row = {
            "trial_index": trial_index,
            "split": split,
            "seed": seed,
            "variant": variant,
            "hop_depth": hop_depth,
            "question": task.query_message,
            "answer": task.answer,
            "relevant_page_ids": list(task.relevant_block_ids),
            "cold_page_token_count": sum(int(ids.shape[-1]) for ids in tokenized.block_ids),
            "query_token_count": scan.query_token_count,
            "cold_prepare_latency_ms": cold_prepare_latency_ms,
            "attention_scan_latency_ms": scan_latency_ms,
            "full_prefill_latency_ms": full_latency_ms,
            "full_prefill_generated_answer": full_text,
            "full_prefill_correct": contains_answer_text(full_text, task.answer_match),
            "fixed_prefix_relevant_pages_selected": len(relevant_ids.intersection(fixed_prefix)),
            "fixed_prefix_all_relevant_selected": relevant_ids.issubset(fixed_prefix),
            "page_scores": [score.to_dict() for score in scan.scores],
            "metrics": metrics,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "trial": trial_index,
                    "split": split,
                    "depth": hop_depth,
                    "full": row["full_prefill_correct"],
                    "scan_ms": scan_latency_ms,
                }
            ),
            flush=True,
        )

    development_rows = [row for row in rows if row["split"] == "development"]
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    selected_metric = _select_global_metric(development_rows)
    aggregates: dict[str, Any] = defaultdict(dict)
    for metric in ATTENTION_METRICS:
        for split_name, split_rows in (
            ("development", development_rows),
            ("holdout", holdout_rows),
            ("all", rows),
        ):
            aggregates[metric][split_name] = {
                "all_trials": _aggregate_metric(split_rows, metric, eligible_only=False),
                "full_prefill_correct_trials": _aggregate_metric(
                    split_rows, metric, eligible_only=True
                ),
            }

    fixed_prefix_all_relevant = sum(row["fixed_prefix_all_relevant_selected"] for row in rows)
    summary = {
        "model": args.model,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "seed_base": args.seed_base,
        "trial_count": args.trial_count,
        "development_trial_count": len(development_rows),
        "holdout_trial_count": len(holdout_rows),
        "blocks": args.blocks,
        "hop_depths": list(args.hop_depths),
        "variant_count": args.variant_count,
        "top_k": args.top_k,
        "prompt_format": args.prompt_format,
        "globally_selected_metric": selected_metric,
        "selection_rule": (
            "maximize development all-relevant top-K rate, then mean recall, then reciprocal "
            "worst-relevant rank; declared metric order breaks exact ties"
        ),
        "selected_metric_development": aggregates[selected_metric]["development"],
        "selected_metric_holdout": aggregates[selected_metric]["holdout"],
        "metric_aggregates": dict(aggregates),
        "full_prefill_correct_count": sum(row["full_prefill_correct"] for row in rows),
        "fixed_prefix_all_relevant_top_k_rate": fixed_prefix_all_relevant / len(rows),
        "attention_scan_latency_ms": {
            "mean": sum(row["attention_scan_latency_ms"] for row in rows) / len(rows),
            "min": min(row["attention_scan_latency_ms"] for row in rows),
            "max": max(row["attention_scan_latency_ms"] for row in rows),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trials.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
