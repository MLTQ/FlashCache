"""Select one cached-value page similarity metric and evaluate it on holdout tasks."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.attention_shortlist import rank_page_ids, scan_query_attention
from flash_cache.kv_similarity import (
    KV_SIMILARITY_METRICS,
    build_packed_cold_value_index,
    rank_kv_similarity_page_ids,
    rank_packed_value_page_ids,
    scan_kv_value_similarity,
    scan_packed_value_max_similarity,
)
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
    parser.add_argument("--seed-base", type=int, default=300)
    parser.add_argument("--trial-count", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--hop-depths", type=_positive_csv, default=(1, 2, 3, 4))
    parser.add_argument("--variant-count", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--tail-layers", type=int, default=4)
    parser.add_argument("--top-pairs", type=int, default=4)
    parser.add_argument("--attention-reference-metric", default="all_query_mass")
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
    if min(args.variant_count, args.tail_layers, args.top_pairs, args.control_horizon) < 1:
        parser.error("variant, tail, top-pair, and horizon values must be positive")
    return args


def _timed_cuda(call: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def _full_prefill_answer(
    model: Any,
    tokenizer: Any,
    task: Any,
    prompt_format: str,
    horizon: int,
    device: torch.device,
) -> str:
    page_text = "".join(task.blocks)
    if prompt_format == "chat":
        text = tokenizer.apply_chat_template(
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
    else:
        text = f"{task.pinned_text}{page_text}{task.recent_text}"
    prompt_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
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


def _aggregate(rows: Sequence[dict[str, Any]], metric: str, eligible_only: bool) -> dict[str, Any]:
    chosen = [row for row in rows if row["full_prefill_correct"]] if eligible_only else list(rows)
    if not chosen:
        return {"trial_count": 0, "all_relevant_top_k_rate": None, "mean_recall": None}
    all_relevant = sum(row["kv_metrics"][metric]["all_relevant_pages_selected"] for row in chosen)
    recall = sum(
        row["kv_metrics"][metric]["relevant_pages_selected"] / row["hop_depth"]
        for row in chosen
    )
    worst_reciprocal = sum(
        1.0 / max(row["kv_metrics"][metric]["relevant_page_ranks"].values())
        for row in chosen
    )
    return {
        "trial_count": len(chosen),
        "all_relevant_top_k_rate": all_relevant / len(chosen),
        "mean_recall": recall / len(chosen),
        "mean_reciprocal_worst_relevant_rank": worst_reciprocal / len(chosen),
    }


def _select_metric(development_rows: Sequence[dict[str, Any]]) -> str:
    best = KV_SIMILARITY_METRICS[0]
    best_objective = (-1.0, -1.0, -1.0)
    for metric in KV_SIMILARITY_METRICS:
        aggregate = _aggregate(development_rows, metric, eligible_only=False)
        objective = (
            float(aggregate["all_relevant_top_k_rate"]),
            float(aggregate["mean_recall"]),
            float(aggregate["mean_reciprocal_worst_relevant_rank"]),
        )
        if objective > best_objective:
            best = metric
            best_objective = objective
    return best


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
        split = "development" if trial_index < development_count else "holdout"
        schedule_index = trial_index if split == "development" else trial_index - development_count
        hop_depth = args.hop_depths[schedule_index % len(args.hop_depths)]
        variant = schedule_index % args.variant_count
        seed = args.seed_base + trial_index
        task = make_multi_hop_task(seed, args.blocks, hop_depth, variant)
        tokenized = tokenize_task(tokenizer, task, device, prompt_format=args.prompt_format)
        prepared, prepare_latency_ms = _timed_cuda(
            lambda: prepare_probe_caches(model, tokenized, position_policy="original")
        )
        packed_index, packed_build_latency_ms = _timed_cuda(
            lambda: build_packed_cold_value_index(prepared)
        )
        packed_scan, packed_scan_latency_ms = _timed_cuda(
            lambda: scan_packed_value_max_similarity(tokenized, prepared, packed_index)
        )
        kv_scan, kv_scan_latency_ms = _timed_cuda(
            lambda: scan_kv_value_similarity(
                tokenized,
                prepared,
                tail_layer_count=args.tail_layers,
                top_pair_count=args.top_pairs,
            )
        )
        attention_scan, attention_scan_latency_ms = _timed_cuda(
            lambda: scan_query_attention(
                model,
                tokenized,
                prepared,
                tail_layer_count=args.tail_layers,
                tail_query_token_count=4,
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
        relevant = set(task.relevant_block_ids)
        kv_metrics: dict[str, Any] = {}
        for metric in KV_SIMILARITY_METRICS:
            ranking = rank_kv_similarity_page_ids(kv_scan.scores, metric, len(kv_scan.scores))
            selected = ranking[: args.top_k]
            kv_metrics[metric] = {
                "selected_page_ids": list(selected),
                "relevant_pages_selected": len(relevant.intersection(selected)),
                "all_relevant_pages_selected": relevant.issubset(selected),
                "relevant_page_ranks": {
                    str(page_id): ranking.index(page_id) + 1 for page_id in sorted(relevant)
                },
            }
        attention_ranking = rank_page_ids(
            attention_scan.scores,
            args.attention_reference_metric,
            len(attention_scan.scores),
        )
        attention_selected = attention_ranking[: args.top_k]
        packed_selected = rank_packed_value_page_ids(packed_scan, args.top_k)
        reference_max_selected = rank_kv_similarity_page_ids(
            kv_scan.scores,
            "all_max_cosine",
            args.top_k,
        )
        row = {
            "trial_index": trial_index,
            "split": split,
            "seed": seed,
            "variant": variant,
            "hop_depth": hop_depth,
            "question": task.query_message,
            "answer": task.answer,
            "relevant_page_ids": list(task.relevant_block_ids),
            "prepare_latency_ms": prepare_latency_ms,
            "packed_index_build_latency_ms": packed_build_latency_ms,
            "packed_scan_latency_ms": packed_scan_latency_ms,
            "kv_scan_latency_ms": kv_scan_latency_ms,
            "attention_scan_latency_ms": attention_scan_latency_ms,
            "full_prefill_latency_ms": full_latency_ms,
            "full_prefill_generated_answer": full_text,
            "full_prefill_correct": contains_answer_text(full_text, task.answer_match),
            "kv_scores": [score.to_dict() for score in kv_scan.scores],
            "kv_metrics": kv_metrics,
            "packed_all_max": {
                "scores": list(packed_scan.scores),
                "selected_page_ids": list(packed_selected),
                "matches_reference_top_k_set": set(packed_selected)
                == set(reference_max_selected),
                "relevant_pages_selected": len(relevant.intersection(packed_selected)),
                "all_relevant_pages_selected": relevant.issubset(packed_selected),
            },
            "attention_reference": {
                "metric": args.attention_reference_metric,
                "selected_page_ids": list(attention_selected),
                "relevant_pages_selected": len(relevant.intersection(attention_selected)),
                "all_relevant_pages_selected": relevant.issubset(attention_selected),
            },
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "trial": trial_index,
                    "split": split,
                    "depth": hop_depth,
                    "full": row["full_prefill_correct"],
                    "packed_ms": packed_scan_latency_ms,
                    "kv_ms": kv_scan_latency_ms,
                    "attention_ms": attention_scan_latency_ms,
                }
            ),
            flush=True,
        )

    development_rows = [row for row in rows if row["split"] == "development"]
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    selected_metric = _select_metric(development_rows)
    aggregates: dict[str, Any] = defaultdict(dict)
    for metric in KV_SIMILARITY_METRICS:
        for name, split_rows in (
            ("development", development_rows),
            ("holdout", holdout_rows),
            ("all", rows),
        ):
            aggregates[metric][name] = {
                "all_trials": _aggregate(split_rows, metric, eligible_only=False),
                "full_prefill_correct_trials": _aggregate(split_rows, metric, eligible_only=True),
            }

    attention_all = sum(
        row["attention_reference"]["all_relevant_pages_selected"] for row in rows
    ) / len(rows)
    summary = {
        "model": args.model,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "seed_base": args.seed_base,
        "trial_count": args.trial_count,
        "blocks": args.blocks,
        "hop_depths": list(args.hop_depths),
        "top_k": args.top_k,
        "globally_selected_kv_metric": selected_metric,
        "selected_metric_development": aggregates[selected_metric]["development"],
        "selected_metric_holdout": aggregates[selected_metric]["holdout"],
        "kv_metric_aggregates": dict(aggregates),
        "attention_reference_metric": args.attention_reference_metric,
        "attention_reference_all_relevant_top_k_rate": attention_all,
        "packed_all_max_top_k_set_match_rate": sum(
            row["packed_all_max"]["matches_reference_top_k_set"] for row in rows
        )
        / len(rows),
        "packed_all_max_all_relevant_top_k_rate": sum(
            row["packed_all_max"]["all_relevant_pages_selected"] for row in rows
        )
        / len(rows),
        "full_prefill_correct_count": sum(row["full_prefill_correct"] for row in rows),
        "packed_index_build_latency_ms": {
            "mean": sum(row["packed_index_build_latency_ms"] for row in rows) / len(rows),
            "min": min(row["packed_index_build_latency_ms"] for row in rows),
            "max": max(row["packed_index_build_latency_ms"] for row in rows),
        },
        "packed_scan_latency_ms": {
            "mean": sum(row["packed_scan_latency_ms"] for row in rows) / len(rows),
            "min": min(row["packed_scan_latency_ms"] for row in rows),
            "max": max(row["packed_scan_latency_ms"] for row in rows),
        },
        "kv_scan_latency_ms": {
            "mean": sum(row["kv_scan_latency_ms"] for row in rows) / len(rows),
            "min": min(row["kv_scan_latency_ms"] for row in rows),
            "max": max(row["kv_scan_latency_ms"] for row in rows),
        },
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
