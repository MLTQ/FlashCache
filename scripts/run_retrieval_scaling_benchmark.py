"""Benchmark packed, attention, and hybrid retrieval as cold-page count grows."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.attention_shortlist import rank_page_ids, scan_query_attention
from flash_cache.kv_similarity import (
    build_packed_cold_value_index,
    rank_packed_value_page_ids,
    scan_packed_value_max_similarity,
)
from flash_cache.multi_hop_tasks import make_multi_hop_task
from flash_cache.probing import prepare_probe_caches, tokenize_task


def _positive_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed-base", type=int, default=500)
    parser.add_argument("--page-counts", type=_positive_csv, default=(12, 32, 64, 128))
    parser.add_argument("--hop-depth", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--task-variant", type=int, default=4)
    parser.add_argument("--coarse-k", type=int, default=16)
    parser.add_argument("--final-k", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if min(args.coarse_k, args.final_k, args.repeats) < 1 or args.warmup < 0:
        parser.error("K and repeat values must be positive; warmup cannot be negative")
    if args.final_k > args.coarse_k:
        parser.error("--final-k cannot exceed --coarse-k")
    if any(count < args.hop_depth + 2 for count in args.page_counts):
        parser.error("every page count must leave room for at least two distractors")
    return args


def _timed_cuda(call: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def _repeat_timed(
    call: Callable[[], Any],
    warmup: int,
    repeats: int,
) -> tuple[Any, list[float]]:
    result = None
    for _ in range(warmup):
        result, _ = _timed_cuda(call)
    latencies: list[float] = []
    for _ in range(repeats):
        result, latency_ms = _timed_cuda(call)
        latencies.append(latency_ms)
    return result, latencies


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _cache_bytes(cache: Any) -> int:
    total = 0
    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return total


def _packed_bytes(index: Any) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in index.normalized_layer_values)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
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
    for size_index, page_count in enumerate(args.page_counts):
        task = make_multi_hop_task(
            seed=args.seed_base + size_index,
            block_count=page_count,
            hop_depth=args.hop_depth,
            variant=args.task_variant,
        )
        tokenized = tokenize_task(tokenizer, task, device, prompt_format=args.prompt_format)
        prepared, prepare_latency_ms = _timed_cuda(
            lambda: prepare_probe_caches(model, tokenized, position_policy="original")
        )
        packed_index, packed_build_latency_ms = _timed_cuda(
            lambda: build_packed_cold_value_index(prepared)
        )
        packed_scan, packed_latencies = _repeat_timed(
            lambda: scan_packed_value_max_similarity(tokenized, prepared, packed_index),
            args.warmup,
            args.repeats,
        )
        assert packed_scan is not None
        effective_coarse_k = min(args.coarse_k, page_count)
        effective_final_k = min(args.final_k, effective_coarse_k)
        coarse_ids = rank_packed_value_page_ids(packed_scan, effective_coarse_k)
        packed_final_ids = rank_packed_value_page_ids(packed_scan, effective_final_k)

        subset_attention, subset_latencies = _repeat_timed(
            lambda: scan_query_attention(
                model,
                tokenized,
                prepared,
                selected_page_ids=coarse_ids,
            ),
            args.warmup,
            args.repeats,
        )
        full_attention, full_attention_latencies = _repeat_timed(
            lambda: scan_query_attention(model, tokenized, prepared),
            args.warmup,
            args.repeats,
        )
        hybrid_ids = rank_page_ids(subset_attention.scores, "all_query_mass", effective_final_k)
        attention_ids = rank_page_ids(full_attention.scores, "all_query_mass", effective_final_k)
        relevant = set(task.relevant_block_ids)
        cold_page_token_count = sum(int(ids.shape[-1]) for ids in tokenized.block_ids)
        hybrid_latencies = [
            packed + subset for packed, subset in zip(packed_latencies, subset_latencies, strict=True)
        ]
        row = {
            "page_count": page_count,
            "seed": task.seed,
            "hop_depth": task.hop_depth,
            "relevant_page_ids": list(task.relevant_block_ids),
            "cold_page_token_count": cold_page_token_count,
            "mean_page_token_count": cold_page_token_count / page_count,
            "query_token_count": int(tokenized.recent_prefix_ids.shape[-1]),
            "coarse_k": effective_coarse_k,
            "final_k": effective_final_k,
            "packed_coarse_all_relevant": relevant.issubset(coarse_ids),
            "packed_final_all_relevant": relevant.issubset(packed_final_ids),
            "hybrid_final_all_relevant": relevant.issubset(hybrid_ids),
            "attention_final_all_relevant": relevant.issubset(attention_ids),
            "prepare_latency_ms": prepare_latency_ms,
            "packed_build_latency_ms": packed_build_latency_ms,
            "cold_kv_bytes": sum(_cache_bytes(cache) for cache in prepared.cold_blocks),
            "packed_value_index_bytes": _packed_bytes(packed_index),
            "packed_scan_latency_ms": _latency_summary(packed_latencies),
            "subset_attention_latency_ms": _latency_summary(subset_latencies),
            "hybrid_scan_latency_ms": _latency_summary(hybrid_latencies),
            "full_attention_latency_ms": _latency_summary(full_attention_latencies),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del prepared, packed_index, packed_scan, subset_attention, full_attention
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "model": args.model,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "page_counts": list(args.page_counts),
        "hop_depth": args.hop_depth,
        "coarse_k": args.coarse_k,
        "final_k": args.final_k,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
