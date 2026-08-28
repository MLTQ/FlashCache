"""Evaluate packed-KV coarse retrieval followed by attention reranking and replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.answer_scoring import contains_asserted_answer
from flash_cache.attention_shortlist import rank_page_ids, run_attention_shortlist, scan_query_attention
from flash_cache.kv_similarity import (
    build_packed_cold_value_index,
    rank_packed_value_page_ids,
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
    parser.add_argument("--seed-base", type=int, default=412)
    parser.add_argument("--trial-count", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--hop-depths", type=_positive_csv, default=(1, 2, 3, 4))
    parser.add_argument("--variant-count", type=int, default=6)
    parser.add_argument("--coarse-k", type=int, default=8)
    parser.add_argument("--final-k", type=int, default=4)
    parser.add_argument("--attention-metric", default="all_query_mass")
    parser.add_argument("--continuation-horizon", type=int, default=48)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.trial_count < 1:
        parser.error("--trial-count must be positive")
    if not 1 <= args.final_k <= args.coarse_k <= args.blocks:
        parser.error("K values must satisfy 1 <= final K <= coarse K <= blocks")
    if any(depth > 4 for depth in args.hop_depths):
        parser.error("hop depths must be between one and four")
    if args.variant_count < 1 or args.continuation_horizon < 1:
        parser.error("variant count and continuation horizon must be positive")
    return args


def _timed_cuda(call: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def _render_prompt(
    tokenizer: Any,
    task: Any,
    page_ids: Sequence[int],
    prompt_format: str,
) -> torch.Tensor:
    page_text = "".join(task.blocks[page_id] for page_id in sorted(page_ids))
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
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


def _exact_replay_answer(
    model: Any,
    tokenizer: Any,
    task: Any,
    page_ids: Sequence[int],
    prompt_format: str,
    horizon: int,
    device: torch.device,
) -> str:
    prompt_ids = _render_prompt(tokenizer, task, page_ids, prompt_format).to(device)
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


def _result(task: Any, text: str, latency_ms: float) -> dict[str, Any]:
    return {
        "generated_answer": text,
        "phrase_present": contains_answer_text(text, task.answer_match),
        "asserted_answer_correct": contains_asserted_answer(task, text),
        "online_latency_ms": latency_ms,
    }


def _aggregate(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    correct = sum(row[condition]["asserted_answer_correct"] for row in rows)
    return {
        "trial_count": len(rows),
        "asserted_answer_correct_count": correct,
        "asserted_answer_accuracy": correct / len(rows) if rows else None,
        "phrase_present_count": sum(row[condition]["phrase_present"] for row in rows),
        "mean_online_latency_ms": (
            sum(row[condition]["online_latency_ms"] for row in rows) / len(rows)
            if rows
            else None
        ),
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

    rows: list[dict[str, Any]] = []
    for trial_index in range(args.trial_count):
        seed = args.seed_base + trial_index
        depth = args.hop_depths[trial_index % len(args.hop_depths)]
        variant = trial_index % args.variant_count
        task = make_multi_hop_task(seed, args.blocks, depth, variant)
        tokenized = tokenize_task(tokenizer, task, device, prompt_format=args.prompt_format)
        prepared, prepare_latency_ms = _timed_cuda(
            lambda: prepare_probe_caches(model, tokenized, position_policy="original")
        )
        packed_index, packed_build_latency_ms = _timed_cuda(
            lambda: build_packed_cold_value_index(prepared)
        )
        no_page, no_page_latency_ms = _timed_cuda(
            lambda: rollout(
                model,
                prepared.baseline_cache,
                tokenized.probe_token,
                tokenized.probe_position,
                args.continuation_horizon,
            )
        )
        packed_scan, packed_scan_latency_ms = _timed_cuda(
            lambda: scan_packed_value_max_similarity(tokenized, prepared, packed_index)
        )
        coarse_ids = rank_packed_value_page_ids(packed_scan, args.coarse_k)
        packed_final_ids = rank_packed_value_page_ids(packed_scan, args.final_k)
        subset_attention, subset_attention_latency_ms = _timed_cuda(
            lambda: scan_query_attention(
                model,
                tokenized,
                prepared,
                tail_layer_count=4,
                tail_query_token_count=4,
                selected_page_ids=coarse_ids,
            )
        )
        hybrid_ids = rank_page_ids(
            subset_attention.scores,
            args.attention_metric,
            args.final_k,
        )
        full_attention, full_attention_latency_ms = _timed_cuda(
            lambda: scan_query_attention(model, tokenized, prepared)
        )
        attention_ids = rank_page_ids(
            full_attention.scores,
            args.attention_metric,
            args.final_k,
        )

        independent, independent_latency_ms = _timed_cuda(
            lambda: run_attention_shortlist(
                model,
                tokenizer,
                task,
                tokenized,
                prepared,
                hybrid_ids,
                args.continuation_horizon,
            )
        )
        condition_ids = {
            "hybrid_exact_replay": hybrid_ids,
            "packed_exact_replay": packed_final_ids,
            "attention_exact_replay": attention_ids,
            "fixed_prefix_exact_replay": tuple(range(args.final_k)),
            "full_prefill": tuple(range(args.blocks)),
        }
        generated: dict[str, tuple[str, float]] = {}
        for name, page_ids in condition_ids.items():
            generated[name] = _timed_cuda(
                lambda page_ids=page_ids: _exact_replay_answer(
                    model,
                    tokenizer,
                    task,
                    page_ids,
                    args.prompt_format,
                    args.continuation_horizon,
                    device,
                )
            )

        relevant = set(task.relevant_block_ids)
        no_page_text = tokenizer.decode(no_page.tokens.tolist(), skip_special_tokens=True)
        row = {
            "trial_index": trial_index,
            "seed": seed,
            "variant": variant,
            "hop_depth": depth,
            "question": task.query_message,
            "answer": task.answer,
            "relevant_page_ids": list(task.relevant_block_ids),
            "coarse_page_ids": list(coarse_ids),
            "hybrid_page_ids": list(hybrid_ids),
            "packed_final_page_ids": list(packed_final_ids),
            "attention_page_ids": list(attention_ids),
            "coarse_all_relevant": relevant.issubset(coarse_ids),
            "hybrid_all_relevant": relevant.issubset(hybrid_ids),
            "packed_final_all_relevant": relevant.issubset(packed_final_ids),
            "attention_all_relevant": relevant.issubset(attention_ids),
            "prepare_latency_ms": prepare_latency_ms,
            "packed_index_build_latency_ms": packed_build_latency_ms,
            "packed_scan_latency_ms": packed_scan_latency_ms,
            "subset_attention_latency_ms": subset_attention_latency_ms,
            "full_attention_latency_ms": full_attention_latency_ms,
            "hybrid_scan_latency_ms": packed_scan_latency_ms + subset_attention_latency_ms,
            "no_page": _result(task, no_page_text, no_page_latency_ms),
            "hybrid_independent_kv": _result(
                task,
                independent.generated_answer,
                independent_latency_ms,
            ),
        }
        for name, (text, latency_ms) in generated.items():
            row[name] = _result(task, text, latency_ms)
        rows.append(row)
        print(
            json.dumps(
                {
                    "trial": trial_index,
                    "depth": depth,
                    "coarse": row["coarse_all_relevant"],
                    "hybrid": row["hybrid_all_relevant"],
                    "packed": row["packed_final_all_relevant"],
                    "attention": row["attention_all_relevant"],
                    "hybrid_answer": row["hybrid_exact_replay"]["asserted_answer_correct"],
                    "full": row["full_prefill"]["asserted_answer_correct"],
                }
            ),
            flush=True,
        )

    eligible = [row for row in rows if row["full_prefill"]["asserted_answer_correct"]]
    conditions = (
        "no_page",
        "hybrid_independent_kv",
        "hybrid_exact_replay",
        "packed_exact_replay",
        "attention_exact_replay",
        "fixed_prefix_exact_replay",
        "full_prefill",
    )
    summary = {
        "model": args.model,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "seed_base": args.seed_base,
        "trial_count": args.trial_count,
        "blocks": args.blocks,
        "coarse_k": args.coarse_k,
        "final_k": args.final_k,
        "packed_metric": "all_max_cosine",
        "attention_metric": args.attention_metric,
        "coverage": {
            name: sum(row[name] for row in rows) / len(rows)
            for name in (
                "coarse_all_relevant",
                "hybrid_all_relevant",
                "packed_final_all_relevant",
                "attention_all_relevant",
            )
        },
        "all_trials": {name: _aggregate(rows, name) for name in conditions},
        "full_prefill_correct_trials": {
            "trial_count": len(eligible),
            "conditions": {name: _aggregate(eligible, name) for name in conditions},
        },
        "scan_latency_ms": {
            name: {
                "mean": sum(row[name] for row in rows) / len(rows),
                "min": min(row[name] for row in rows),
                "max": max(row[name] for row in rows),
            }
            for name in (
                "packed_scan_latency_ms",
                "subset_attention_latency_ms",
                "hybrid_scan_latency_ms",
                "full_attention_latency_ms",
            )
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
