"""Run a fixed attention-shortlist policy on a deterministic multi-task suite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.answer_scoring import contains_asserted_answer
from flash_cache.attention_shortlist import (
    ATTENTION_METRICS,
    rank_page_ids,
    run_attention_shortlist,
    scan_query_attention,
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
    parser.add_argument("--seed-base", type=int, default=212)
    parser.add_argument("--trial-count", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--hop-depths", type=_positive_csv, default=(1, 2, 3, 4))
    parser.add_argument("--variant-count", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--metric", choices=ATTENTION_METRICS, default="all_query_mass")
    parser.add_argument("--tail-layers", type=int, default=4)
    parser.add_argument("--tail-query-tokens", type=int, default=4)
    parser.add_argument("--continuation-horizon", type=int, default=48)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.trial_count < 1:
        parser.error("--trial-count must be positive")
    if not 1 <= args.top_k <= args.blocks:
        parser.error("--top-k must be between one and --blocks")
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


def _generation_result(task: Any, text: str, latency_ms: float) -> dict[str, Any]:
    return {
        "generated_answer": text,
        "phrase_present": contains_answer_text(text, task.answer_match),
        "asserted_answer_correct": contains_asserted_answer(task, text),
        "online_latency_ms": latency_ms,
    }


def _condition_aggregate(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    return {
        "trial_count": len(rows),
        "phrase_present_count": sum(row[condition]["phrase_present"] for row in rows),
        "asserted_answer_correct_count": sum(
            row[condition]["asserted_answer_correct"] for row in rows
        ),
        "asserted_answer_accuracy": (
            sum(row[condition]["asserted_answer_correct"] for row in rows) / len(rows)
            if rows
            else None
        ),
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
        hop_depth = args.hop_depths[trial_index % len(args.hop_depths)]
        variant = trial_index % args.variant_count
        task = make_multi_hop_task(seed, args.blocks, hop_depth, variant)
        tokenized = tokenize_task(tokenizer, task, device, prompt_format=args.prompt_format)
        prepared, cold_prepare_latency_ms = _timed_cuda(
            lambda: prepare_probe_caches(model, tokenized, position_policy="original")
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
        no_page_text = tokenizer.decode(no_page.tokens.tolist(), skip_special_tokens=True)
        scan, scan_latency_ms = _timed_cuda(
            lambda: scan_query_attention(
                model,
                tokenized,
                prepared,
                tail_layer_count=args.tail_layers,
                tail_query_token_count=args.tail_query_tokens,
            )
        )
        ranking = rank_page_ids(scan.scores, args.metric, len(scan.scores))
        selected_ids = ranking[: args.top_k]
        independent, independent_latency_ms = _timed_cuda(
            lambda: run_attention_shortlist(
                model,
                tokenizer,
                task,
                tokenized,
                prepared,
                selected_ids,
                args.continuation_horizon,
            )
        )
        exact_text, exact_latency_ms = _timed_cuda(
            lambda: _exact_replay_answer(
                model,
                tokenizer,
                task,
                selected_ids,
                args.prompt_format,
                args.continuation_horizon,
                device,
            )
        )
        fixed_ids = tuple(range(args.top_k))
        fixed_text, fixed_latency_ms = _timed_cuda(
            lambda: _exact_replay_answer(
                model,
                tokenizer,
                task,
                fixed_ids,
                args.prompt_format,
                args.continuation_horizon,
                device,
            )
        )
        all_ids = tuple(range(args.blocks))
        full_text, full_latency_ms = _timed_cuda(
            lambda: _exact_replay_answer(
                model,
                tokenizer,
                task,
                all_ids,
                args.prompt_format,
                args.continuation_horizon,
                device,
            )
        )
        relevant_ids = set(task.relevant_block_ids)
        row = {
            "trial_index": trial_index,
            "seed": seed,
            "variant": variant,
            "hop_depth": hop_depth,
            "question": task.query_message,
            "answer": task.answer,
            "relevant_page_ids": list(task.relevant_block_ids),
            "selected_page_ids": list(selected_ids),
            "relevant_pages_selected": len(relevant_ids.intersection(selected_ids)),
            "all_relevant_pages_selected": relevant_ids.issubset(selected_ids),
            "relevant_page_ranks": {
                str(page_id): ranking.index(page_id) + 1 for page_id in sorted(relevant_ids)
            },
            "cold_prepare_latency_ms": cold_prepare_latency_ms,
            "attention_scan_latency_ms": scan_latency_ms,
            "attention_plus_exact_replay_latency_ms": scan_latency_ms + exact_latency_ms,
            "no_page": _generation_result(task, no_page_text, no_page_latency_ms),
            "attention_independent_kv": _generation_result(
                task, independent.generated_answer, independent_latency_ms
            ),
            "attention_exact_replay": _generation_result(task, exact_text, exact_latency_ms),
            "fixed_prefix_exact_replay": _generation_result(task, fixed_text, fixed_latency_ms),
            "full_prefill": _generation_result(task, full_text, full_latency_ms),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "trial": trial_index,
                    "depth": hop_depth,
                    "coverage": row["all_relevant_pages_selected"],
                    "no_page": row["no_page"]["asserted_answer_correct"],
                    "kv": row["attention_independent_kv"]["asserted_answer_correct"],
                    "replay": row["attention_exact_replay"]["asserted_answer_correct"],
                    "full": row["full_prefill"]["asserted_answer_correct"],
                }
            ),
            flush=True,
        )

    full_eligible_rows = [row for row in rows if row["full_prefill"]["asserted_answer_correct"]]
    conditions = (
        "no_page",
        "attention_independent_kv",
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
        "hop_depths": list(args.hop_depths),
        "top_k": args.top_k,
        "globally_fixed_metric": args.metric,
        "prompt_format": args.prompt_format,
        "all_relevant_top_k_count": sum(row["all_relevant_pages_selected"] for row in rows),
        "all_relevant_top_k_rate": sum(row["all_relevant_pages_selected"] for row in rows)
        / len(rows),
        "all_trials": {condition: _condition_aggregate(rows, condition) for condition in conditions},
        "full_prefill_correct_trials": {
            "trial_count": len(full_eligible_rows),
            "all_relevant_top_k_count": sum(
                row["all_relevant_pages_selected"] for row in full_eligible_rows
            ),
            "conditions": {
                condition: _condition_aggregate(full_eligible_rows, condition)
                for condition in conditions
            },
        },
        "attention_scan_latency_ms": {
            "mean": sum(row["attention_scan_latency_ms"] for row in rows) / len(rows),
            "min": min(row["attention_scan_latency_ms"] for row in rows),
            "max": max(row["attention_scan_latency_ms"] for row in rows),
        },
        "attention_plus_exact_replay_latency_ms": {
            "mean": sum(row["attention_plus_exact_replay_latency_ms"] for row in rows)
            / len(rows),
            "min": min(row["attention_plus_exact_replay_latency_ms"] for row in rows),
            "max": max(row["attention_plus_exact_replay_latency_ms"] for row in rows),
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
