"""Run uninterrupted rotating-page free-decode and exact-replay controls."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.continuous_carrier import (
    make_continuous_carrier_task,
    replay_continuous_carrier,
    run_continuous_carrier,
)
from flash_cache.dense_cache import concatenate_caches
from flash_cache.multi_hop_tasks import make_multi_hop_task
from flash_cache.probing import flash_candidate, prepare_probe_caches, rollout, tokenize_task
from flash_cache.synthetic import contains_answer_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--hop-depth", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--task-variant", type=int, default=0)
    parser.add_argument("--carrier-tokens", type=int, default=48)
    parser.add_argument("--page-window-tokens", type=int, default=1)
    parser.add_argument("--continuation-horizon", type=int, default=48)
    parser.add_argument(
        "--carrier-prompt",
        choices=("ordinary", "continuous_instruction"),
        default="ordinary",
    )
    parser.add_argument("--position-policy", choices=("original", "hot_slot"), default="original")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.carrier_tokens < 1:
        parser.error("--carrier-tokens must be at least one")
    if args.page_window_tokens < 1:
        parser.error("--page-window-tokens must be at least one")
    if args.continuation_horizon < 1:
        parser.error("--continuation-horizon must be at least one")
    return args


def _timed_cuda(call: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def _write_rows(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def _full_prefill_answer(
    model: Any,
    tokenizer: Any,
    task: Any,
    device: torch.device,
    prompt_format: str,
    horizon: int,
) -> str:
    if prompt_format == "chat":
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": task.system_message},
                {
                    "role": "user",
                    "content": f"Archived records:\n{''.join(task.blocks)}\n{task.query_message}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        prompt_text = f"{task.pinned_text}{''.join(task.blocks)}{task.recent_text}"
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)
    with torch.inference_mode():
        generated = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=horizon,
            do_sample=False,
            use_cache=True,
        )
    return tokenizer.decode(
        generated[0, prompt_ids.shape[-1] :].tolist(),
        skip_special_tokens=True,
    )


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

    task = make_multi_hop_task(
        seed=args.seed,
        block_count=args.blocks,
        hop_depth=args.hop_depth,
        variant=args.task_variant,
    )
    ordinary_tokenized = tokenize_task(tokenizer, task, device, prompt_format=args.prompt_format)
    ordinary_prepared = prepare_probe_caches(
        model,
        ordinary_tokenized,
        position_policy=args.position_policy,
    )
    no_page = rollout(
        model,
        ordinary_prepared.baseline_cache,
        ordinary_tokenized.probe_token,
        ordinary_tokenized.probe_position,
        args.continuation_horizon,
    )
    no_page_text = tokenizer.decode(no_page.tokens.tolist(), skip_special_tokens=True)
    full_prefill_text = _full_prefill_answer(
        model,
        tokenizer,
        task,
        device,
        args.prompt_format,
        args.continuation_horizon,
    )

    original_prepared = (
        ordinary_prepared
        if args.position_policy == "original"
        else prepare_probe_caches(model, ordinary_tokenized, position_policy="original")
    )
    all_pages_cache = concatenate_caches(original_prepared.cold_blocks)
    all_flash_cache = flash_candidate(
        original_prepared.baseline_cache,
        all_pages_cache,
        int(ordinary_tokenized.pinned_ids.shape[-1]),
    )
    all_flash = rollout(
        model,
        all_flash_cache,
        ordinary_tokenized.probe_token,
        ordinary_tokenized.probe_position,
        args.continuation_horizon,
    )
    all_flash_text = tokenizer.decode(all_flash.tokens.tolist(), skip_special_tokens=True)

    carrier_task = (
        task if args.carrier_prompt == "ordinary" else make_continuous_carrier_task(task)
    )
    carrier_tokenized = tokenize_task(
        tokenizer,
        carrier_task,
        device,
        prompt_format=args.prompt_format,
    )
    carrier_prepared = prepare_probe_caches(
        model,
        carrier_tokenized,
        position_policy=args.position_policy,
    )
    poisoned, poisoned_latency_ms = _timed_cuda(
        lambda: run_continuous_carrier(
            model,
            tokenizer,
            task,
            carrier_tokenized,
            carrier_prepared,
            args.carrier_tokens,
            args.page_window_tokens,
            args.continuation_horizon,
        )
    )
    exact_replay, exact_replay_latency_ms = _timed_cuda(
        lambda: replay_continuous_carrier(
            model,
            tokenizer,
            task,
            carrier_tokenized,
            carrier_prepared,
            poisoned.processed_token_ids,
            args.continuation_horizon,
        )
    )
    carrier_no_page, carrier_no_page_latency_ms = _timed_cuda(
        lambda: replay_continuous_carrier(
            model,
            tokenizer,
            task,
            carrier_tokenized,
            carrier_prepared,
            (int(carrier_tokenized.probe_token.item()),),
            args.continuation_horizon,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "continuous_steps.jsonl", poisoned.steps)
    summary = {
        "seed": args.seed,
        "model": args.model,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "task_family": task.task_family,
        "task_variant": args.task_variant,
        "hop_depth": task.hop_depth,
        "question": task.query_message,
        "answer": task.answer,
        "block_count": args.blocks,
        "relevant_block_ids_logical_order": list(task.relevant_block_ids),
        "carrier_tokens": args.carrier_tokens,
        "page_window_tokens": args.page_window_tokens,
        "continuation_horizon": args.continuation_horizon,
        "carrier_prompt": args.carrier_prompt,
        "position_policy": args.position_policy,
        "prompt_format": args.prompt_format,
        "no_page_answer": no_page_text,
        "no_page_answer_correct": contains_answer_text(no_page_text, task.answer_match),
        "carrier_no_page_answer": carrier_no_page.generated_answer,
        "carrier_no_page_answer_correct": carrier_no_page.answer_correct,
        "carrier_no_page_latency_ms": carrier_no_page_latency_ms,
        "full_prefill_answer": full_prefill_text,
        "full_prefill_answer_correct": contains_answer_text(full_prefill_text, task.answer_match),
        "all_flash_answer": all_flash_text,
        "all_flash_answer_correct": contains_answer_text(all_flash_text, task.answer_match),
        "carrier_visible_token_ids": list(poisoned.visible_token_ids),
        "carrier_visible_text": tokenizer.decode(
            poisoned.visible_token_ids,
            skip_special_tokens=True,
        ),
        "poisoned": {
            "generated_answer": poisoned.generated_answer,
            "answer_correct": poisoned.answer_correct,
            "latency_ms": poisoned_latency_ms,
            "processed_token_count": len(poisoned.processed_token_ids),
            "page_flash_count": len(poisoned.steps),
            "relevant_page_flashes_seen": sum(
                bool(row["ground_truth_relevant"]) for row in poisoned.steps
            ),
            "max_page_conditioned_token_delta": max(
                float(row["page_conditioned_token_delta_max_abs"])
                for row in poisoned.steps
            ),
            "mean_page_conditioned_token_delta": sum(
                float(row["page_conditioned_token_delta_mean_abs"])
                for row in poisoned.steps
            )
            / len(poisoned.steps),
        },
        "exact_clean_replay": {
            "generated_answer": exact_replay.generated_answer,
            "answer_correct": exact_replay.answer_correct,
            "latency_ms": exact_replay_latency_ms,
            "token_ids_exactly_match": (
                exact_replay.processed_token_ids == poisoned.processed_token_ids
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
