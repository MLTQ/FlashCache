"""Compare refreshed-query cold KV against stale insertion and full prefill."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.dense_cache import concatenate_caches
from flash_cache.multi_hop_tasks import make_multi_hop_task
from flash_cache.probing import flash_candidate, prepare_probe_caches, rollout, tokenize_task
from flash_cache.query_refresh import run_query_refresh
from flash_cache.synthetic import contains_answer_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--hop-depth", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--task-variant", type=int, default=0)
    parser.add_argument("--continuation-horizon", type=int, default=48)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.continuation_horizon < 1:
        parser.error("--continuation-horizon must be at least one")
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

    all_pages_cache = concatenate_caches(prepared.cold_blocks)
    stale_cache = flash_candidate(
        prepared.baseline_cache,
        all_pages_cache,
        int(tokenized.pinned_ids.shape[-1]),
    )
    stale, stale_latency_ms = _timed_cuda(
        lambda: rollout(
            model,
            stale_cache,
            tokenized.probe_token,
            tokenized.probe_position,
            args.continuation_horizon,
        )
    )
    stale_text = tokenizer.decode(stale.tokens.tolist(), skip_special_tokens=True)

    refreshed, refreshed_latency_ms = _timed_cuda(
        lambda: run_query_refresh(
            model,
            tokenizer,
            task,
            tokenized,
            prepared,
            args.continuation_horizon,
        )
    )
    full_prefill_text, full_prefill_latency_ms = _timed_cuda(
        lambda: _full_prefill_answer(
            model,
            tokenizer,
            task,
            device,
            args.prompt_format,
            args.continuation_horizon,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        "continuation_horizon": args.continuation_horizon,
        "prompt_format": args.prompt_format,
        "pinned_token_count": int(tokenized.pinned_ids.shape[-1]),
        "cold_page_token_count": sum(int(ids.shape[-1]) for ids in tokenized.block_ids),
        "recent_query_prefix_token_count": int(tokenized.recent_prefix_ids.shape[-1]),
        "cold_prepare_latency_ms": cold_prepare_latency_ms,
        "no_page": {
            "generated_answer": no_page_text,
            "answer_correct": contains_answer_text(no_page_text, task.answer_match),
            "online_latency_ms": no_page_latency_ms,
        },
        "stale_query_all_pages": {
            "generated_answer": stale_text,
            "answer_correct": contains_answer_text(stale_text, task.answer_match),
            "online_latency_ms": stale_latency_ms,
        },
        "query_refresh": {
            "generated_answer": refreshed.generated_answer,
            "answer_correct": refreshed.answer_correct,
            "online_latency_ms": refreshed_latency_ms,
            "cold_archive_token_count": refreshed.cold_archive_token_count,
            "refreshed_query_token_count": refreshed.refreshed_query_token_count,
            "final_cache_token_count": refreshed.final_cache_token_count,
        },
        "full_prefill": {
            "generated_answer": full_prefill_text,
            "answer_correct": contains_answer_text(full_prefill_text, task.answer_match),
            "online_latency_ms": full_prefill_latency_ms,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
