"""Run page-conditioned carrier-state accumulation on a multi-hop corpus."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.carrier_stream import (
    CarrierStreamResult,
    make_carrier_stream_task,
    run_carrier_stream,
)
from flash_cache.dense_cache import concatenate_caches
from flash_cache.multi_hop_tasks import make_multi_hop_task
from flash_cache.probing import flash_candidate, prepare_probe_caches, rollout, tokenize_task
from flash_cache.synthetic import contains_answer_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--hop-depth", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--task-variant", type=int, default=0)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--carrier-tokens-per-page", type=int, default=1)
    parser.add_argument(
        "--stream-mode",
        choices=("sentinel", "forced_sweep", "warmup_then_break"),
        default="sentinel",
    )
    parser.add_argument("--continuation-horizon", type=int, default=48)
    parser.add_argument("--baseline-horizon", type=int, default=48)
    parser.add_argument("--position-policy", choices=("original", "hot_slot"), default="hot_slot")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be at least one")
    if args.carrier_tokens_per_page < 1:
        parser.error("--carrier-tokens-per-page must be at least one")
    return args


def _write_steps(path: Path, steps: tuple[dict[str, object], ...]) -> None:
    with path.open("w") as output:
        for row in steps:
            output.write(json.dumps(row) + "\n")


def _result_summary(result: CarrierStreamResult) -> dict[str, object]:
    steps = result.steps
    return {
        "steps_completed": len(steps),
        "break_page_id": result.break_page_id,
        "break_page_relevant": result.break_page_relevant,
        "generated_answer": result.generated_answer,
        "answer_correct": result.answer_correct,
        "answer_source": result.answer_source,
        "relevant_page_flashes_seen": sum(bool(row["ground_truth_relevant"]) for row in steps),
        "logical_relevant_steps_seen": sorted(
            {
                int(row["logical_relevant_step"])
                for row in steps
                if row["logical_relevant_step"] is not None
            }
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

    task = make_multi_hop_task(
        seed=args.seed,
        block_count=args.blocks,
        hop_depth=args.hop_depth,
        variant=args.task_variant,
    )
    answer_tokenized = tokenize_task(tokenizer, task, device, prompt_format=args.prompt_format)
    answer_prepared = prepare_probe_caches(
        model,
        answer_tokenized,
        position_policy=args.position_policy,
    )
    baseline = rollout(
        model,
        answer_prepared.baseline_cache,
        answer_tokenized.probe_token,
        answer_tokenized.probe_position,
        args.baseline_horizon,
    )
    baseline_text = tokenizer.decode(baseline.tokens.tolist())

    if args.prompt_format == "chat":
        full_prompt_text = tokenizer.apply_chat_template(
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
        full_prompt_ids = tokenizer(
            full_prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)
    else:
        full_prompt_ids = tokenizer(
            f"{task.pinned_text}{''.join(task.blocks)}{task.recent_text}",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)
    with torch.inference_mode():
        full_prefill_output = model.generate(
            full_prompt_ids,
            attention_mask=torch.ones_like(full_prompt_ids),
            max_new_tokens=args.baseline_horizon,
            do_sample=False,
            use_cache=True,
        )
    full_prefill_text = tokenizer.decode(full_prefill_output[0, full_prompt_ids.shape[-1] :].tolist())

    original_position_prepared = (
        answer_prepared
        if args.position_policy == "original"
        else prepare_probe_caches(model, answer_tokenized, position_policy="original")
    )
    all_pages_cache = concatenate_caches(original_position_prepared.cold_blocks)
    full_corpus_cache = flash_candidate(
        original_position_prepared.baseline_cache,
        all_pages_cache,
        int(answer_tokenized.pinned_ids.shape[-1]),
    )
    full_corpus = rollout(
        model,
        full_corpus_cache,
        answer_tokenized.probe_token,
        answer_tokenized.probe_position,
        args.baseline_horizon,
    )
    full_corpus_text = tokenizer.decode(full_corpus.tokens.tolist())

    stream_task = make_carrier_stream_task(task) if args.stream_mode == "sentinel" else task
    stream_tokenized = tokenize_task(
        tokenizer,
        stream_task,
        device,
        prompt_format=args.prompt_format,
    )
    stream_prepared = prepare_probe_caches(
        model,
        stream_tokenized,
        position_policy=args.position_policy,
    )
    sentinel_ids = tokenizer(".", add_special_tokens=False)["input_ids"]
    if len(sentinel_ids) != 1:
        raise ValueError("Carrier stream requires a single-token period sentinel")
    sentinel_token_id = int(sentinel_ids[0])
    page_order = tuple(range(args.blocks)) * args.passes
    if args.stream_mode == "sentinel":
        break_after_steps: int | None = 0
    elif args.stream_mode == "warmup_then_break":
        break_after_steps = args.blocks
        if args.passes < 2:
            raise ValueError("warmup_then_break requires at least two corpus passes")
    else:
        break_after_steps = None

    torch.cuda.synchronize()
    started = time.perf_counter()
    poisoned = run_carrier_stream(
        model,
        tokenizer,
        task,
        stream_tokenized,
        stream_prepared,
        page_order,
        sentinel_token_id,
        args.continuation_horizon,
        carry_page_state=True,
        break_after_steps=break_after_steps,
        carrier_tokens_per_page=args.carrier_tokens_per_page,
    )
    torch.cuda.synchronize()
    poisoned_latency_ms = (time.perf_counter() - started) * 1000.0

    torch.cuda.synchronize()
    started = time.perf_counter()
    clean = run_carrier_stream(
        model,
        tokenizer,
        task,
        stream_tokenized,
        stream_prepared,
        page_order,
        sentinel_token_id,
        args.continuation_horizon,
        carry_page_state=False,
        break_after_steps=break_after_steps,
        carrier_tokens_per_page=args.carrier_tokens_per_page,
    )
    torch.cuda.synchronize()
    clean_latency_ms = (time.perf_counter() - started) * 1000.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_steps(args.output_dir / "poisoned_steps.jsonl", poisoned.steps)
    _write_steps(args.output_dir / "clean_steps.jsonl", clean.steps)
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
        "passes": args.passes,
        "carrier_tokens_per_page": args.carrier_tokens_per_page,
        "stream_mode": args.stream_mode,
        "page_order": list(page_order),
        "position_policy": args.position_policy,
        "prompt_format": args.prompt_format,
        "sentinel_token_id": sentinel_token_id,
        "baseline_answer": baseline_text,
        "baseline_answer_correct": contains_answer_text(baseline_text, task.answer_match),
        "full_prefill_answer": full_prefill_text,
        "full_prefill_answer_correct": contains_answer_text(full_prefill_text, task.answer_match),
        "all_flash_answer": full_corpus_text,
        "all_flash_answer_correct": contains_answer_text(full_corpus_text, task.answer_match),
        "poisoned": {
            **_result_summary(poisoned),
            "latency_ms": poisoned_latency_ms,
        },
        "clean": {
            **_result_summary(clean),
            "latency_ms": clean_latency_ms,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
