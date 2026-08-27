"""Run sequential sentinel-token search across independently cached pages."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.iterative_search import (
    make_chat_miss_transition_ids,
    make_inline_miss_transition_ids,
    make_sentinel_search_task,
    run_iterative_flash_search,
)
from flash_cache.probing import advance_cache, prepare_probe_caches, tokenize_task
from flash_cache.task_families import TASK_FAMILIES, make_experiment_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--task-family", choices=TASK_FAMILIES, default="history_person")
    parser.add_argument("--task-variant", type=int, default=2)
    parser.add_argument("--target-identifier", default="X-17")
    parser.add_argument("--target-pressure", type=int, default=413)
    parser.add_argument("--gate-horizon", type=int, default=8)
    parser.add_argument("--continuation-horizon", type=int, default=40)
    parser.add_argument(
        "--miss-transition",
        choices=("inline", "chat_turn", "single"),
        default="inline",
    )
    parser.add_argument("--relevant-first-control", action="store_true")
    parser.add_argument("--position-policy", choices=("original", "hot_slot"), default="hot_slot")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


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

    source_task = make_experiment_task(
        args.seed,
        args.blocks,
        task_family=args.task_family,
        variant=args.task_variant,
        target_identifier=args.target_identifier,
        target_pressure=args.target_pressure,
    )
    search_task = make_sentinel_search_task(source_task)
    tokenized = tokenize_task(
        tokenizer,
        search_task,
        torch.device("cuda:0"),
        prompt_format=args.prompt_format,
    )
    prepared = prepare_probe_caches(model, tokenized, position_policy=args.position_policy)

    sentinel_ids = tokenizer(".", add_special_tokens=False)["input_ids"]
    if len(sentinel_ids) != 1:
        raise ValueError(f"Sentinel must encode as one token, got {sentinel_ids}")
    sentinel_token_id = int(sentinel_ids[0])
    if args.miss_transition == "inline":
        miss_transition_ids = make_inline_miss_transition_ids(
            tokenizer,
            torch.device("cuda:0"),
        )
    elif args.miss_transition == "chat_turn":
        if args.prompt_format != "chat":
            raise ValueError("chat_turn miss transitions require chat prompt format")
        miss_transition_ids = make_chat_miss_transition_ids(
            tokenizer,
            search_task,
            torch.device("cuda:0"),
        )
    else:
        miss_transition_ids = None
    if miss_transition_ids is not None and int(miss_transition_ids[0].item()) != sentinel_token_id:
        raise ValueError("Miss transition does not begin with the period sentinel")
    baseline_step = advance_cache(
        model,
        prepared.baseline_cache,
        tokenized.probe_token,
        tokenized.probe_position,
    )
    baseline_greedy_id = int(baseline_step.logits.argmax(dim=-1).item())

    torch.cuda.synchronize()
    started = time.perf_counter()
    candidate_order = None
    if args.relevant_first_control:
        candidate_order = (
            source_task.relevant_block_id,
            *(
                block_id
                for block_id in range(args.blocks)
                if block_id != source_task.relevant_block_id
            ),
        )
    result = run_iterative_flash_search(
        model,
        tokenizer,
        source_task,
        tokenized,
        prepared,
        sentinel_token_id,
        args.gate_horizon,
        args.continuation_horizon,
        miss_transition_ids=miss_transition_ids,
        candidate_order=candidate_order,
    )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "steps.jsonl").open("w") as output:
        for row in result.steps:
            output.write(json.dumps(row) + "\n")

    selected_step = result.steps[-1] if result.selected_candidate_id is not None else None
    summary = {
        "seed": source_task.seed,
        "model": args.model,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "task_family": source_task.task_family,
        "task_variant": args.task_variant,
        "target_key": source_task.target_key,
        "answer": source_task.answer,
        "block_count": args.blocks,
        "gate_horizon": args.gate_horizon,
        "relevant_block_id": source_task.relevant_block_id,
        "position_policy": args.position_policy,
        "prompt_format": args.prompt_format,
        "sentinel_token_id": sentinel_token_id,
        "sentinel_text": tokenizer.decode([sentinel_token_id]),
        "miss_transition_mode": args.miss_transition,
        "relevant_first_control": args.relevant_first_control,
        "miss_transition_token_count": (
            int(miss_transition_ids.shape[0]) if miss_transition_ids is not None else 1
        ),
        "miss_transition_text": (
            tokenizer.decode(miss_transition_ids.tolist())
            if miss_transition_ids is not None
            else tokenizer.decode([sentinel_token_id])
        ),
        "baseline_no_page_greedy_token_id": baseline_greedy_id,
        "baseline_no_page_greedy_text": tokenizer.decode([baseline_greedy_id]),
        "visited_candidate_ids": [int(row["candidate_block_id"]) for row in result.steps],
        "visited_count": len(result.steps),
        "selected_candidate_id": result.selected_candidate_id,
        "selected_ground_truth_relevant": result.selected_ground_truth_relevant,
        "selected_source_text": selected_step["source_text"] if selected_step else None,
        "relevant_page_reached": any(
            bool(row["ground_truth_relevant"]) for row in result.steps
        ),
        "generated_continuation": result.generated_continuation,
        "answer_correct": result.answer_correct,
        "false_break": (
            result.selected_candidate_id is not None
            and not bool(result.selected_ground_truth_relevant)
        ),
        "latency_ms": latency_ms,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
