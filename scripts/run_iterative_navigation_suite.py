"""Evaluate top-1 rare-token iterative navigation across held-out 128-page tasks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.answer_scoring import contains_asserted_answer
from flash_cache.diverse_navigation_tasks import (
    DIVERSE_NAVIGATION_FAMILIES,
    make_diverse_navigation_task,
)
from flash_cache.iterative_navigation import (
    NAVIGATION_SYSTEM_MESSAGE,
    NavigationDecision,
    canonicalize_lookup_entities,
    make_navigation_repair_user_message,
    make_navigation_user_message,
    navigation_decision_needs_target_repair,
    parse_navigation_decision,
    replace_task_question,
)
from flash_cache.multi_hop_tasks import make_multi_hop_task
from flash_cache.probing import prepare_baseline_cache, rollout, tokenize_task
from flash_cache.synthetic import contains_answer_text
from flash_cache.token_index import (
    build_cold_token_index,
    rank_token_overlap_page_ids,
    scan_query_token_overlap,
)


def _positive_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed-base", type=int, default=600)
    parser.add_argument("--trial-count", type=int, default=12)
    parser.add_argument("--task-set", choices=("preference", "diverse"), default="preference")
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--hop-depths", type=_positive_csv, default=(1, 2, 3, 4))
    parser.add_argument("--variant-count", type=int, default=6)
    parser.add_argument("--variant-offset", type=int, default=0)
    parser.add_argument("--retrieval-k", type=int, default=1)
    parser.add_argument("--max-document-fraction", type=float, default=0.5)
    parser.add_argument("--max-navigation-steps", type=int, default=5)
    parser.add_argument("--navigation-horizon", type=int, default=32)
    parser.add_argument("--answer-horizon", type=int, default=64)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.trial_count < 1 or args.variant_count < 1:
        parser.error("trial and variant counts must be positive")
    if not 1 <= args.retrieval_k <= args.blocks:
        parser.error("--retrieval-k must be between one and --blocks")
    if not 0.0 < args.max_document_fraction <= 1.0:
        parser.error("--max-document-fraction must be in (0, 1]")
    if min(args.max_navigation_steps, args.navigation_horizon, args.answer_horizon) < 1:
        parser.error("step and horizon values must be positive")
    return args


def _timed_cuda(call: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def _navigation_generation(
    model: Any,
    tokenizer: Any,
    question: str,
    page_texts: tuple[str, ...],
    horizon: int,
    device: torch.device,
    repeated_output: str | None = None,
) -> str:
    user_message = (
        make_navigation_repair_user_message(question, page_texts, repeated_output)
        if repeated_output is not None
        else make_navigation_user_message(question, page_texts)
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": NAVIGATION_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": user_message,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
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


def _full_prefill_answer(
    model: Any,
    tokenizer: Any,
    task: Any,
    page_ids: Sequence[int],
    prompt_format: str,
    horizon: int,
    device: torch.device,
) -> str:
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


def _condition(task: Any, text: str, latency_ms: float) -> dict[str, Any]:
    return {
        "generated_answer": text,
        "phrase_present": contains_answer_text(text, task.answer_match),
        "asserted_answer_correct": contains_asserted_answer(task, text),
        "online_latency_ms": latency_ms,
    }


def _aggregate(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    correct = sum(row[condition]["asserted_answer_correct"] for row in rows)
    phrase = sum(row[condition]["phrase_present"] for row in rows)
    return {
        "trial_count": len(rows),
        "asserted_answer_correct_count": correct,
        "asserted_answer_accuracy": correct / len(rows) if rows else None,
        "phrase_present_count": phrase,
        "phrase_present_rate": phrase / len(rows) if rows else None,
        "mean_online_latency_ms": (
            sum(row[condition]["online_latency_ms"] for row in rows) / len(rows)
            if rows
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this suite")
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
        if args.task_set == "diverse":
            task_family = DIVERSE_NAVIGATION_FAMILIES[
                trial_index % len(DIVERSE_NAVIGATION_FAMILIES)
            ]
            depth = args.hop_depths[
                (trial_index // len(DIVERSE_NAVIGATION_FAMILIES)) % len(args.hop_depths)
            ]
        else:
            task_family = "multi_hop_preference"
            depth = args.hop_depths[trial_index % len(args.hop_depths)]
        variant = (args.variant_offset + trial_index) % args.variant_count
        task = (
            make_diverse_navigation_task(
                seed,
                args.blocks,
                depth,
                task_family,
                variant,
            )
            if args.task_set == "diverse"
            else make_multi_hop_task(seed, args.blocks, depth, variant)
        )
        tokenized = tokenize_task(tokenizer, task, device, prompt_format=args.prompt_format)
        baseline_cache, baseline_prepare_latency_ms = _timed_cuda(
            lambda: prepare_baseline_cache(model, tokenized)
        )
        token_index, index_build_latency_ms = _timed_cuda(
            lambda: build_cold_token_index(
                tokenized.block_ids,
                max_document_fraction=args.max_document_fraction,
            )
        )
        no_page, no_page_latency_ms = _timed_cuda(
            lambda: rollout(
                model,
                baseline_cache,
                tokenized.probe_token,
                tokenized.probe_position,
                args.answer_horizon,
            )
        )
        no_page_text = tokenizer.decode(no_page.tokens.tolist(), skip_special_tokens=True)

        question = task.query_message
        seen = {question.casefold()}
        steps: list[dict[str, Any]] = []
        final_answer: str | None = None
        stop_reason = "step_budget"
        iterative_latency_ms = 0.0
        for step_index in range(args.max_navigation_steps):
            step_task = replace_task_question(task, question)
            step_tokenized = tokenize_task(
                tokenizer,
                step_task,
                device,
                prompt_format=args.prompt_format,
            )
            scores, retrieval_latency_ms = _timed_cuda(
                lambda: scan_query_token_overlap(step_tokenized.recent_prefix_ids, token_index)
            )
            selected_ids = rank_token_overlap_page_ids(scores, args.retrieval_k)
            page_texts = tuple(task.blocks[page_id] for page_id in sorted(selected_ids))
            generated, navigation_latency_ms = _timed_cuda(
                lambda: _navigation_generation(
                    model,
                    tokenizer,
                    question,
                    page_texts,
                    args.navigation_horizon,
                    device,
                )
            )
            decision = parse_navigation_decision(generated, current_question=question)
            if decision.kind == "lookup":
                canonical = canonicalize_lookup_entities(decision.content, page_texts)
                if canonical != decision.content:
                    decision = NavigationDecision(decision.kind, canonical, decision.raw_text)
            repair_generated = None
            repair_latency_ms = 0.0
            repair_reason = None
            if decision.kind == "lookup" and decision.content.casefold() in seen:
                repair_reason = "repeated_lookup"
            elif navigation_decision_needs_target_repair(decision, question):
                repair_reason = "target_type_mismatch"
            if repair_reason is not None:
                repair_generated, repair_latency_ms = _timed_cuda(
                    lambda: _navigation_generation(
                        model,
                        tokenizer,
                        question,
                        page_texts,
                        args.navigation_horizon,
                        device,
                        repeated_output=generated,
                    )
                )
                repair_decision = parse_navigation_decision(
                    repair_generated,
                    current_question=question,
                )
                if repair_decision.kind == "lookup":
                    repair_canonical = canonicalize_lookup_entities(
                        repair_decision.content,
                        page_texts,
                    )
                    if repair_canonical != repair_decision.content:
                        repair_decision = NavigationDecision(
                            repair_decision.kind,
                            repair_canonical,
                            repair_decision.raw_text,
                        )
                decision = repair_decision
            step_latency_ms = retrieval_latency_ms + navigation_latency_ms + repair_latency_ms
            iterative_latency_ms += step_latency_ms
            steps.append(
                {
                    "step_index": step_index,
                    "current_question": question,
                    "selected_page_ids": list(selected_ids),
                    "selected_relevant_page_ids": [
                        page_id for page_id in task.relevant_block_ids if page_id in selected_ids
                    ],
                    "expected_next_page_id_evaluation_only": (
                        task.relevant_block_ids[step_index]
                        if step_index < len(task.relevant_block_ids)
                        else None
                    ),
                    "selected_expected_next_page_evaluation_only": (
                        step_index < len(task.relevant_block_ids)
                        and task.relevant_block_ids[step_index] in selected_ids
                    ),
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "navigation_latency_ms": navigation_latency_ms,
                    "generated_navigation": generated,
                    "repair_generated_navigation": repair_generated,
                    "repair_navigation_latency_ms": repair_latency_ms,
                    "repair_reason": repair_reason,
                    "decision_kind": decision.kind,
                    "decision_content": decision.content,
                }
            )
            if decision.kind == "answer":
                final_answer = decision.content
                stop_reason = "model_answer"
                break
            if decision.kind != "lookup":
                stop_reason = "invalid_action"
                break
            normalized = decision.content.casefold()
            if normalized in seen:
                stop_reason = "repeated_question"
                break
            seen.add(normalized)
            question = decision.content

        full_text, full_latency_ms = _timed_cuda(
            lambda: _full_prefill_answer(
                model,
                tokenizer,
                task,
                tuple(range(args.blocks)),
                args.prompt_format,
                args.answer_horizon,
                device,
            )
        )
        final_text = f"ANSWER: {final_answer}" if final_answer is not None else ""
        row = {
            "trial_index": trial_index,
            "seed": seed,
            "task_family": task.task_family,
            "variant": variant,
            "hop_depth_evaluation_only": depth,
            "question": task.query_message,
            "answer": task.answer,
            "relevant_page_ids_evaluation_only": list(task.relevant_block_ids),
            "baseline_prepare_latency_ms_offline": baseline_prepare_latency_ms,
            "token_index_build_latency_ms_offline": index_build_latency_ms,
            "steps": steps,
            "stop_reason": stop_reason,
            "final_answer": final_answer,
            "iterative": {
                "generated_answer": final_answer or "",
                "phrase_present": (
                    contains_answer_text(final_answer, task.answer_match)
                    if final_answer is not None
                    else False
                ),
                "asserted_answer_correct": contains_asserted_answer(task, final_text),
                "online_latency_ms": iterative_latency_ms,
            },
            "no_page": _condition(task, no_page_text, no_page_latency_ms),
            "full_prefill": _condition(task, full_text, full_latency_ms),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "trial": trial_index,
                    "family": task.task_family,
                    "depth": depth,
                    "steps": len(steps),
                    "stop": stop_reason,
                    "iterative": row["iterative"]["asserted_answer_correct"],
                    "no_page": row["no_page"]["asserted_answer_correct"],
                    "full_phrase": row["full_prefill"]["phrase_present"],
                }
            ),
            flush=True,
        )

    by_depth: dict[str, Any] = {}
    for depth in args.hop_depths:
        depth_rows = [row for row in rows if row["hop_depth_evaluation_only"] == depth]
        by_depth[str(depth)] = {
            condition: _aggregate(depth_rows, condition)
            for condition in ("iterative", "no_page", "full_prefill")
        }
    by_family = {
        family: {
            condition: _aggregate(
                [row for row in rows if row["task_family"] == family],
                condition,
            )
            for condition in ("iterative", "no_page", "full_prefill")
        }
        for family in sorted({str(row["task_family"]) for row in rows})
    }
    summary = {
        "model": args.model,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "seed_base": args.seed_base,
        "trial_count": args.trial_count,
        "task_set": args.task_set,
        "blocks": args.blocks,
        "hop_depths": list(args.hop_depths),
        "variant_count": args.variant_count,
        "variant_offset": args.variant_offset,
        "retrieval_k": args.retrieval_k,
        "max_document_fraction": args.max_document_fraction,
        "max_navigation_steps": args.max_navigation_steps,
        "all_trials": {
            condition: _aggregate(rows, condition)
            for condition in ("iterative", "no_page", "full_prefill")
        },
        "by_depth": by_depth,
        "by_family": by_family,
        "model_answer_stop_count": sum(row["stop_reason"] == "model_answer" for row in rows),
        "all_selected_steps_ground_truth_relevant_count": sum(
            all(step["selected_relevant_page_ids"] for step in row["steps"]) for row in rows
        ),
        "all_selected_steps_in_logical_order_count": sum(
            len(row["steps"]) == row["hop_depth_evaluation_only"]
            and all(
                step["selected_expected_next_page_evaluation_only"]
                for step in row["steps"]
            )
            for row in rows
        ),
        "mean_token_index_build_latency_ms_offline": (
            sum(row["token_index_build_latency_ms_offline"] for row in rows) / len(rows)
        ),
        "mean_retrieval_latency_ms_per_step": (
            sum(step["retrieval_latency_ms"] for row in rows for step in row["steps"])
            / sum(len(row["steps"]) for row in rows)
        ),
        "answer_free_repair_count": sum(
            step["repair_reason"] is not None for row in rows for step in row["steps"]
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trials.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
