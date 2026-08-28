"""Use packed KV retrieval and model-written query substitutions for unknown-depth chains."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.answer_scoring import contains_asserted_answer
from flash_cache.attention_shortlist import rank_page_ids, scan_query_attention
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
from flash_cache.kv_similarity import (
    KV_SIMILARITY_METRICS,
    build_packed_cold_value_index,
    rank_kv_similarity_page_ids,
    rank_packed_value_page_ids,
    scan_kv_value_similarity,
    scan_packed_value_max_similarity,
)
from flash_cache.multi_hop_tasks import make_multi_hop_task
from flash_cache.probing import (
    PreparedProbeCaches,
    prepare_baseline_cache,
    prepare_probe_caches,
    rollout,
    tokenize_task,
)
from flash_cache.synthetic import contains_answer_text
from flash_cache.token_index import (
    build_cold_token_index,
    rank_token_overlap_page_ids,
    scan_query_token_overlap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=503)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--hop-depth", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--task-variant", type=int, default=4)
    parser.add_argument("--retrieval-k", type=int, default=16)
    parser.add_argument(
        "--retriever",
        choices=("packed", "attention", "union", "kv_union", "token"),
        default="union",
    )
    parser.add_argument("--attention-metric", default="all_query_mass")
    parser.add_argument("--max-navigation-steps", type=int, default=5)
    parser.add_argument("--navigation-horizon", type=int, default=32)
    parser.add_argument("--answer-horizon", type=int, default=48)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.retrieval_k <= args.blocks:
        parser.error("--retrieval-k must be between one and --blocks")
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
    current_question: str,
    page_texts: tuple[str, ...],
    horizon: int,
    device: torch.device,
    repeated_output: str | None = None,
) -> str:
    user_message = (
        make_navigation_repair_user_message(current_question, page_texts, repeated_output)
        if repeated_output is not None
        else make_navigation_user_message(current_question, page_texts)
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": NAVIGATION_SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
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
    original_tokenized = tokenize_task(
        tokenizer,
        task,
        device,
        prompt_format=args.prompt_format,
    )
    original_prepared, cold_prepare_latency_ms = _timed_cuda(
        lambda: prepare_probe_caches(model, original_tokenized, position_policy="original")
    )
    packed_index, packed_build_latency_ms = _timed_cuda(
        lambda: build_packed_cold_value_index(original_prepared)
    )
    token_index, token_index_build_latency_ms = _timed_cuda(
        lambda: build_cold_token_index(original_tokenized.block_ids)
    )
    # Pay one-time kernel initialization outside the measured online loop.
    scan_packed_value_max_similarity(original_tokenized, original_prepared, packed_index)
    no_page, no_page_latency_ms = _timed_cuda(
        lambda: rollout(
            model,
            original_prepared.baseline_cache,
            original_tokenized.probe_token,
            original_tokenized.probe_position,
            args.answer_horizon,
        )
    )
    no_page_text = tokenizer.decode(no_page.tokens.tolist(), skip_special_tokens=True)

    current_question = task.query_message
    seen_questions = {current_question.casefold()}
    steps: list[dict[str, Any]] = []
    final_answer_text: str | None = None
    stop_reason = "step_budget"
    online_latency_ms = 0.0

    for step_index in range(args.max_navigation_steps):
        step_task = replace_task_question(task, current_question)
        step_tokenized = tokenize_task(
            tokenizer,
            step_task,
            device,
            prompt_format=args.prompt_format,
        )
        query_prepare_latency_ms = 0.0
        scan_latency_ms = 0.0
        token_scan_latency_ms = 0.0
        packed_ranking: tuple[int, ...] = ()
        attention_latency_ms = 0.0
        kv_reference_latency_ms = 0.0
        attention_ranking: tuple[int, ...] = ()
        if args.retriever == "token":
            token_scores, token_scan_latency_ms = _timed_cuda(
                lambda: scan_query_token_overlap(
                    step_tokenized.recent_prefix_ids,
                    token_index,
                )
            )
            selected_ids = rank_token_overlap_page_ids(
                token_scores,
                args.retrieval_k,
            )
        else:
            step_baseline, query_prepare_latency_ms = _timed_cuda(
                lambda: prepare_baseline_cache(model, step_tokenized)
            )
            step_prepared = PreparedProbeCaches(
                baseline_cache=step_baseline,
                cold_blocks=original_prepared.cold_blocks,
                effective_block_positions=original_prepared.effective_block_positions,
            )
            scan, scan_latency_ms = _timed_cuda(
                lambda: scan_packed_value_max_similarity(
                    step_tokenized,
                    step_prepared,
                    packed_index,
                )
            )
            packed_ranking = rank_packed_value_page_ids(scan, len(scan.scores))
            if args.retriever in ("attention", "union"):
                attention_scan, attention_latency_ms = _timed_cuda(
                    lambda: scan_query_attention(model, step_tokenized, step_prepared)
                )
                attention_ranking = rank_page_ids(
                    attention_scan.scores,
                    args.attention_metric,
                    len(attention_scan.scores),
                )
            if args.retriever == "kv_union":
                kv_reference, kv_reference_latency_ms = _timed_cuda(
                    lambda: scan_kv_value_similarity(step_tokenized, step_prepared)
                )
                per_metric = max(1, args.retrieval_k // len(KV_SIMILARITY_METRICS))
                combined = []
                metric_rankings: list[tuple[int, ...]] = []
                for metric in KV_SIMILARITY_METRICS:
                    ranking = rank_kv_similarity_page_ids(
                        kv_reference.scores,
                        metric,
                        len(kv_reference.scores),
                    )
                    metric_rankings.append(ranking)
                    for page_id in ranking[:per_metric]:
                        if page_id not in combined:
                            combined.append(page_id)
                for ranking in metric_rankings:
                    for page_id in ranking:
                        if len(combined) >= args.retrieval_k:
                            break
                        if page_id not in combined:
                            combined.append(page_id)
                    if len(combined) >= args.retrieval_k:
                        break
                selected_ids = tuple(combined)
            elif args.retriever == "packed":
                selected_ids = packed_ranking[: args.retrieval_k]
            elif args.retriever == "attention":
                selected_ids = attention_ranking[: args.retrieval_k]
            else:
                per_source = max(1, args.retrieval_k // 2)
                combined = []
                for page_id in (*packed_ranking[:per_source], *attention_ranking[:per_source]):
                    if page_id not in combined:
                        combined.append(page_id)
                for page_id in (*packed_ranking, *attention_ranking):
                    if len(combined) >= args.retrieval_k:
                        break
                    if page_id not in combined:
                        combined.append(page_id)
                selected_ids = tuple(combined)
        page_texts = tuple(task.blocks[page_id] for page_id in sorted(selected_ids))
        generated, navigation_latency_ms = _timed_cuda(
            lambda: _navigation_generation(
                model,
                tokenizer,
                current_question,
                page_texts,
                args.navigation_horizon,
                device,
            )
        )
        decision = parse_navigation_decision(generated, current_question=current_question)
        if decision.kind == "lookup":
            canonical_content = canonicalize_lookup_entities(decision.content, page_texts)
            if canonical_content != decision.content:
                decision = NavigationDecision(
                    kind=decision.kind,
                    content=canonical_content,
                    raw_text=decision.raw_text,
                )
        repair_generated = None
        repair_latency_ms = 0.0
        repair_reason = None
        if decision.kind == "lookup" and decision.content.casefold() in seen_questions:
            repair_reason = "repeated_lookup"
        elif navigation_decision_needs_target_repair(decision, current_question):
            repair_reason = "target_type_mismatch"
        if repair_reason is not None:
            repair_generated, repair_latency_ms = _timed_cuda(
                lambda: _navigation_generation(
                    model,
                    tokenizer,
                    current_question,
                    page_texts,
                    args.navigation_horizon,
                    device,
                    repeated_output=generated,
                )
            )
            repair_decision = parse_navigation_decision(
                repair_generated,
                current_question=current_question,
            )
            if repair_decision.kind == "lookup":
                canonical_repair = canonicalize_lookup_entities(
                    repair_decision.content,
                    page_texts,
                )
                if canonical_repair != repair_decision.content:
                    repair_decision = NavigationDecision(
                        repair_decision.kind,
                        canonical_repair,
                        repair_decision.raw_text,
                    )
            decision = repair_decision
        step_latency_ms = (
            query_prepare_latency_ms
            + scan_latency_ms
            + token_scan_latency_ms
            + attention_latency_ms
            + kv_reference_latency_ms
            + navigation_latency_ms
            + repair_latency_ms
        )
        online_latency_ms += step_latency_ms
        steps.append(
            {
                "step_index": step_index,
                "current_question": current_question,
                "selected_page_ids": list(selected_ids),
                "selected_relevant_page_ids": [
                    page_id for page_id in task.relevant_block_ids if page_id in selected_ids
                ],
                "query_prepare_latency_ms": query_prepare_latency_ms,
                "packed_scan_latency_ms": scan_latency_ms,
                "token_scan_latency_ms": token_scan_latency_ms,
                "attention_scan_latency_ms": attention_latency_ms,
                "kv_reference_scan_latency_ms": kv_reference_latency_ms,
                "navigation_latency_ms": navigation_latency_ms,
                "step_online_latency_ms": step_latency_ms,
                "generated_navigation": generated,
                "repair_generated_navigation": repair_generated,
                "repair_navigation_latency_ms": repair_latency_ms,
                "repair_reason": repair_reason,
                "decision_kind": decision.kind,
                "decision_content": decision.content,
            }
        )
        if decision.kind == "answer":
            final_answer_text = decision.content
            stop_reason = "model_answer"
            break
        if decision.kind != "lookup":
            stop_reason = "invalid_action"
            break
        normalized_question = decision.content.casefold()
        if normalized_question in seen_questions:
            stop_reason = "repeated_question"
            break
        seen_questions.add(normalized_question)
        current_question = decision.content

    full_prefill_text, full_prefill_latency_ms = _timed_cuda(
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
    final_assertion = f"ANSWER: {final_answer_text}" if final_answer_text is not None else ""
    summary = {
        "seed": args.seed,
        "model": args.model,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "block_count": args.blocks,
        "hop_depth_evaluation_only": task.hop_depth,
        "question": task.query_message,
        "answer": task.answer,
        "relevant_page_ids_evaluation_only": list(task.relevant_block_ids),
        "retrieval_k": args.retrieval_k,
        "retriever": args.retriever,
        "attention_metric": args.attention_metric,
        "max_navigation_steps": args.max_navigation_steps,
        "cold_prepare_latency_ms_offline": cold_prepare_latency_ms,
        "packed_build_latency_ms_offline": packed_build_latency_ms,
        "token_index_build_latency_ms_offline": token_index_build_latency_ms,
        "steps": steps,
        "stop_reason": stop_reason,
        "final_answer": final_answer_text,
        "final_answer_phrase_present": (
            contains_answer_text(final_answer_text, task.answer_match)
            if final_answer_text is not None
            else False
        ),
        "final_answer_asserted_correct": contains_asserted_answer(task, final_assertion),
        "iterative_online_latency_ms": online_latency_ms,
        "no_page": {
            "generated_answer": no_page_text,
            "answer_correct": contains_asserted_answer(task, no_page_text),
            "online_latency_ms": no_page_latency_ms,
        },
        "full_prefill": {
            "generated_answer": full_prefill_text,
            "answer_correct": contains_asserted_answer(task, full_prefill_text),
            "online_latency_ms": full_prefill_latency_ms,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
