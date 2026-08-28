"""Measure answer-free query-attention page ranking and fixed top-K reconstruction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.answer_scoring import extract_archive_answer_choices, score_answer_choices
from flash_cache.attention_shortlist import (
    ATTENTION_METRICS,
    assemble_selected_archive_cache,
    rank_page_ids,
    run_attention_shortlist,
    scan_query_attention,
)
from flash_cache.multi_hop_tasks import make_multi_hop_task
from flash_cache.probing import prepare_probe_caches, rollout, tokenize_task
from flash_cache.query_refresh import refresh_query_prefix, run_query_refresh
from flash_cache.synthetic import contains_answer_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--hop-depth", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--task-variant", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--tail-layers", type=int, default=4)
    parser.add_argument("--tail-query-tokens", type=int, default=4)
    parser.add_argument("--continuation-horizon", type=int, default=48)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="2070 SUPER")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.top_k <= args.blocks:
        parser.error("--top-k must be between one and --blocks")
    if args.tail_layers < 1 or args.tail_query_tokens < 1:
        parser.error("attention tail widths must be positive")
    if args.continuation_horizon < 1:
        parser.error("--continuation-horizon must be at least one")
    return args


def _timed_cuda(call: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def _render_prompt(tokenizer: Any, task: Any, page_ids: Sequence[int], prompt_format: str) -> str:
    page_text = "".join(task.blocks[page_id] for page_id in sorted(page_ids))
    if prompt_format == "chat":
        return tokenizer.apply_chat_template(
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
    return f"{task.pinned_text}{page_text}{task.recent_text}"


def _exact_replay_answer(
    model: Any,
    tokenizer: Any,
    task: Any,
    page_ids: Sequence[int],
    prompt_format: str,
    horizon: int,
    device: torch.device,
) -> str:
    cache, probe_token, probe_position = _exact_replay_context(
        model,
        tokenizer,
        task,
        page_ids,
        prompt_format,
        device,
    )
    answer = rollout(model, cache, probe_token, probe_position, horizon)
    return tokenizer.decode(answer.tokens.tolist(), skip_special_tokens=True)


def _exact_replay_context(
    model: Any,
    tokenizer: Any,
    task: Any,
    page_ids: Sequence[int],
    prompt_format: str,
    device: torch.device,
) -> tuple[Any, torch.Tensor, int]:
    """Prefill selected source text and return the immutable answer-start state."""
    prompt = _render_prompt(tokenizer, task, page_ids, prompt_format)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if prompt_ids.shape[-1] < 2:
        raise ValueError("Rendered prompt must contain a prefix and final probe token")
    prefix_ids = prompt_ids[:, :-1]
    probe_token = prompt_ids[:, -1:]
    with torch.inference_mode():
        outputs = model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            return_dict=True,
        )
    return outputs.past_key_values, probe_token, int(prefix_ids.shape[-1])


def _score_exact_replay(
    model: Any,
    tokenizer: Any,
    task: Any,
    page_ids: Sequence[int],
    prompt_format: str,
    device: torch.device,
    answer_choices: Sequence[str],
) -> dict[str, Any]:
    cache, probe_token, probe_position = _exact_replay_context(
        model,
        tokenizer,
        task,
        page_ids,
        prompt_format,
        device,
    )
    return score_answer_choices(
        model,
        tokenizer,
        cache,
        probe_token,
        probe_position,
        answer_choices,
        task.answer,
    )


def _score_independent_kv(
    model: Any,
    tokenizer: Any,
    task: Any,
    tokenized: Any,
    prepared: Any,
    page_ids: Sequence[int],
    answer_choices: Sequence[str],
) -> dict[str, Any]:
    archive_cache = assemble_selected_archive_cache(tokenized, prepared, page_ids)
    refreshed_cache = refresh_query_prefix(
        model,
        archive_cache,
        tokenized.recent_prefix_ids,
        tokenized.recent_prefix_positions,
    )
    return score_answer_choices(
        model,
        tokenizer,
        refreshed_cache,
        tokenized.probe_token,
        tokenized.probe_position,
        answer_choices,
        task.answer,
    )


def _selection_key(page_ids: Sequence[int]) -> str:
    return ",".join(str(page_id) for page_id in sorted(page_ids))


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
    answer_choices = extract_archive_answer_choices(task)
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
    no_page_answer_scores = score_answer_choices(
        model,
        tokenizer,
        prepared.baseline_cache,
        tokenized.probe_token,
        tokenized.probe_position,
        answer_choices,
        task.answer,
    )
    all_page_refresh, all_page_refresh_latency_ms = _timed_cuda(
        lambda: run_query_refresh(
            model,
            tokenizer,
            task,
            tokenized,
            prepared,
            args.continuation_horizon,
        )
    )
    scan, attention_scan_latency_ms = _timed_cuda(
        lambda: scan_query_attention(
            model,
            tokenized,
            prepared,
            tail_layer_count=args.tail_layers,
            tail_query_token_count=args.tail_query_tokens,
        )
    )

    relevant_ids = set(task.relevant_block_ids)
    metric_rankings: dict[str, Any] = {}
    selections: dict[str, tuple[int, ...]] = {}
    for metric in ATTENTION_METRICS:
        ranking = rank_page_ids(scan.scores, metric, len(scan.scores))
        selected = ranking[: args.top_k]
        selection_key = _selection_key(selected)
        selections[selection_key] = selected
        metric_rankings[metric] = {
            "ranking": list(ranking),
            "selected_page_ids": list(selected),
            "selection_key": selection_key,
            "relevant_pages_selected": len(relevant_ids.intersection(selected)),
            "all_relevant_pages_selected": relevant_ids.issubset(selected),
            "relevant_page_ranks": {
                str(page_id): ranking.index(page_id) + 1 for page_id in sorted(relevant_ids)
            },
        }

    fixed_prefix_ids = tuple(range(args.top_k))
    oracle_ids = list(sorted(relevant_ids))
    oracle_ids.extend(
        page_id
        for page_id in range(args.blocks)
        if page_id not in relevant_ids and len(oracle_ids) < args.top_k
    )
    controls = {
        "fixed_prefix": fixed_prefix_ids,
        "oracle_relevant_plus_fill": tuple(oracle_ids),
    }
    for selected in controls.values():
        selections[_selection_key(selected)] = selected

    selection_results: dict[str, Any] = {}
    for selection_key, selected in selections.items():
        kv_result, kv_latency_ms = _timed_cuda(
            lambda selected=selected: run_attention_shortlist(
                model,
                tokenizer,
                task,
                tokenized,
                prepared,
                selected,
                args.continuation_horizon,
            )
        )
        replay_text, replay_latency_ms = _timed_cuda(
            lambda selected=selected: _exact_replay_answer(
                model,
                tokenizer,
                task,
                selected,
                args.prompt_format,
                args.continuation_horizon,
                device,
            )
        )
        kv_answer_scores = _score_independent_kv(
            model,
            tokenizer,
            task,
            tokenized,
            prepared,
            selected,
            answer_choices,
        )
        replay_answer_scores = _score_exact_replay(
            model,
            tokenizer,
            task,
            selected,
            args.prompt_format,
            device,
            answer_choices,
        )
        selection_results[selection_key] = {
            "selected_page_ids": list(selected),
            "relevant_pages_selected": len(relevant_ids.intersection(selected)),
            "all_relevant_pages_selected": relevant_ids.issubset(selected),
            "independent_kv_refresh": {
                "generated_answer": kv_result.generated_answer,
                "answer_correct": kv_result.answer_correct,
                "shortlist_latency_ms": kv_latency_ms,
                "total_with_attention_scan_latency_ms": attention_scan_latency_ms + kv_latency_ms,
                "selected_page_token_count": kv_result.selected_page_token_count,
                "final_cache_token_count": kv_result.final_cache_token_count,
                "answer_choice_scores": kv_answer_scores,
            },
            "exact_text_replay": {
                "generated_answer": replay_text,
                "answer_correct": contains_answer_text(replay_text, task.answer_match),
                "shortlist_latency_ms": replay_latency_ms,
                "total_with_attention_scan_latency_ms": attention_scan_latency_ms + replay_latency_ms,
                "answer_choice_scores": replay_answer_scores,
            },
        }

    all_page_ids = tuple(range(args.blocks))
    full_prefill_text, full_prefill_latency_ms = _timed_cuda(
        lambda: _exact_replay_answer(
            model,
            tokenizer,
            task,
            all_page_ids,
            args.prompt_format,
            args.continuation_horizon,
            device,
        )
    )
    full_prefill_answer_scores = _score_exact_replay(
        model,
        tokenizer,
        task,
        all_page_ids,
        args.prompt_format,
        device,
        answer_choices,
    )
    all_page_refresh_answer_scores = _score_independent_kv(
        model,
        tokenizer,
        task,
        tokenized,
        prepared,
        all_page_ids,
        answer_choices,
    )

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
        "top_k": args.top_k,
        "continuation_horizon": args.continuation_horizon,
        "prompt_format": args.prompt_format,
        "pinned_token_count": int(tokenized.pinned_ids.shape[-1]),
        "cold_page_token_count": sum(int(ids.shape[-1]) for ids in tokenized.block_ids),
        "recent_query_prefix_token_count": int(tokenized.recent_prefix_ids.shape[-1]),
        "answer_choices": list(answer_choices),
        "cold_prepare_latency_ms": cold_prepare_latency_ms,
        "attention_scan": {
            "online_latency_ms": attention_scan_latency_ms,
            "archive_token_count": scan.archive_token_count,
            "query_token_count": scan.query_token_count,
            "layer_count": scan.layer_count,
            "tail_layer_count": scan.tail_layer_count,
            "tail_query_token_count": scan.tail_query_token_count,
            "page_scores": [score.to_dict() for score in scan.scores],
            "metric_rankings": metric_rankings,
        },
        "no_page": {
            "generated_answer": no_page_text,
            "answer_correct": contains_answer_text(no_page_text, task.answer_match),
            "online_latency_ms": no_page_latency_ms,
            "answer_choice_scores": no_page_answer_scores,
        },
        "all_page_query_refresh": {
            "generated_answer": all_page_refresh.generated_answer,
            "answer_correct": all_page_refresh.answer_correct,
            "online_latency_ms": all_page_refresh_latency_ms,
            "answer_choice_scores": all_page_refresh_answer_scores,
        },
        "controls": {name: _selection_key(ids) for name, ids in controls.items()},
        "selection_results": selection_results,
        "full_prefill_all_pages": {
            "generated_answer": full_prefill_text,
            "answer_correct": contains_answer_text(full_prefill_text, task.answer_match),
            "online_latency_ms": full_prefill_latency_ms,
            "answer_choice_scores": full_prefill_answer_scores,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
