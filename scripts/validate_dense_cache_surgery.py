"""Prove that same-order cache splitting and reassembly is numerically exact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.dense_cache import cache_tensor_error, concatenate_caches, slice_cache, spans_from_boundaries
from flash_cache.equivalence import logit_error
from flash_cache.hybrid_cache import clone_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase0/dense_cache_surgery.json"))
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

    text = (
        "Pinned instructions establish the experiment. "
        "Historical block alpha describes a blue turbine. "
        "Historical block beta records a pressure limit. "
        "Recent context asks what the cache should reproduce exactly."
    )
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda:0")
    with torch.inference_mode():
        prefill = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True, return_dict=True)

    token_count = int(input_ids.shape[-1])
    cuts = [0, token_count // 3, (2 * token_count) // 3, token_count]
    spans = spans_from_boundaries(cuts)
    blocks = [slice_cache(prefill.past_key_values, start, stop) for start, stop in spans]
    reconstructed = concatenate_caches(blocks)
    tensor_error = cache_tensor_error(prefill.past_key_values, reconstructed)

    probe_id = tokenizer(" Therefore", return_tensors="pt", add_special_tokens=False)["input_ids"][:, :1].to("cuda:0")
    attention_mask = torch.ones((1, token_count + 1), dtype=torch.long, device="cuda:0")
    with torch.inference_mode():
        reference = model(
            input_ids=probe_id,
            attention_mask=attention_mask,
            past_key_values=clone_cache(prefill.past_key_values),
            use_cache=True,
            return_dict=True,
        ).logits[:, -1, :]
        candidate = model(
            input_ids=probe_id,
            attention_mask=attention_mask,
            past_key_values=clone_cache(reconstructed),
            use_cache=True,
            return_dict=True,
        ).logits[:, -1, :]

    probe_error = logit_error(reference, candidate)
    report = {
        "model": args.model,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "input_tokens": token_count,
        "block_spans": [list(span) for span in spans],
        "cache_tensor_error": tensor_error,
        "probe_logit_error": probe_error,
        "reference_argmax": int(reference.argmax(dim=-1).item()),
        "reconstructed_argmax": int(candidate.argmax(dim=-1).item()),
        "passes_exact_reconstruction": tensor_error["max_abs"] == 0.0 and probe_error["max_abs"] == 0.0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passes_exact_reconstruction"]:
        raise RuntimeError("Same-order cache reconstruction was not exact")


if __name__ == "__main__":
    main()
