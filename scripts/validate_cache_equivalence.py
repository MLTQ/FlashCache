"""Run text-model cache-equivalence checks and record numerical error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_cache.cache_inspection import inspect_cache
from flash_cache.equivalence import validate_cache_equivalence
from flash_cache.hybrid_cache import UnsupportedCacheSurgery, require_token_block_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase0/cache_equivalence.json"))
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

    input_ids = tokenizer(
        "Cache equivalence must reproduce the same predicted continuation after an exact restore.",
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to("cuda:0")
    report = validate_cache_equivalence(model, input_ids)

    with torch.inference_mode():
        cache = model(input_ids=input_ids[:, :-1], use_cache=True, return_dict=True).past_key_values
    structure = inspect_cache(cache)
    try:
        require_token_block_cache(structure)
        block_surgery = {"supported": True, "reason": None}
    except UnsupportedCacheSurgery as error:
        block_surgery = {"supported": False, "reason": str(error)}

    report.update(
        {
            "model": args.model,
            "gpu": gpu_name,
            "torch_version": torch.__version__,
            "cache_class": structure["cache_class"],
            "cache_layer_summary": {
                "layers": structure["layer_count"],
                "token_addressable": structure["token_addressable_layer_count"],
                "recurrent": structure["recurrent_layer_count"],
            },
            "arbitrary_token_block_surgery": block_surgery,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
