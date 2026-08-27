"""Load Qwen3.5 on the selected GPU and persist its actual cache structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from flash_cache.cache_inspection import inspect_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase0/qwen35_cache.json"))
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
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).to("cuda:0")
    model.eval()

    encoded = tokenizer(
        "Flash Cache phase zero checks whether cached state can be reconstructed exactly.",
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {name: value.to("cuda:0") for name, value in encoded.items()}

    with torch.inference_mode():
        outputs = model(**encoded, use_cache=True, return_dict=True)

    report = inspect_cache(outputs.past_key_values)
    report.update(
        {
            "model": args.model,
            "gpu": gpu_name,
            "input_tokens": int(encoded["input_ids"].shape[-1]),
            "torch_version": torch.__version__,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
