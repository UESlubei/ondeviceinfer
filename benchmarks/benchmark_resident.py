#!/usr/bin/env python3
"""Benchmark the custom resident Qwen3 runtime and save raw measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ondeviceinfer.generation import greedy_generate
from ondeviceinfer.weights import load_resident_model


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("benchmarks/prompts.jsonl"))
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-on-eos", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[:limit]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the resident benchmark")
    device = torch.device("cuda:0")
    dtype = DTYPES[args.dtype]
    prompts = load_prompts(args.prompts, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_resident_model(args.model, device=device, dtype=dtype)

    measurements: list[dict[str, object]] = []
    total_runs = args.warmup_runs + args.repeats
    for prompt_row in prompts:
        tokenized = tokenizer(prompt_row["prompt"], return_tensors="pt").input_ids.to(device)
        for repetition in range(total_runs):
            is_warmup = repetition < args.warmup_runs
            cache = model.make_resident_cache(
                batch_size=1,
                max_seq_len=tokenized.shape[1] + args.max_new_tokens,
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            result = greedy_generate(
                model,
                tokenized,
                max_new_tokens=args.max_new_tokens,
                cache=cache,
                eos_token_id=model.config.eos_token_id if args.stop_on_eos else None,
            )
            row = {
                "prompt_id": prompt_row["id"],
                "repetition": repetition,
                "is_warmup": is_warmup,
                "input_tokens": int(tokenized.shape[1]),
                "output_tokens": int(result.generated_ids.shape[1]),
                "prefill_ms": result.prefill_seconds * 1000,
                "ttft_ms": result.prefill_seconds * 1000,
                "decode_ms": result.decode_seconds * 1000,
                "decode_tokens_per_s": result.decode_tokens_per_second,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "kv_cache_bytes": cache.storage_bytes,
                "output_token_ids": result.generated_ids[0].tolist(),
            }
            measurements.append(row)
            label = "warmup" if is_warmup else "measure"
            print(
                f"{prompt_row['id']} {label} {repetition}: "
                f"TTFT={row['ttft_ms']:.2f} ms, "
                f"decode={row['decode_tokens_per_s']:.2f} tok/s"
            )
            del cache, result

    measured = [row for row in measurements if not row["is_warmup"]]
    report = {
        "manifest": {
            "schema_version": 1,
            "created_unix_seconds": time.time(),
            "framework": "ondeviceinfer-resident",
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "model": str(args.model.resolve()),
            "dtype": args.dtype,
            "prompt_set": str(args.prompts.resolve()),
            "prompt_set_sha256": sha256_file(args.prompts),
            "max_new_tokens": args.max_new_tokens,
            "stop_on_eos": args.stop_on_eos,
            "warmup_runs": args.warmup_runs,
            "measurement_runs": args.repeats,
        },
        "summary": {
            "measurement_count": len(measured),
            "median_ttft_ms": statistics.median(row["ttft_ms"] for row in measured),
            "median_decode_tokens_per_s": statistics.median(
                row["decode_tokens_per_s"] for row in measured
            ),
            "max_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in measured),
            "max_peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in measured),
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
