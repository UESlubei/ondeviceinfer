#!/usr/bin/env python3
"""Single-request vLLM benchmark using the same tokenized prompt set."""

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
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("benchmarks/prompts.jsonl"))
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def metric_value(metrics: object, *names: str) -> float | None:
    for name in names:
        value = getattr(metrics, name, None)
        if value is not None:
            return float(value)
    return None


def nvml_used_bytes() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    prompts = [
        json.loads(line)
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.limit is not None:
        prompts = prompts[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = LLM(
        model=str(args.model),
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        disable_log_stats=False,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
    )
    engine_memory_bytes = nvml_used_bytes()
    measurements: list[dict[str, object]] = []
    total_runs = args.warmup_runs + args.repeats
    for prompt in prompts:
        ids = tokenizer(prompt["prompt"], add_special_tokens=True).input_ids
        token_prompt = TokensPrompt(prompt_token_ids=ids)
        for repetition in range(total_runs):
            is_warmup = repetition < args.warmup_runs
            wall_start = time.perf_counter()
            request_output = llm.generate(
                [token_prompt], sampling_params=sampling, use_tqdm=False
            )[0]
            wall_seconds = time.perf_counter() - wall_start
            completion = request_output.outputs[0]
            metrics = request_output.metrics

            first_token_latency = metric_value(
                metrics, "first_token_latency"
            ) if metrics is not None else None
            if first_token_latency is None and metrics is not None:
                arrival = metric_value(metrics, "arrival_time")
                first = metric_value(metrics, "first_token_time", "first_token_ts")
                if arrival is not None and first is not None:
                    first_token_latency = first - arrival

            first_ts = metric_value(metrics, "first_token_time", "first_token_ts") if metrics else None
            last_ts = metric_value(metrics, "last_token_time", "last_token_ts") if metrics else None
            decode_seconds = (
                last_ts - first_ts
                if first_ts is not None and last_ts is not None
                else max(0.0, wall_seconds - (first_token_latency or 0.0))
            )
            output_tokens = len(completion.token_ids)
            row = {
                "prompt_id": prompt["id"],
                "repetition": repetition,
                "is_warmup": is_warmup,
                "input_tokens": len(ids),
                "output_tokens": output_tokens,
                "prefill_ms": first_token_latency * 1000 if first_token_latency is not None else None,
                "ttft_ms": first_token_latency * 1000 if first_token_latency is not None else None,
                "decode_ms": decode_seconds * 1000,
                "decode_tokens_per_s": (output_tokens - 1) / decode_seconds if decode_seconds > 0 else 0.0,
                "wall_ms": wall_seconds * 1000,
                "device_used_bytes": nvml_used_bytes(),
                "output_token_ids": list(completion.token_ids),
            }
            measurements.append(row)
            label = "warmup" if is_warmup else "measure"
            print(
                f"{prompt['id']} {label} {repetition}: "
                f"TTFT={row['ttft_ms']:.2f} ms, "
                f"decode={row['decode_tokens_per_s']:.2f} tok/s",
                flush=True,
            )

    measured = [row for row in measurements if not row["is_warmup"]]
    summary = {
        "measurement_count": len(measured),
        "median_ttft_ms": statistics.median(row["ttft_ms"] for row in measured),
        "median_decode_tokens_per_s": statistics.median(
            row["decode_tokens_per_s"] for row in measured
        ),
        "max_device_used_bytes": max(
            row["device_used_bytes"] or 0 for row in measured
        ),
    }
    report = {
        "manifest": {
            "schema_version": 1,
            "created_unix_seconds": time.time(),
            "framework": "vllm",
            "vllm_version": __import__("vllm").__version__,
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "model": str(args.model.resolve()),
            "dtype": args.dtype,
            "cpu_offload_gb": args.cpu_offload_gb,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "prompt_set_sha256": sha256_file(args.prompts),
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": True,
            "warmup_runs": args.warmup_runs,
            "measurement_runs": args.repeats,
            "engine_memory_bytes_after_init": engine_memory_bytes,
        },
        "summary": summary,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
