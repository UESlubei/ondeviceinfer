#!/usr/bin/env python3
"""Benchmark Qwen3 INT8 weight-only sync, prefetch, or resident execution."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ondeviceinfer.generation import greedy_generate
from ondeviceinfer.quantized_model import QuantizedQwen3ForCausalLM
from ondeviceinfer.runtime.quantized import (
    QuantizedPrefetchOffloadRuntime,
    QuantizedResidentRuntime,
    QuantizedSyncOffloadRuntime,
    estimate_quantized_gpu_weight_bytes,
    load_quantized_cpu_weight_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--runtime", choices=["sync", "prefetch", "resident"], required=True
    )
    parser.add_argument("--gpu-layer-budget", type=int, default=2)
    parser.add_argument("--prompt", default="你好，请用一句话介绍南京。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-new-tokens", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-new-tokens", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def create_runtime(args: argparse.Namespace, store: object, device: torch.device):
    if args.runtime == "sync":
        return QuantizedSyncOffloadRuntime(
            store, device=device, gpu_layer_budget=args.gpu_layer_budget
        )
    if args.runtime == "prefetch":
        return QuantizedPrefetchOffloadRuntime(
            store, device=device, gpu_layer_budget=args.gpu_layer_budget
        )
    return QuantizedResidentRuntime(store, device=device)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.max_new_tokens <= 0 or args.profile_new_tokens <= 0:
        raise ValueError("token counts must be positive")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    load_start = time.perf_counter()
    store = load_quantized_cpu_weight_store(
        args.model, pin_memory=True, progress=True
    )
    store_load_seconds = time.perf_counter() - load_start
    effective_budget = (
        store.config.num_hidden_layers
        if args.runtime == "resident"
        else args.gpu_layer_budget
    )
    estimate = estimate_quantized_gpu_weight_bytes(store, effective_budget)
    device_total = torch.cuda.get_device_properties(device).total_memory
    if estimate > int(device_total * 0.92):
        raise RuntimeError(
            f"Estimated GPU weight bytes {estimate} exceed the 92% safety limit"
        )

    runtime = create_runtime(args, store, device)
    model = QuantizedQwen3ForCausalLM(store, runtime, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    warmups: list[dict[str, float]] = []
    for repetition in range(args.warmup_runs):
        cache = model.make_resident_cache(
            batch_size=1,
            max_seq_len=input_ids.shape[1] + args.warmup_new_tokens,
        )
        runtime.stats.reset_forward_counters()
        result = greedy_generate(
            model,
            input_ids,
            max_new_tokens=args.warmup_new_tokens,
            cache=cache,
            eos_token_id=None,
        )
        torch.cuda.synchronize(device)
        warmups.append(
            {
                "repetition": repetition,
                "prefill_ms": result.prefill_seconds * 1000,
                "decode_tokens_per_s": result.decode_tokens_per_second,
            }
        )
        print(
            f"warmup {repetition}: {result.decode_tokens_per_second:.3f} tok/s",
            flush=True,
        )

    measurements: list[dict[str, object]] = []
    for repetition in range(args.repeats):
        cache = model.make_resident_cache(
            batch_size=1,
            max_seq_len=input_ids.shape[1] + args.max_new_tokens,
        )
        runtime.stats.reset_forward_counters()
        model.compute_stats.enabled = False
        model.compute_stats.reset()
        torch.cuda.reset_peak_memory_stats(device)
        result = greedy_generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            cache=cache,
            eos_token_id=None,
        )
        torch.cuda.synchronize(device)
        copy_seconds = (
            runtime.stats.synchronous_copy_seconds
            if args.runtime == "sync"
            else runtime.stats.async_copy_seconds()
        )
        ids = result.generated_ids[0].tolist()
        row: dict[str, object] = {
            "repetition": repetition,
            "status": "success",
            "input_tokens": int(input_ids.shape[1]),
            "output_tokens": len(ids),
            "prefill_ms": result.prefill_seconds * 1000,
            "ttft_ms": result.prefill_seconds * 1000,
            "decode_ms": result.decode_seconds * 1000,
            "decode_tokens_per_s": result.decode_tokens_per_second,
            "weight_h2d_bytes": runtime.stats.weight_h2d_bytes,
            "copy_seconds_sum": copy_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "dequantized_weight_bytes": model.compute_stats.dequantized_weight_bytes,
            "linear_calls": model.compute_stats.linear_calls,
            "output_token_ids": ids,
            "output_text": tokenizer.decode(ids, skip_special_tokens=False),
        }
        measurements.append(row)
        print(
            f"measure {repetition}: TTFT={row['ttft_ms']:.2f} ms, "
            f"decode={row['decode_tokens_per_s']:.3f} tok/s, "
            f"peak={row['peak_allocated_bytes'] / 2**30:.2f} GiB",
            flush=True,
        )

    # CUDA-event profiling uses a separate short generation so event creation
    # overhead never contaminates the formal throughput rows above.
    profile_cache = model.make_resident_cache(
        batch_size=1,
        max_seq_len=input_ids.shape[1] + args.profile_new_tokens,
    )
    runtime.stats.reset_forward_counters()
    model.compute_stats.reset()
    model.compute_stats.enabled = True
    profile_result = greedy_generate(
        model,
        input_ids,
        max_new_tokens=args.profile_new_tokens,
        cache=profile_cache,
        eos_token_id=None,
    )
    torch.cuda.synchronize(device)
    profile = {
        "output_tokens": args.profile_new_tokens,
        "prefill_ms": profile_result.prefill_seconds * 1000,
        "decode_ms": profile_result.decode_seconds * 1000,
        "weight_h2d_bytes": runtime.stats.weight_h2d_bytes,
        "copy_seconds_sum": (
            runtime.stats.synchronous_copy_seconds
            if args.runtime == "sync"
            else runtime.stats.async_copy_seconds()
        ),
        "dequant_seconds_sum": model.compute_stats.dequant_seconds(),
        "gemm_seconds_sum": model.compute_stats.gemm_seconds(),
        "linear_calls": model.compute_stats.linear_calls,
        "dequantized_weight_bytes": model.compute_stats.dequantized_weight_bytes,
    }
    model.compute_stats.enabled = False

    decode_values = [float(row["decode_tokens_per_s"]) for row in measurements]
    ttft_values = [float(row["ttft_ms"]) for row in measurements]
    report = {
        "manifest": {
            "schema_version": 1,
            "model": str(args.model.resolve()),
            "source_model": store.manifest.get("source_model"),
            "source_revision": store.manifest.get("source_revision"),
            "runtime": args.runtime,
            "gpu_layer_budget": effective_budget,
            "resident_layers": runtime.resident_count,
            "weight_dtype": "int8_per_output_channel_symmetric",
            "compute_dtype": "float16",
            "prompt": args.prompt,
            "input_tokens": int(input_ids.shape[1]),
            "max_new_tokens": args.max_new_tokens,
            "warmup_new_tokens": args.warmup_new_tokens,
            "warmup_runs": args.warmup_runs,
            "measurement_runs": args.repeats,
            "all_cpu_weights_pinned": store.all_pinned,
            "cpu_weight_bytes": store.total_bytes,
            "decoder_weight_bytes": store.decoder_bytes,
            "store_load_seconds": store_load_seconds,
            "estimated_gpu_weight_bytes": estimate,
            "startup_h2d_bytes": runtime.stats.startup_h2d_bytes,
        },
        "summary": {
            "median_ttft_ms": statistics.median(ttft_values),
            "min_ttft_ms": min(ttft_values),
            "max_ttft_ms": max(ttft_values),
            "median_decode_tokens_per_s": statistics.median(decode_values),
            "min_decode_tokens_per_s": min(decode_values),
            "max_decode_tokens_per_s": max(decode_values),
            "max_peak_allocated_bytes": max(
                int(row["peak_allocated_bytes"]) for row in measurements
            ),
            "max_peak_reserved_bytes": max(
                int(row["peak_reserved_bytes"]) for row in measurements
            ),
        },
        "warmups": warmups,
        "measurements": measurements,
        "profile_sample": profile,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
