#!/usr/bin/env python3
"""Benchmark Qwen3-14B layer budgets while loading the pinned store once."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ondeviceinfer.generation import greedy_generate
from ondeviceinfer.offload_model import OffloadedQwen3ForCausalLM
from ondeviceinfer.runtime.offload import (
    PrefetchOffloadRuntime,
    SyncOffloadRuntime,
    estimate_gpu_weight_bytes,
    load_cpu_weight_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--budgets", default="1,2,4,8,16,24,32")
    parser.add_argument("--prompt", default="你好，请用一句话介绍南京。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-new-tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expected-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def expected_ids_from(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "generated_ids" in data:
        return data["generated_ids"]
    if "reference_generated_ids" in data:
        return data["reference_generated_ids"]
    return data["measurements"][-1]["output_token_ids"]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    budgets = [int(value) for value in args.budgets.split(",")]
    reference_ids = expected_ids_from(args.expected_json)

    load_start = time.perf_counter()
    store = load_cpu_weight_store(args.model, dtype=torch.float16, pin_memory=True)
    store_load_seconds = time.perf_counter() - load_start
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    device_total = torch.cuda.get_device_properties(device).total_memory
    measurements: list[dict[str, object]] = []

    for budget in budgets:
        estimate = estimate_gpu_weight_bytes(store, budget)
        if estimate > int(device_total * 0.92):
            measurements.append(
                {
                    "gpu_layer_budget": budget,
                    "status": "oom_risk_skipped",
                    "estimated_gpu_weight_bytes": estimate,
                }
            )
            continue
        runtime_name = "sync" if budget == 1 else "prefetch"
        runtime_class = SyncOffloadRuntime if budget == 1 else PrefetchOffloadRuntime
        runtime = runtime_class(store, device=device, gpu_layer_budget=budget)
        model = OffloadedQwen3ForCausalLM(store, runtime, device=device)

        # Short warmup; it is reported but not included in formal statistics.
        warmup_cache = model.make_resident_cache(
            batch_size=1,
            max_seq_len=input_ids.shape[1] + args.warmup_new_tokens,
        )
        runtime.stats.reset_forward_counters()
        warmup = greedy_generate(
            model,
            input_ids,
            max_new_tokens=args.warmup_new_tokens,
            cache=warmup_cache,
            eos_token_id=None,
        )
        torch.cuda.synchronize(device)
        print(
            f"budget={budget} warmup: {warmup.decode_tokens_per_second:.3f} tok/s",
            flush=True,
        )
        del warmup_cache, warmup

        for repetition in range(args.repeats):
            cache = model.make_resident_cache(
                batch_size=1,
                max_seq_len=input_ids.shape[1] + args.max_new_tokens,
            )
            runtime.stats.reset_forward_counters()
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
                if runtime_name == "sync"
                else runtime.stats.async_copy_seconds()
            )
            ids = result.generated_ids[0].tolist()
            first_mismatch = None
            if reference_ids is not None:
                for index, (actual, expected) in enumerate(zip(ids, reference_ids)):
                    if actual != expected:
                        first_mismatch = index
                        break
            row = {
                "gpu_layer_budget": budget,
                "runtime": runtime_name,
                "resident_layers": runtime.resident_count,
                "status": "success",
                "repetition": repetition,
                "input_tokens": int(input_ids.shape[1]),
                "output_tokens": len(ids),
                "prefill_ms": result.prefill_seconds * 1000,
                "ttft_ms": result.prefill_seconds * 1000,
                "decode_ms": result.decode_seconds * 1000,
                "decode_tokens_per_s": result.decode_tokens_per_second,
                "weight_h2d_bytes": runtime.stats.weight_h2d_bytes,
                "copy_seconds_sum": copy_seconds,
                "startup_h2d_bytes": runtime.stats.startup_h2d_bytes,
                "estimated_gpu_weight_bytes": estimate,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "first_mismatch": first_mismatch,
                "output_token_ids": ids,
            }
            measurements.append(row)
            print(
                f"budget={budget} repeat={repetition}: "
                f"TTFT={row['ttft_ms']:.1f} ms, "
                f"decode={row['decode_tokens_per_s']:.3f} tok/s, "
                f"peak={row['peak_allocated_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )
            del cache, result

        torch.cuda.synchronize(device)
        del model, runtime
        gc.collect()
        torch.cuda.empty_cache()

    successful = [row for row in measurements if row["status"] == "success"]
    summaries = []
    for budget in budgets:
        rows = [row for row in successful if row["gpu_layer_budget"] == budget]
        if rows:
            summaries.append(
                {
                    "gpu_layer_budget": budget,
                    "runtime": rows[0]["runtime"],
                    "resident_layers": rows[0]["resident_layers"],
                    "median_ttft_ms": statistics.median(row["ttft_ms"] for row in rows),
                    "median_decode_tokens_per_s": statistics.median(
                        row["decode_tokens_per_s"] for row in rows
                    ),
                    "max_peak_allocated_bytes": max(
                        row["peak_allocated_bytes"] for row in rows
                    ),
                    "weight_h2d_bytes_per_run": rows[0]["weight_h2d_bytes"],
                }
            )
    report = {
        "manifest": {
            "model": str(args.model.resolve()),
            "dtype": "float16",
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "warmup_new_tokens": args.warmup_new_tokens,
            "repeats": args.repeats,
            "budgets": budgets,
            "store_load_seconds": store_load_seconds,
            "all_cpu_weights_pinned": store.all_pinned,
            "cpu_weight_bytes": store.total_bytes,
        },
        "summaries": summaries,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
