#!/usr/bin/env python3
"""Run sync/prefetch weight offloading and optionally check expected token IDs."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--runtime", choices=["sync", "prefetch"], required=True)
    parser.add_argument("--gpu-layer-budget", type=int, required=True)
    parser.add_argument("--prompt", default="你好，请用一句话介绍南京。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--expected-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    load_start = time.perf_counter()
    store = load_cpu_weight_store(args.model, dtype=torch.float16, pin_memory=True)
    store_load_seconds = time.perf_counter() - load_start
    estimate = estimate_gpu_weight_bytes(store, args.gpu_layer_budget)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    if estimate > int(total_memory * 0.92):
        raise RuntimeError(
            f"Estimated GPU weight bytes {estimate} exceed the 92% safety limit"
        )

    runtime_class = (
        SyncOffloadRuntime if args.runtime == "sync" else PrefetchOffloadRuntime
    )
    runtime = runtime_class(
        store, device=device, gpu_layer_budget=args.gpu_layer_budget
    )
    model = OffloadedQwen3ForCausalLM(store, runtime, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
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
    async_copy_seconds = runtime.stats.async_copy_seconds()

    expected_ids = None
    first_mismatch = None
    if args.expected_json:
        expected = json.loads(args.expected_json.read_text(encoding="utf-8"))
        if "reference_generated_ids" in expected:
            expected_ids = expected["reference_generated_ids"]
        elif "measurements" in expected:
            expected_ids = expected["measurements"][-1]["output_token_ids"]
        else:
            expected_ids = expected["generated_ids"]
        actual_ids = result.generated_ids[0].tolist()
        for index, (actual, wanted) in enumerate(zip(actual_ids, expected_ids)):
            if actual != wanted:
                first_mismatch = index
                break
        if first_mismatch is None and len(actual_ids) != len(expected_ids):
            first_mismatch = min(len(actual_ids), len(expected_ids))

    report = {
        "model": str(args.model.resolve()),
        "runtime": args.runtime,
        "gpu_layer_budget": args.gpu_layer_budget,
        "resident_layers": runtime.resident_count,
        "dtype": "float16",
        "prompt": args.prompt,
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": int(result.generated_ids.shape[1]),
        "generated_ids": result.generated_ids[0].tolist(),
        "expected_ids": expected_ids,
        "first_mismatch": first_mismatch,
        "all_cpu_weights_pinned": store.all_pinned,
        "cpu_weight_bytes": store.total_bytes,
        "store_load_seconds": store_load_seconds,
        "estimated_gpu_weight_bytes": estimate,
        "prefill_ms": result.prefill_seconds * 1000,
        "decode_ms": result.decode_seconds * 1000,
        "decode_tokens_per_s": result.decode_tokens_per_second,
        "weight_h2d_bytes": runtime.stats.weight_h2d_bytes,
        "startup_h2d_bytes": runtime.stats.startup_h2d_bytes,
        "synchronous_copy_seconds": runtime.stats.synchronous_copy_seconds,
        "async_copy_seconds_sum": async_copy_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "kv_cache_bytes": cache.storage_bytes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if first_mismatch is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
