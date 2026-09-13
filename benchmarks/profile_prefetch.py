#!/usr/bin/env python3
"""Export a short CUDA trace for Qwen3 weight-prefetch overlap inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer

from ondeviceinfer.offload_model import OffloadedQwen3ForCausalLM
from ondeviceinfer.runtime.offload import PrefetchOffloadRuntime, load_cpu_weight_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu-layer-budget", type=int, default=2)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    store = load_cpu_weight_store(args.model, dtype=torch.float16, pin_memory=True)
    runtime = PrefetchOffloadRuntime(
        store, device="cuda:0", gpu_layer_budget=args.gpu_layer_budget
    )
    model = OffloadedQwen3ForCausalLM(store, runtime, device="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids = tokenizer("你好，请介绍南京。", return_tensors="pt").input_ids.to("cuda:0")

    # Warm one complete forward before collecting the trace.
    cache = model.make_resident_cache(batch_size=1, max_seq_len=16)
    model(input_ids, cache=cache, logits_to_keep=1)
    torch.cuda.synchronize()
    cache.reset()
    runtime.stats.reset_forward_counters()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
    ) as profiler:
        model(input_ids, cache=cache, logits_to_keep=1)
        torch.cuda.synchronize()

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.trace))
    table = profiler.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=30
    )
    summary = {
        "model": str(args.model.resolve()),
        "gpu_layer_budget": args.gpu_layer_budget,
        "trace": str(args.trace),
        "weight_h2d_bytes": runtime.stats.weight_h2d_bytes,
        "async_copy_seconds_sum": runtime.stats.async_copy_seconds(),
        "profiler_table": table,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
