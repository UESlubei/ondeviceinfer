#!/usr/bin/env python3
"""Bounded Qwen3-14B workloads for NVIDIA Nsight Systems.

Model loading, CUDA initialization, warmup, and target-cache prefill happen
outside ``cudaProfilerStart``/``cudaProfilerStop``.  Nsight therefore records
only the decode forwards selected by ``--capture-steps``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ondeviceinfer.kv_offload import OffloadedKVCache
from ondeviceinfer.profiling import nvtx_enabled, nvtx_range
from ondeviceinfer.quantized_model import QuantizedQwen3ForCausalLM
from ondeviceinfer.runtime.quantized import (
    QuantizedPrefetchOffloadRuntime,
    QuantizedResidentRuntime,
    load_quantized_cpu_weight_store,
)


DEFAULT_TEXT = (
    "南京是一座历史文化名城，也是现代化创新城市。请结合地理、历史、文化、"
    "科技和日常生活介绍这座城市，并保持叙述连贯。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=("weight-prefetch", "resident-decode", "kv-prefetch"),
        required=True,
    )
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--capture-steps", type=int, default=1)
    parser.add_argument("--gpu-layer-budget", type=int, default=2)
    parser.add_argument("--kv-budget-mib", type=int, default=40)
    parser.add_argument(
        "--kv-storage-format",
        choices=("fp16", "int8_hqq"),
        default="fp16",
    )
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--hqq-optimize", action="store_true")
    parser.add_argument(
        "--hqq-runtime-backend",
        choices=("torch", "hqq_reference"),
        default="torch",
    )
    parser.add_argument("--warmup-input-tokens", type=int, default=8)
    parser.add_argument("--seed-text", default=DEFAULT_TEXT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def exact_length_ids(tokenizer: object, text: str, length: int) -> torch.Tensor:
    if length <= 0:
        raise ValueError("input token count must be positive")
    seed = tokenizer(text, add_special_tokens=True).input_ids
    if not seed:
        raise ValueError("seed text produced no tokens")
    repeated = (seed * math.ceil(length / len(seed)))[:length]
    return torch.tensor([repeated], dtype=torch.long)


def make_runtime(
    args: argparse.Namespace, store: object, device: torch.device
) -> object:
    if args.scenario == "weight-prefetch":
        return QuantizedPrefetchOffloadRuntime(
            store,
            device=device,
            gpu_layer_budget=args.gpu_layer_budget,
        )
    return QuantizedResidentRuntime(store, device=device)


def make_target_cache(
    args: argparse.Namespace, model: QuantizedQwen3ForCausalLM
) -> object:
    capacity = args.input_tokens + args.capture_steps + 1
    if args.scenario != "kv-prefetch":
        return model.make_resident_cache(batch_size=1, max_seq_len=capacity)
    return OffloadedKVCache(
        num_layers=model.config.num_hidden_layers,
        batch_size=1,
        num_key_value_heads=model.config.num_key_value_heads,
        max_seq_len=capacity,
        head_dim=model.config.head_dim,
        device=model.device,
        dtype=model.dtype,
        mode="prefetch",
        gpu_budget_bytes=args.kv_budget_mib * 2**20,
        pin_memory=True,
        track_transfer_timing=True,
        storage_format=args.kv_storage_format,
        kv_group_size=args.kv_group_size,
        hqq_optimize=args.hqq_optimize,
        hqq_runtime_backend=args.hqq_runtime_backend,
    )


def reset_profile_counters(model: QuantizedQwen3ForCausalLM, cache: object) -> None:
    model.runtime.stats.reset_forward_counters()  # type: ignore[attr-defined]
    model.compute_stats.reset()
    if hasattr(cache, "reset_stats"):
        cache.reset_stats()  # type: ignore[attr-defined]


def cache_metrics(cache: object) -> dict[str, object]:
    stats = getattr(cache, "stats", None)
    result: dict[str, object] = {
        "type": type(cache).__name__,
        "sequence_length": int(cache.get_seq_length(0)),
        "storage_bytes": int(getattr(cache, "storage_bytes", 0)),
        "managed_gpu_bytes": int(getattr(cache, "managed_gpu_bytes", 0)),
        "managed_cpu_bytes": int(getattr(cache, "managed_cpu_bytes", 0)),
        "managed_cpu_data_bytes": int(
            getattr(cache, "managed_cpu_data_bytes", 0)
        ),
        "managed_cpu_metadata_bytes": int(
            getattr(cache, "managed_cpu_metadata_bytes", 0)
        ),
        "storage_format": str(getattr(cache, "storage_format", "resident_fp16")),
    }
    if stats is not None:
        result.update(
            {
                "h2d_bytes": int(getattr(stats, "h2d_bytes", 0)),
                "d2h_bytes": int(getattr(stats, "d2h_bytes", 0)),
                "h2d_copies": int(getattr(stats, "cpu_to_gpu_copies", 0)),
                "d2h_copies": int(getattr(stats, "gpu_to_cpu_copies", 0)),
                "h2d_data_bytes": int(getattr(stats, "h2d_data_bytes", 0)),
                "h2d_metadata_bytes": int(
                    getattr(stats, "h2d_metadata_bytes", 0)
                ),
                "d2h_data_bytes": int(getattr(stats, "d2h_data_bytes", 0)),
                "d2h_metadata_bytes": int(
                    getattr(stats, "d2h_metadata_bytes", 0)
                ),
                "quantize_calls": int(getattr(stats, "quantize_calls", 0)),
                "dequantize_calls": int(getattr(stats, "dequantize_calls", 0)),
            }
        )
        if hasattr(stats, "h2d_seconds"):
            result["h2d_seconds_sum"] = float(stats.h2d_seconds())
        if hasattr(stats, "d2h_seconds"):
            result["d2h_seconds_sum"] = float(stats.d2h_seconds())
        if hasattr(stats, "quantize_seconds"):
            result["quantize_seconds_sum"] = float(stats.quantize_seconds())
        if hasattr(stats, "dequantize_seconds"):
            result["dequantize_seconds_sum"] = float(stats.dequantize_seconds())
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.capture_steps <= 0 or args.warmup_input_tokens <= 0:
        raise ValueError("token counts must be positive")
    if not nvtx_enabled():
        print(
            "warning: ONDEVICEINFER_NVTX is disabled; CUDA capture still works "
            "but semantic ranges will be absent",
            flush=True,
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    load_start = time.perf_counter()
    store = load_quantized_cpu_weight_store(
        args.model,
        pin_memory=True,
        progress=True,
    )
    runtime = make_runtime(args, store, device)
    model = QuantizedQwen3ForCausalLM(store, runtime, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    load_seconds = time.perf_counter() - load_start

    # Process-level warmup: CUDA libraries and every decoder operator execute
    # before capture, with a small cache that is discarded afterwards.
    warmup_ids = exact_length_ids(
        tokenizer, args.seed_text, args.warmup_input_tokens
    ).to(device)
    warmup_cache = model.make_resident_cache(
        batch_size=1,
        max_seq_len=args.warmup_input_tokens,
    )
    model(warmup_ids, cache=warmup_cache, logits_to_keep=1)
    torch.cuda.synchronize(device)
    del warmup_cache, warmup_ids

    # Prime the target cache outside capture.  The captured range is exactly a
    # steady-state decode step, including the selected placement transfers.
    input_ids = exact_length_ids(tokenizer, args.seed_text, args.input_tokens).to(device)
    cache = make_target_cache(args, model)
    priming_start = time.perf_counter()
    output = model(input_ids, cache=cache, logits_to_keep=1)
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    priming_seconds = time.perf_counter() - priming_start

    reset_profile_counters(model, cache)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    profiler = torch.cuda.cudart()
    start_status = int(profiler.cudaProfilerStart())
    if start_status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {start_status}")
    capture_start = time.perf_counter()
    captured_ids: list[int] = []
    with nvtx_range(
        f"nsight_capture::{args.scenario}::{args.kv_storage_format}"
    ):
        for step in range(args.capture_steps):
            with nvtx_range(f"decode_step::{step}"):
                output = model(next_token, cache=cache, logits_to_keep=1)
                next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                captured_ids.append(int(next_token.item()))
    torch.cuda.synchronize(device)
    capture_seconds = time.perf_counter() - capture_start
    stop_status = int(profiler.cudaProfilerStop())
    if stop_status != 0:
        raise RuntimeError(f"cudaProfilerStop failed with status {stop_status}")

    if hasattr(cache, "synchronize"):
        cache.synchronize()  # type: ignore[attr-defined]
    runtime_copy_seconds = (
        runtime.stats.async_copy_seconds()
        if hasattr(runtime.stats, "async_copy_seconds")
        else 0.0
    )
    properties = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": 1,
        "scenario": args.scenario,
        "capture_scope": "steady_state_decode_only",
        "model": str(args.model.resolve()),
        "source_revision": store.manifest.get("source_revision"),
        "input_tokens": args.input_tokens,
        "capture_steps": args.capture_steps,
        "kv_storage_format": args.kv_storage_format,
        "kv_group_size": args.kv_group_size,
        "hqq_optimize": args.hqq_optimize,
        "hqq_runtime_backend": args.hqq_runtime_backend,
        "captured_output_token_ids": captured_ids,
        "nvtx_enabled": nvtx_enabled(),
        "cuda_profiler_start_status": start_status,
        "cuda_profiler_stop_status": stop_status,
        "timing": {
            "load_seconds": load_seconds,
            "target_priming_seconds": priming_seconds,
            "capture_wall_seconds": capture_seconds,
            "captured_tokens_per_second": args.capture_steps / capture_seconds,
            "weight_copy_cuda_seconds_sum": runtime_copy_seconds,
        },
        "weights": {
            "runtime": type(runtime).__name__,
            "gpu_layer_budget": int(runtime.gpu_layer_budget),
            "resident_layers": int(runtime.resident_count),
            "captured_h2d_bytes": int(runtime.stats.weight_h2d_bytes),
            "decoder_bytes": int(store.decoder_bytes),
            "all_cpu_weights_pinned": bool(store.all_pinned),
        },
        "kv_cache": cache_metrics(cache),
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_total_bytes": int(properties.total_memory),
            "ondeviceinfer_nvtx": os.environ.get("ONDEVICEINFER_NVTX", "0"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
