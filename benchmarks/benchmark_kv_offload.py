#!/usr/bin/env python3
"""Benchmark resident, synchronous, and prefetched KV-cache placement."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import socket
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ondeviceinfer.kv_offload import OffloadedKVCache
from ondeviceinfer.quantized_model import QuantizedQwen3ForCausalLM
from ondeviceinfer.runtime.quantized import (
    QuantizedResidentRuntime,
    estimate_quantized_gpu_weight_bytes,
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
        "--runs",
        default="resident,sync:20,prefetch:40",
        help="Comma-separated resident or mode:budget_mib entries",
    )
    parser.add_argument(
        "--storage-formats",
        default="fp16",
        help="Comma-separated offload formats: fp16,int8_hqq",
    )
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument(
        "--hqq-optimize",
        action="store_true",
        help="Enable iterative HQQ optimization (normally too costly online)",
    )
    parser.add_argument(
        "--hqq-runtime-backend",
        choices=("torch", "hqq_reference"),
        default="torch",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--cache-capacity",
        type=int,
        help="Preallocated token capacity; defaults to input + generated tokens",
    )
    parser.add_argument("--window-tokens", type=int, default=128)
    parser.add_argument("--seed-text", default=DEFAULT_TEXT)
    parser.add_argument("--warmup-input-tokens", type=int, default=16)
    parser.add_argument("--warmup-new-tokens", type=int, default=4)
    parser.add_argument(
        "--track-transfer-timing",
        action="store_true",
        help="Record per-copy CUDA timing; use for short profiles, not formal long runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def exact_length_ids(tokenizer: object, text: str, length: int) -> torch.Tensor:
    if length <= 0:
        raise ValueError("input-tokens must be positive")
    seed = tokenizer(text, add_special_tokens=True).input_ids
    if not seed:
        raise ValueError("seed text produced no tokens")
    repeated = (seed * math.ceil(length / len(seed)))[:length]
    return torch.tensor([repeated], dtype=torch.long)


def parse_runs(specification: str) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for raw in specification.split(","):
        item = raw.strip()
        if item == "resident":
            parsed.append({"mode": "resident", "budget_mib": None})
            continue
        try:
            mode, budget = item.split(":", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"Invalid run specification {item!r}") from error
        if mode not in {"sync", "prefetch"}:
            raise ValueError(f"Invalid KV mode {mode!r}")
        parsed.append({"mode": mode, "budget_mib": int(budget)})
    if not parsed:
        raise ValueError("At least one run is required")
    return parsed


def parse_storage_formats(specification: str) -> list[str]:
    formats = [item.strip() for item in specification.split(",") if item.strip()]
    if not formats:
        raise ValueError("At least one storage format is required")
    invalid = [item for item in formats if item not in {"fp16", "int8_hqq"}]
    if invalid:
        raise ValueError(f"Invalid KV storage formats: {invalid}")
    if len(set(formats)) != len(formats):
        raise ValueError("KV storage formats must not be duplicated")
    return formats


def expand_runs(
    runs: list[dict[str, object]], storage_formats: list[str], repeats: int
) -> list[dict[str, object]]:
    expanded = []
    for run in runs:
        formats = ["resident_fp16"] if run["mode"] == "resident" else storage_formats
        for storage_format in formats:
            for repeat_index in range(repeats):
                expanded.append(
                    {
                        **run,
                        "storage_format": storage_format,
                        "repeat_index": repeat_index,
                    }
                )
    return expanded


def percentile(values: list[float], probability: float) -> float:
    index = max(0, math.ceil(probability * len(values)) - 1)
    return sorted(values)[index]


def tensor_ids_sha256(input_ids: torch.Tensor) -> str:
    values = ",".join(str(value) for value in input_ids[0].tolist())
    return hashlib.sha256(values.encode("ascii")).hexdigest()


def cache_counters(cache: object) -> tuple[int, int]:
    stats = getattr(cache, "stats", None)
    return (
        int(getattr(stats, "h2d_bytes", 0)),
        int(getattr(stats, "d2h_bytes", 0)),
    )


@torch.inference_mode()
def generate_profiled(
    model: QuantizedQwen3ForCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    window_tokens: int,
    cache: object,
) -> dict[str, object]:
    device = input_ids.device
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = model(input_ids, cache=cache, logits_to_keep=1)
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated_ids = torch.empty(
        (input_ids.shape[0], max_new_tokens),
        dtype=next_token.dtype,
        device=device,
    )
    generated_ids[:, :1].copy_(next_token)
    next_token = generated_ids[:, :1]
    torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - start
    windows: list[dict[str, object]] = []
    decode_start = time.perf_counter()
    window_start = decode_start
    measured_decode_seconds = 0.0
    timing_drain_overhead_seconds = 0.0
    inter_token_latency_ms: list[float] = []
    pending_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    previous_h2d, previous_d2h = cache_counters(cache)
    previous_count = 1

    for generated_count in range(2, max_new_tokens + 1):
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record(torch.cuda.current_stream(device))
        output = model(next_token, cache=cache, logits_to_keep=1)
        selected = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated_ids[:, generated_count - 1 : generated_count].copy_(selected)
        step_end.record(torch.cuda.current_stream(device))
        pending_step_events.append((step_start, step_end))
        next_token = generated_ids[:, generated_count - 1 : generated_count]
        decode_forward_count = generated_count - 1
        at_window = decode_forward_count % window_tokens == 0
        at_end = generated_count == max_new_tokens
        if at_window or at_end:
            torch.cuda.synchronize(device)
            measured_now = time.perf_counter()
            h2d, d2h = cache_counters(cache)
            inter_token_latency_ms.extend(
                start_event.elapsed_time(end_event)
                for start_event, end_event in pending_step_events
            )
            pending_step_events.clear()
            tokens_in_window = generated_count - previous_count
            window_seconds = measured_now - window_start
            measured_decode_seconds += window_seconds
            windows.append(
                {
                    "generated_token_start": previous_count + 1,
                    "generated_token_end": generated_count,
                    "decode_forwards": tokens_in_window,
                    "seconds": window_seconds,
                    "tokens_per_second": tokens_in_window / window_seconds,
                    "kv_h2d_bytes": h2d - previous_h2d,
                    "kv_d2h_bytes": d2h - previous_d2h,
                    "cache_length": cache.get_seq_length(0),
                    "memory_allocated_bytes": torch.cuda.memory_allocated(device),
                    "memory_reserved_bytes": torch.cuda.memory_reserved(device),
                }
            )
            drain_start = time.perf_counter()
            if hasattr(cache, "drain_timing_events"):
                cache.drain_timing_events()
            timing_drain_overhead_seconds += time.perf_counter() - drain_start
            window_start = time.perf_counter()
            previous_count = generated_count
            previous_h2d, previous_d2h = h2d, d2h

    torch.cuda.synchronize(device)
    decode_seconds = measured_decode_seconds
    if hasattr(cache, "synchronize"):
        cache.synchronize()
    h2d, d2h = cache_counters(cache)
    cache_stats = getattr(cache, "stats", None)
    h2d_seconds = (
        cache_stats.h2d_seconds()
        if cache_stats is not None and hasattr(cache_stats, "h2d_seconds")
        else 0.0
    )
    d2h_seconds = (
        cache_stats.d2h_seconds()
        if cache_stats is not None and hasattr(cache_stats, "d2h_seconds")
        else 0.0
    )
    quantize_seconds = (
        cache_stats.quantize_seconds()
        if cache_stats is not None and hasattr(cache_stats, "quantize_seconds")
        else 0.0
    )
    dequantize_seconds = (
        cache_stats.dequantize_seconds()
        if cache_stats is not None and hasattr(cache_stats, "dequantize_seconds")
        else 0.0
    )
    expected_final_length = input_ids.shape[1] + max_new_tokens - 1
    layer_lengths = list(getattr(cache, "lengths", [cache.get_seq_length(0)]))
    if any(length != expected_final_length for length in layer_lengths):
        raise RuntimeError(
            f"KV layer lengths differ from expected {expected_final_length}: "
            f"{layer_lengths}"
        )
    return {
        "prefill_ms": prefill_seconds * 1000,
        "ttft_ms": prefill_seconds * 1000,
        "decode_ms": decode_seconds * 1000,
        "decode_tokens_per_s": (
            (max_new_tokens - 1) / decode_seconds
            if max_new_tokens > 1 and decode_seconds > 0
            else 0.0
        ),
        "timing_drain_overhead_seconds": timing_drain_overhead_seconds,
        "output_tokens": max_new_tokens,
        "final_cache_length": cache.get_seq_length(0),
        "all_layer_cache_lengths": layer_lengths,
        "kv_h2d_bytes": h2d,
        "kv_d2h_bytes": d2h,
        "kv_h2d_seconds_sum": h2d_seconds,
        "kv_d2h_seconds_sum": d2h_seconds,
        "kv_quantize_seconds_sum": quantize_seconds,
        "kv_dequantize_seconds_sum": dequantize_seconds,
        "kv_h2d_data_bytes": int(getattr(cache_stats, "h2d_data_bytes", 0)),
        "kv_h2d_metadata_bytes": int(
            getattr(cache_stats, "h2d_metadata_bytes", 0)
        ),
        "kv_d2h_data_bytes": int(getattr(cache_stats, "d2h_data_bytes", 0)),
        "kv_d2h_metadata_bytes": int(
            getattr(cache_stats, "d2h_metadata_bytes", 0)
        ),
        "kv_quantize_calls": int(getattr(cache_stats, "quantize_calls", 0)),
        "kv_dequantize_calls": int(getattr(cache_stats, "dequantize_calls", 0)),
        "itl_ms_p50": (
            statistics.median(inter_token_latency_ms)
            if inter_token_latency_ms
            else 0.0
        ),
        "itl_ms_p95": (
            percentile(inter_token_latency_ms, 0.95)
            if inter_token_latency_ms
            else 0.0
        ),
        "itl_ms_min": min(inter_token_latency_ms) if inter_token_latency_ms else 0.0,
        "itl_ms_max": max(inter_token_latency_ms) if inter_token_latency_ms else 0.0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "output_token_ids": generated_ids[0].tolist(),
        "windows": windows,
    }


def make_cache(
    run: dict[str, object],
    model: QuantizedQwen3ForCausalLM,
    *,
    max_seq_len: int,
    track_transfer_timing: bool,
    kv_group_size: int,
    hqq_optimize: bool,
    hqq_runtime_backend: str,
) -> object:
    mode = run["mode"]
    if mode == "resident":
        return model.make_resident_cache(batch_size=1, max_seq_len=max_seq_len)
    budget_bytes = int(run["budget_mib"]) * 2**20
    return OffloadedKVCache(
        num_layers=model.config.num_hidden_layers,
        batch_size=1,
        num_key_value_heads=model.config.num_key_value_heads,
        max_seq_len=max_seq_len,
        head_dim=model.config.head_dim,
        device=model.device,
        dtype=model.dtype,
        mode=mode,
        gpu_budget_bytes=budget_bytes,
        pin_memory=True,
        track_transfer_timing=track_transfer_timing,
        storage_format=str(run["storage_format"]),
        kv_group_size=kv_group_size,
        hqq_optimize=hqq_optimize,
        hqq_runtime_backend=hqq_runtime_backend,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.max_new_tokens <= 0 or args.window_tokens <= 0 or args.repeats <= 0:
        raise ValueError("token counts must be positive")
    base_runs = parse_runs(args.runs)
    storage_formats = parse_storage_formats(args.storage_formats)
    runs = expand_runs(base_runs, storage_formats, args.repeats)
    device = torch.device("cuda:0")
    load_start = time.perf_counter()
    store = load_quantized_cpu_weight_store(args.model, pin_memory=True, progress=True)
    runtime = QuantizedResidentRuntime(store, device=device)
    model = QuantizedQwen3ForCausalLM(store, runtime, device=device)
    load_seconds = time.perf_counter() - load_start
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids_cpu = exact_length_ids(tokenizer, args.seed_text, args.input_tokens)
    input_ids = input_ids_cpu.to(device)
    required_capacity = args.input_tokens + args.max_new_tokens
    max_seq_len = args.cache_capacity or required_capacity
    if max_seq_len < required_capacity:
        raise ValueError(
            f"cache-capacity {max_seq_len} is below required {required_capacity}"
        )

    # One process-level CUDA and kernel warmup; every measured mode then sees
    # the same already-initialized all-resident INT8 weight runtime.
    warmup_ids = exact_length_ids(
        tokenizer, args.seed_text, args.warmup_input_tokens
    ).to(device)
    warmup_cache = model.make_resident_cache(
        batch_size=1,
        max_seq_len=args.warmup_input_tokens + args.warmup_new_tokens,
    )
    warmup = generate_profiled(
        model,
        warmup_ids,
        max_new_tokens=args.warmup_new_tokens,
        window_tokens=max(1, args.warmup_new_tokens),
        cache=warmup_cache,
    )
    del warmup_cache, warmup_ids
    torch.cuda.empty_cache()

    report: dict[str, object] = {
        "manifest": {
            "schema_version": 2,
            "model": str(args.model.resolve()),
            "source_revision": store.manifest.get("source_revision"),
            "weight_runtime": "int8_all_decoder_layers_resident",
            "compute_dtype": "float16",
            "input_tokens": args.input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_seq_len": max_seq_len,
            "ignore_eos": True,
            "expected_final_cache_length": args.input_tokens
            + args.max_new_tokens
            - 1,
            "window_tokens": args.window_tokens,
            "repeats": args.repeats,
            "storage_formats": storage_formats,
            "kv_group_size": args.kv_group_size,
            "hqq_optimize": args.hqq_optimize,
            "hqq_runtime_backend": args.hqq_runtime_backend,
            "track_transfer_timing": args.track_transfer_timing,
            "input_ids_sha256": tensor_ids_sha256(input_ids_cpu),
            "seed_text": args.seed_text,
            "runs": runs,
            "store_load_and_gpu_placement_seconds": load_seconds,
            "estimated_gpu_weight_bytes": estimate_quantized_gpu_weight_bytes(
                store, store.config.num_hidden_layers
            ),
            "process_warmup": {
                "input_tokens": args.warmup_input_tokens,
                "output_tokens": args.warmup_new_tokens,
                "prefill_ms": warmup["prefill_ms"],
                "decode_tokens_per_s": warmup["decode_tokens_per_s"],
            },
            "environment": {
                "hostname": socket.gethostname(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
        },
        "measurements": [],
    }
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    reference_ids: list[int] | None = None

    for run_idx, run in enumerate(runs):
        gc.collect()
        torch.cuda.empty_cache()
        cache = make_cache(
            run,
            model,
            max_seq_len=max_seq_len,
            track_transfer_timing=args.track_transfer_timing,
            kv_group_size=args.kv_group_size,
            hqq_optimize=args.hqq_optimize,
            hqq_runtime_backend=args.hqq_runtime_backend,
        )
        if hasattr(cache, "reset_stats"):
            cache.reset_stats()
        torch.cuda.reset_peak_memory_stats(device)
        result = generate_profiled(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            window_tokens=args.window_tokens,
            cache=cache,
        )
        ids = result["output_token_ids"]
        if reference_ids is None:
            reference_ids = ids
            mismatch = None
        else:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(ids, reference_ids)
                    )
                    if actual != expected
                ),
                None,
            )
            if mismatch is None and len(ids) != len(reference_ids):
                mismatch = min(len(ids), len(reference_ids))
        result.update(
            {
                "run_index": run_idx,
                "requested_mode": run["mode"],
                "requested_budget_mib": run["budget_mib"],
                "storage_format": run["storage_format"],
                "repeat_index": run["repeat_index"],
                "effective_mode": getattr(cache, "effective_mode", "resident"),
                "fallback_reason": getattr(cache, "fallback_reason", None),
                "slot_count": getattr(cache, "slot_count", None),
                "host_is_pinned": getattr(cache, "host_is_pinned", None),
                "managed_kv_cpu_bytes": getattr(cache, "managed_cpu_bytes", 0),
                "managed_kv_cpu_data_bytes": getattr(
                    cache, "managed_cpu_data_bytes", 0
                ),
                "managed_kv_cpu_metadata_bytes": getattr(
                    cache, "managed_cpu_metadata_bytes", 0
                ),
                "managed_kv_gpu_bytes": getattr(
                    cache, "managed_gpu_bytes", cache.storage_bytes
                ),
                "allocated_kv_storage_bytes": cache.storage_bytes,
                "attention_layer_capacity_bytes": getattr(
                    cache, "attention_layer_capacity_bytes", None
                ),
                "encoded_layer_capacity_bytes": getattr(
                    cache, "encoded_layer_capacity_bytes", None
                ),
                "slot_capacity_bytes": getattr(cache, "slot_capacity_bytes", None),
                "host_compression_ratio": getattr(
                    cache, "host_compression_ratio", None
                ),
                "hqq_version": getattr(cache, "hqq_version", None),
                "hqq_runtime_backend": getattr(
                    cache, "hqq_runtime_backend", None
                ),
                "first_mismatch_vs_first_run": mismatch,
                "token_agreement_vs_first_run": (
                    sum(actual == expected for actual, expected in zip(ids, reference_ids))
                    / max(len(ids), len(reference_ids))
                ),
            }
        )
        report["measurements"].append(result)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{run['mode']}:{run['budget_mib']} {run['storage_format']} "
            f"repeat={run['repeat_index']} complete: "
            f"TTFT={result['ttft_ms']:.1f} ms, "
            f"decode={result['decode_tokens_per_s']:.3f} tok/s, "
            f"mismatch={mismatch}",
            flush=True,
        )
        del cache

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in report["measurements"]:
        label = f"{row['requested_mode']}::{row['storage_format']}"
        grouped.setdefault(label, []).append(row)
    group_summaries = {}
    for label, rows in grouped.items():
        decode_values = [float(row["decode_tokens_per_s"]) for row in rows]
        ttft_values = [float(row["ttft_ms"]) for row in rows]
        group_summaries[label] = {
            "runs": len(rows),
            "median_decode_tokens_per_s": statistics.median(decode_values),
            "min_decode_tokens_per_s": min(decode_values),
            "max_decode_tokens_per_s": max(decode_values),
            "median_ttft_ms": statistics.median(ttft_values),
            "min_ttft_ms": min(ttft_values),
            "max_ttft_ms": max(ttft_values),
            "median_token_agreement_vs_resident": statistics.median(
                float(row["token_agreement_vs_first_run"]) for row in rows
            ),
            "median_itl_ms_p50": statistics.median(
                float(row["itl_ms_p50"]) for row in rows
            ),
            "median_itl_ms_p95": statistics.median(
                float(row["itl_ms_p95"]) for row in rows
            ),
        }
    report["summary"] = {
        "all_token_sequences_equal": all(
            row["first_mismatch_vs_first_run"] is None
            for row in report["measurements"]
        ),
        "completed_run_count": len(report["measurements"]),
        "groups": group_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
