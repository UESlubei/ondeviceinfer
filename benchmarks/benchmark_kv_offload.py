#!/usr/bin/env python3
"""Benchmark resident, synchronous, and prefetched KV-cache placement."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import resource
import socket
import statistics
import traceback
import time
from pathlib import Path
from typing import Callable

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

HETEROGENEOUS_TEXTS = (
    DEFAULT_TEXT,
    "请编写一个 Python 函数，使用动态规划计算最长递增子序列，并解释复杂度。",
    "甲乙两车从两地同时相向而行，请逐步推导相遇时间并检查每一步假设。",
    "Summarize how memory bandwidth affects autoregressive language model inference.",
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
    parser.add_argument(
        "--execution-order",
        choices=("rotating", "grouped"),
        default="rotating",
        help="Rotate the first mode per repeat to reduce temperature/order bias",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--batch-mode",
        choices=("identical", "heterogeneous"),
        default="identical",
        help="Use repeated identical prompts or a built-in equal-length prompt mix",
    )
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
    parser.add_argument(
        "--status-output",
        type=Path,
        help="Optional crash-safe status sidecar used by capacity probes",
    )
    return parser.parse_args()


def update_status(path: Path | None, **values: object) -> None:
    if path is None:
        return
    current: dict[str, object] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def exact_length_ids(tokenizer: object, text: str, length: int) -> torch.Tensor:
    if length <= 0:
        raise ValueError("input-tokens must be positive")
    seed = tokenizer(text, add_special_tokens=True).input_ids
    if not seed:
        raise ValueError("seed text produced no tokens")
    repeated = (seed * math.ceil(length / len(seed)))[:length]
    return torch.tensor([repeated], dtype=torch.long)


def exact_length_batch_ids(
    tokenizer: object,
    *,
    seed_text: str,
    length: int,
    batch_size: int,
    batch_mode: str,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    texts = (
        [seed_text] * batch_size
        if batch_mode == "identical"
        else [HETEROGENEOUS_TEXTS[index % len(HETEROGENEOUS_TEXTS)] for index in range(batch_size)]
    )
    rows = [exact_length_ids(tokenizer, text, length)[0] for text in texts]
    return torch.stack(rows)


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
        budget_value: int | str
        if budget == "auto":
            budget_value = "auto"
        else:
            budget_value = int(budget)
        parsed.append({"mode": mode, "budget_mib": budget_value})
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
    runs: list[dict[str, object]],
    storage_formats: list[str],
    repeats: int,
    execution_order: str = "rotating",
) -> list[dict[str, object]]:
    configurations = []
    for run in runs:
        formats = ["resident_fp16"] if run["mode"] == "resident" else storage_formats
        for storage_format in formats:
            configurations.append({**run, "storage_format": storage_format})
    expanded = []
    if execution_order == "grouped":
        for configuration in configurations:
            for repeat_index in range(repeats):
                expanded.append({**configuration, "repeat_index": repeat_index})
        return expanded
    if execution_order != "rotating":
        raise ValueError("execution_order must be 'rotating' or 'grouped'")
    for repeat_index in range(repeats):
        offset = repeat_index % len(configurations)
        rotated = configurations[offset:] + configurations[:offset]
        for configuration in rotated:
            expanded.append({**configuration, "repeat_index": repeat_index})
    return expanded


def percentile(values: list[float], probability: float) -> float:
    index = max(0, math.ceil(probability * len(values)) - 1)
    return sorted(values)[index]


def tensor_ids_sha256(input_ids: torch.Tensor) -> str:
    values = "\n".join(
        ",".join(str(value) for value in row.tolist()) for row in input_ids
    )
    return hashlib.sha256(values.encode("ascii")).hexdigest()


def row_ids_sha256(input_ids: torch.Tensor) -> list[str]:
    return [
        hashlib.sha256(
            ",".join(str(value) for value in row.tolist()).encode("ascii")
        ).hexdigest()
        for row in input_ids
    ]


def compare_token_batches(
    actual: list[list[int]], expected: list[list[int]]
) -> tuple[list[int | None], float]:
    if len(actual) != len(expected):
        raise ValueError("batch size changed between benchmark runs")
    mismatches: list[int | None] = []
    equal_tokens = 0
    compared_tokens = 0
    for actual_row, expected_row in zip(actual, expected):
        mismatch = next(
            (
                index
                for index, (actual_token, expected_token) in enumerate(
                    zip(actual_row, expected_row)
                )
                if actual_token != expected_token
            ),
            None,
        )
        if mismatch is None and len(actual_row) != len(expected_row):
            mismatch = min(len(actual_row), len(expected_row))
        mismatches.append(mismatch)
        equal_tokens += sum(
            actual_token == expected_token
            for actual_token, expected_token in zip(actual_row, expected_row)
        )
        compared_tokens += max(len(actual_row), len(expected_row))
    return mismatches, equal_tokens / compared_tokens if compared_tokens else 1.0


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
    stage_callback: Callable[[str], None] | None = None,
    separate_prefill_drain: bool = False,
) -> dict[str, object]:
    device = input_ids.device
    if stage_callback is not None:
        stage_callback("prefill")
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
    if separate_prefill_drain:
        torch.cuda.current_stream(device).synchronize()
        ttft_seconds = time.perf_counter() - start
        drain_start = time.perf_counter()
        torch.cuda.synchronize(device)
        if hasattr(cache, "drain_timing_events"):
            cache.drain_timing_events()
        prefill_seal_drain_seconds = time.perf_counter() - drain_start
    else:
        torch.cuda.synchronize(device)
        ttft_seconds = time.perf_counter() - start
        prefill_seal_drain_seconds = 0.0
    prefill_seconds = time.perf_counter() - start
    if stage_callback is not None:
        stage_callback("decode")
    windows: list[dict[str, object]] = []
    decode_start = time.perf_counter()
    window_start = decode_start
    measured_decode_seconds = 0.0
    timing_drain_overhead_seconds = 0.0
    inter_token_latency_ms: list[float] = []
    seal_step_latency_ms: list[float] = []
    non_seal_step_latency_ms: list[float] = []
    pending_step_events: list[
        tuple[torch.cuda.Event, torch.cuda.Event, bool]
    ] = []
    previous_h2d, previous_d2h = cache_counters(cache)
    previous_count = 1

    for generated_count in range(2, max_new_tokens + 1):
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        page_seals_before = int(
            getattr(getattr(cache, "stats", None), "page_seals", 0)
        )
        step_start.record(torch.cuda.current_stream(device))
        output = model(next_token, cache=cache, logits_to_keep=1)
        selected = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated_ids[:, generated_count - 1 : generated_count].copy_(selected)
        step_end.record(torch.cuda.current_stream(device))
        page_seals_after = int(
            getattr(getattr(cache, "stats", None), "page_seals", 0)
        )
        pending_step_events.append(
            (step_start, step_end, page_seals_after > page_seals_before)
        )
        next_token = generated_ids[:, generated_count - 1 : generated_count]
        decode_forward_count = generated_count - 1
        at_window = decode_forward_count % window_tokens == 0
        at_end = generated_count == max_new_tokens
        if at_window or at_end:
            torch.cuda.synchronize(device)
            measured_now = time.perf_counter()
            h2d, d2h = cache_counters(cache)
            for start_event, end_event, is_seal_step in pending_step_events:
                latency = start_event.elapsed_time(end_event)
                inter_token_latency_ms.append(latency)
                target = (
                    seal_step_latency_ms
                    if is_seal_step
                    else non_seal_step_latency_ms
                )
                target.append(latency)
            pending_step_events.clear()
            tokens_in_window = generated_count - previous_count
            window_seconds = measured_now - window_start
            measured_decode_seconds += window_seconds
            windows.append(
                {
                    "generated_token_start": previous_count + 1,
                    "generated_token_end": generated_count,
                    "decode_forwards": tokens_in_window,
                    "batch_size": int(input_ids.shape[0]),
                    "aggregate_decode_tokens": int(input_ids.shape[0])
                    * tokens_in_window,
                    "seconds": window_seconds,
                    "decode_steps_per_second": tokens_in_window / window_seconds,
                    "tokens_per_second": (
                        int(input_ids.shape[0]) * tokens_in_window / window_seconds
                    ),
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
    decode_steps = max(0, max_new_tokens - 1)
    batch_size = int(input_ids.shape[0])
    output_ids_cpu = generated_ids.cpu()
    return {
        "batch_size": batch_size,
        "prefill_ms": prefill_seconds * 1000,
        "ttft_ms": ttft_seconds * 1000,
        "prefill_seal_drain_ms": prefill_seal_drain_seconds * 1000,
        "decode_ms": decode_seconds * 1000,
        "decode_steps": decode_steps,
        "decoded_tokens": batch_size * decode_steps,
        "decode_steps_per_s": (
            decode_steps / decode_seconds if decode_seconds > 0 else 0.0
        ),
        "decode_tokens_per_s": (
            batch_size * decode_steps / decode_seconds if decode_seconds > 0 else 0.0
        ),
        "requests_per_s": (
            batch_size / (prefill_seconds + decode_seconds)
            if prefill_seconds + decode_seconds > 0
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
        "itl_ms_p99": (
            percentile(inter_token_latency_ms, 0.99)
            if inter_token_latency_ms
            else 0.0
        ),
        "inter_token_latency_ms": inter_token_latency_ms,
        "itl_ms_min": min(inter_token_latency_ms) if inter_token_latency_ms else 0.0,
        "itl_ms_max": max(inter_token_latency_ms) if inter_token_latency_ms else 0.0,
        "seal_step_count": len(seal_step_latency_ms),
        "seal_step_itl_ms_p50": (
            statistics.median(seal_step_latency_ms)
            if seal_step_latency_ms
            else 0.0
        ),
        "seal_step_itl_ms_p95": (
            percentile(seal_step_latency_ms, 0.95)
            if seal_step_latency_ms
            else 0.0
        ),
        "seal_step_itl_ms_max": (
            max(seal_step_latency_ms) if seal_step_latency_ms else 0.0
        ),
        "non_seal_step_count": len(non_seal_step_latency_ms),
        "non_seal_step_itl_ms_p50": (
            statistics.median(non_seal_step_latency_ms)
            if non_seal_step_latency_ms
            else 0.0
        ),
        "non_seal_step_itl_ms_p95": (
            percentile(non_seal_step_latency_ms, 0.95)
            if non_seal_step_latency_ms
            else 0.0
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "process_max_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "output_token_ids": output_ids_cpu[0].tolist(),
        "output_token_ids_by_sequence": output_ids_cpu.tolist(),
        "output_token_sha256_by_sequence": row_ids_sha256(output_ids_cpu),
        "windows": windows,
    }


def make_cache(
    run: dict[str, object],
    model: QuantizedQwen3ForCausalLM,
    *,
    batch_size: int,
    max_seq_len: int,
    track_transfer_timing: bool,
    kv_group_size: int,
    hqq_optimize: bool,
    hqq_runtime_backend: str,
) -> object:
    mode = run["mode"]
    if mode == "resident":
        return model.make_resident_cache(batch_size=batch_size, max_seq_len=max_seq_len)
    budget = run["budget_mib"]
    budget_bytes = None if budget == "auto" else int(budget) * 2**20
    return OffloadedKVCache(
        num_layers=model.config.num_hidden_layers,
        batch_size=batch_size,
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
    update_status(
        args.status_output,
        status="running",
        failure_stage="model_load",
        batch_size=args.batch_size,
        context=args.input_tokens,
        generation=args.max_new_tokens,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if (
        args.max_new_tokens <= 0
        or args.window_tokens <= 0
        or args.repeats <= 0
        or args.batch_size <= 0
    ):
        raise ValueError("token counts must be positive")
    base_runs = parse_runs(args.runs)
    storage_formats = parse_storage_formats(args.storage_formats)
    runs = expand_runs(
        base_runs, storage_formats, args.repeats, args.execution_order
    )
    device = torch.device("cuda:0")
    load_start = time.perf_counter()
    store = load_quantized_cpu_weight_store(args.model, pin_memory=True, progress=True)
    runtime = QuantizedResidentRuntime(store, device=device)
    model = QuantizedQwen3ForCausalLM(store, runtime, device=device)
    load_seconds = time.perf_counter() - load_start
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids_cpu = exact_length_batch_ids(
        tokenizer,
        seed_text=args.seed_text,
        length=args.input_tokens,
        batch_size=args.batch_size,
        batch_mode=args.batch_mode,
    )
    input_ids = input_ids_cpu.to(device)
    required_capacity = args.input_tokens + args.max_new_tokens
    max_seq_len = args.cache_capacity or required_capacity
    if max_seq_len < required_capacity:
        raise ValueError(
            f"cache-capacity {max_seq_len} is below required {required_capacity}"
        )

    # One process-level CUDA and kernel warmup; every measured mode then sees
    # the same already-initialized all-resident INT8 weight runtime.
    warmup_ids = exact_length_batch_ids(
        tokenizer,
        seed_text=args.seed_text,
        length=args.warmup_input_tokens,
        batch_size=args.batch_size,
        batch_mode=args.batch_mode,
    ).to(device)
    warmup_cache = model.make_resident_cache(
        batch_size=args.batch_size,
        max_seq_len=args.warmup_input_tokens + args.warmup_new_tokens,
    )
    update_status(args.status_output, failure_stage="prefill", phase="warmup")
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
            "schema_version": 3,
            "model": str(args.model.resolve()),
            "source_revision": store.manifest.get("source_revision"),
            "weight_runtime": "int8_all_decoder_layers_resident",
            "compute_dtype": "float16",
            "input_tokens": args.input_tokens,
            "batch_size": args.batch_size,
            "batch_mode": args.batch_mode,
            "max_new_tokens": args.max_new_tokens,
            "max_seq_len": max_seq_len,
            "ignore_eos": True,
            "expected_final_cache_length": args.input_tokens
            + args.max_new_tokens
            - 1,
            "window_tokens": args.window_tokens,
            "repeats": args.repeats,
            "execution_order": args.execution_order,
            "storage_formats": storage_formats,
            "kv_group_size": args.kv_group_size,
            "hqq_optimize": args.hqq_optimize,
            "hqq_runtime_backend": args.hqq_runtime_backend,
            "track_transfer_timing": args.track_transfer_timing,
            "input_ids_sha256": tensor_ids_sha256(input_ids_cpu),
            "input_ids_sha256_by_sequence": row_ids_sha256(input_ids_cpu),
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
                "decode_steps_per_s": warmup["decode_steps_per_s"],
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
    reference_ids: list[list[int]] | None = None

    for run_idx, run in enumerate(runs):
        gc.collect()
        torch.cuda.empty_cache()
        update_status(
            args.status_output,
            failure_stage="cache_allocate",
            phase="measurement",
            requested_mode=run["mode"],
            storage_format=run["storage_format"],
        )
        cache = make_cache(
            run,
            model,
            batch_size=args.batch_size,
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
            stage_callback=lambda stage: update_status(
                args.status_output, failure_stage=stage
            ),
        )
        ids = result["output_token_ids_by_sequence"]
        if reference_ids is None:
            reference_ids = ids
            mismatches = [None] * args.batch_size
            agreement = 1.0
        else:
            mismatches, agreement = compare_token_batches(ids, reference_ids)
        mismatch = next(
            (
                {"batch_index": batch_index, "token_index": token_index}
                for batch_index, token_index in enumerate(mismatches)
                if token_index is not None
            ),
            None,
        )
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
                "first_mismatch_by_sequence_vs_first_run": mismatches,
                "token_agreement_vs_first_run": agreement,
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
            f"aggregate_decode={result['decode_tokens_per_s']:.3f} tok/s, "
            f"step_rate={result['decode_steps_per_s']:.3f} step/s, "
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
        step_values = [float(row["decode_steps_per_s"]) for row in rows]
        ttft_values = [float(row["ttft_ms"]) for row in rows]
        group_summaries[label] = {
            "runs": len(rows),
            "batch_size": args.batch_size,
            "median_decode_tokens_per_s": statistics.median(decode_values),
            "median_decode_steps_per_s": statistics.median(step_values),
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
    update_status(args.status_output, failure_stage="finalize")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if partial_path.exists():
        partial_path.unlink()
    update_status(
        args.status_output,
        status="pass",
        failure_stage=None,
        output=str(args.output),
        peak_allocated_bytes=max(
            int(row["peak_allocated_bytes"]) for row in report["measurements"]
        ),
        peak_reserved_bytes=max(
            int(row["peak_reserved_bytes"]) for row in report["measurements"]
        ),
        process_max_rss_bytes=max(
            int(row["process_max_rss_bytes"]) for row in report["measurements"]
        ),
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


def run_cli() -> None:
    try:
        main()
    except BaseException as error:
        status_path = None
        try:
            raw_args = parse_args()
            status_path = raw_args.status_output
        except BaseException:
            pass
        if isinstance(error, torch.cuda.OutOfMemoryError):
            status = "cuda_oom"
        elif isinstance(error, MemoryError):
            status = "host_oom"
        elif isinstance(error, (ValueError, RuntimeError, FloatingPointError)):
            status = "correctness_fail"
        else:
            status = "error"
        cuda_allocated = 0
        cuda_reserved = 0
        if torch.cuda.is_available():
            cuda_allocated = torch.cuda.memory_allocated()
            cuda_reserved = torch.cuda.memory_reserved()
        update_status(
            status_path,
            status=status,
            error_type=type(error).__name__,
            error_summary=str(error).splitlines()[0][:500],
            traceback_tail="\n".join(traceback.format_exc().splitlines()[-8:]),
            allocated_bytes_at_failure=cuda_allocated,
            reserved_bytes_at_failure=cuda_reserved,
            process_max_rss_bytes=int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        )
        raise


if __name__ == "__main__":
    run_cli()
