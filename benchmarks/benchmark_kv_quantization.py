#!/usr/bin/env python3
"""Microbenchmark HQQ INT8 K/V encode/decode on representative Qwen3 shapes."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ondeviceinfer.kv_quantization import HQQInt8KVCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", default="1,512,4096")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--include-hqq-optimized",
        action="store_true",
        help="Also measure HQQ's iterative optimization; runtime default is disabled",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    index = max(0, math.ceil(probability * len(values)) - 1)
    return sorted(values)[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 0.95),
    }


def timed_cuda_call(function: object) -> tuple[float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    function()  # type: ignore[operator]
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    return float(start_event.elapsed_time(end_event)), wall_ms


@torch.inference_mode()
def measure_shape(
    *,
    sequence_length: int,
    heads: int,
    head_dim: int,
    group_size: int,
    optimize: bool,
    runtime_backend: str,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, object]:
    shape = (1, sequence_length, heads, head_dim)
    codec = HQQInt8KVCodec(
        compute_dtype=torch.float16,
        head_dim=head_dim,
        group_size=group_size,
        optimize=optimize,
        runtime_backend=runtime_backend,
    )
    source = torch.randn(shape, dtype=torch.float16, device=device)
    encoded = codec.allocate(shape, device=device)
    reconstructed = torch.empty_like(source)
    for _ in range(warmup):
        codec.encode_into(source, encoded)
        codec.decode_into(encoded, reconstructed)
        torch.cuda.synchronize(device)

    persistent_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    quant_cuda_ms: list[float] = []
    quant_wall_ms: list[float] = []
    dequant_cuda_ms: list[float] = []
    dequant_wall_ms: list[float] = []
    for _ in range(repeats):
        cuda_ms, wall_ms = timed_cuda_call(
            lambda: codec.encode_into(source, encoded)
        )
        quant_cuda_ms.append(cuda_ms)
        quant_wall_ms.append(wall_ms)
        cuda_ms, wall_ms = timed_cuda_call(
            lambda: codec.decode_into(encoded, reconstructed)
        )
        dequant_cuda_ms.append(cuda_ms)
        dequant_wall_ms.append(wall_ms)

    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    source_fp32 = source.float()
    reconstructed_fp32 = reconstructed.float()
    mse = float(torch.mean((source_fp32 - reconstructed_fp32) ** 2).item())
    cosine = float(
        F.cosine_similarity(
            source_fp32.flatten(), reconstructed_fp32.flatten(), dim=0
        ).item()
    )
    q_bytes = encoded.data.numel() * encoded.data.element_size()
    assert encoded.scale is not None and encoded.zero is not None
    metadata_bytes = (
        encoded.scale.numel() * encoded.scale.element_size()
        + encoded.zero.numel() * encoded.zero.element_size()
    )
    fp16_bytes = source.numel() * source.element_size()
    return {
        "shape": list(shape),
        "hqq_optimize": optimize,
        "runtime_backend": runtime_backend,
        "hqq_version": codec.library_version,
        "qdata_bytes": q_bytes,
        "metadata_bytes": metadata_bytes,
        "encoded_bytes": q_bytes + metadata_bytes,
        "fp16_bytes": fp16_bytes,
        "encoded_ratio_vs_fp16": (q_bytes + metadata_bytes) / fp16_bytes,
        "space_reduction_vs_fp16": 1 - (q_bytes + metadata_bytes) / fp16_bytes,
        "mse": mse,
        "cosine_similarity": cosine,
        "quant_cuda_ms": summarize(quant_cuda_ms),
        "quant_wall_ms": summarize(quant_wall_ms),
        "dequant_cuda_ms": summarize(dequant_cuda_ms),
        "dequant_wall_ms": summarize(dequant_wall_ms),
        "persistent_allocated_bytes": persistent_allocated,
        "allocated_after_bytes": allocated_after,
        "allocated_growth_bytes": allocated_after - persistent_allocated,
        "peak_extra_allocated_bytes": peak_allocated - persistent_allocated,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    sequence_lengths = [int(item) for item in args.sequence_lengths.split(",")]
    if any(length <= 0 for length in sequence_lengths):
        raise ValueError("all sequence lengths must be positive")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(20260912)
    variants = [("torch", False), ("hqq_reference", False)]
    if args.include_hqq_optimized:
        variants.append(("hqq_reference", True))
    measurements = []
    for runtime_backend, optimize in variants:
        for sequence_length in sequence_lengths:
            row = measure_shape(
                sequence_length=sequence_length,
                heads=args.heads,
                head_dim=args.head_dim,
                group_size=args.group_size,
                optimize=optimize,
                runtime_backend=runtime_backend,
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )
            measurements.append(row)
            print(
                f"S={sequence_length} backend={runtime_backend} "
                f"optimize={optimize}: "
                f"quant={row['quant_wall_ms']['median']:.3f} ms, "  # type: ignore[index]
                f"dequant={row['dequant_wall_ms']['median']:.3f} ms, "  # type: ignore[index]
                f"cos={row['cosine_similarity']:.8f}",
                flush=True,
            )
            del row
            torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": 1,
        "benchmark": "hqq_int8_kv_codec",
        "configuration": {
            "sequence_lengths": sequence_lengths,
            "batch_size": 1,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "group_size": args.group_size,
            "nbits": 8,
            "metadata_dtype": "float16",
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
