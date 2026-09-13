#!/usr/bin/env python3
"""Nsight Compute microbenchmark for the explicit INT8-to-FP16 linear path."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from ondeviceinfer.profiling import nvtx_enabled, nvtx_range
from ondeviceinfer.quantized_model import int8_weight_only_linear
from ondeviceinfer.runtime.quantized import qweight_name, scale_name


QWEN3_14B_LINEAR_SHAPES = {
    "self_attn.q_proj.weight": (5120, 5120),
    "self_attn.k_proj.weight": (1024, 5120),
    "self_attn.v_proj.weight": (1024, 5120),
    "self_attn.o_proj.weight": (5120, 5120),
    "mlp.gate_proj.weight": (17408, 5120),
    "mlp.up_proj.weight": (17408, 5120),
    "mlp.down_proj.weight": (5120, 17408),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight-name",
        choices=tuple(QWEN3_14B_LINEAR_SHAPES),
        default="mlp.up_proj.weight",
    )
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.m <= 0 or args.warmups < 0 or args.repeats <= 0:
        raise ValueError("invalid benchmark counts")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    n, k = QWEN3_14B_LINEAR_SHAPES[args.weight_name]
    qweight = torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8)
    scale = torch.full((n,), 0.01, device=device, dtype=torch.float16)
    inputs = torch.randn((args.m, k), device=device, dtype=torch.float16)
    scratch = torch.empty(n * k, device=device, dtype=torch.float16)
    weights = {
        qweight_name(args.weight_name): qweight,
        scale_name(args.weight_name): scale,
    }

    for _ in range(args.warmups):
        output = int8_weight_only_linear(
            inputs, weights, args.weight_name, scratch
        )
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    profiler = torch.cuda.cudart()
    start_status = int(profiler.cudaProfilerStart())
    if start_status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {start_status}")
    start_event.record()
    with nvtx_range("ncu_capture::int8_weight_only_linear"):
        for repeat in range(args.repeats):
            with nvtx_range(f"linear_repeat::{repeat}"):
                output = int8_weight_only_linear(
                    inputs, weights, args.weight_name, scratch
                )
    end_event.record()
    end_event.synchronize()
    stop_status = int(profiler.cudaProfilerStop())
    if stop_status != 0:
        raise RuntimeError(f"cudaProfilerStop failed with status {stop_status}")

    elapsed_ms = start_event.elapsed_time(end_event)
    report = {
        "schema_version": 1,
        "workload": "int8_storage_explicit_fp16_dequant_then_gemm",
        "weight_name": args.weight_name,
        "shape": {"m": args.m, "n": n, "k": k},
        "warmups": args.warmups,
        "captured_repeats": args.repeats,
        "elapsed_ms": elapsed_ms,
        "milliseconds_per_call": elapsed_ms / args.repeats,
        "qweight_bytes": qweight.numel() * qweight.element_size(),
        "dequantized_weight_bytes": scratch.numel() * scratch.element_size(),
        "nvtx_enabled": nvtx_enabled(),
        "cuda_profiler_start_status": start_status,
        "cuda_profiler_stop_status": stop_status,
        "output_checksum": float(output.float().sum().item()),
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
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
