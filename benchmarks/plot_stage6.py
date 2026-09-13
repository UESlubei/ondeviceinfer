#!/usr/bin/env python3
"""Render the Stage 6 FP16/INT8 offloading comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--int8-sync", type=Path, required=True)
    parser.add_argument("--int8-prefetch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fp16 = json.loads(args.fp16.read_text(encoding="utf-8"))
    int8_sync = json.loads(args.int8_sync.read_text(encoding="utf-8"))
    int8_prefetch = json.loads(args.int8_prefetch.read_text(encoding="utf-8"))
    fp16_by_budget = {
        row["gpu_layer_budget"]: row for row in fp16["summaries"]
    }
    rows = [
        {
            "label": "FP16 sync",
            "decode": fp16_by_budget[1]["median_decode_tokens_per_s"],
            "h2d": fp16_by_budget[1]["weight_h2d_bytes_per_run"] / 2**30,
            "peak": fp16_by_budget[1]["max_peak_allocated_bytes"] / 2**30,
        },
        {
            "label": "FP16 prefetch",
            "decode": fp16_by_budget[2]["median_decode_tokens_per_s"],
            "h2d": fp16_by_budget[2]["weight_h2d_bytes_per_run"] / 2**30,
            "peak": fp16_by_budget[2]["max_peak_allocated_bytes"] / 2**30,
        },
        {
            "label": "INT8 sync",
            "decode": int8_sync["summary"]["median_decode_tokens_per_s"],
            "h2d": int8_sync["measurements"][0]["weight_h2d_bytes"] / 2**30,
            "peak": int8_sync["summary"]["max_peak_allocated_bytes"] / 2**30,
        },
        {
            "label": "INT8 prefetch",
            "decode": int8_prefetch["summary"]["median_decode_tokens_per_s"],
            "h2d": int8_prefetch["measurements"][0]["weight_h2d_bytes"] / 2**30,
            "peak": int8_prefetch["summary"]["max_peak_allocated_bytes"] / 2**30,
        },
    ]
    labels = [row["label"] for row in rows]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [
        ("decode", "Decode throughput", "token/s"),
        ("h2d", "Weight H2D per 32-token run", "GiB"),
        ("peak", "Peak allocated memory", "GiB"),
    ]
    for axis, (key, title, ylabel) in zip(axes, metrics):
        values = [row[key] for row in rows]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle("Qwen3-14B weight offloading: FP16 vs INT8")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
