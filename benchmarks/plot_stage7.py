#!/usr/bin/env python3
"""Render Stage 7 KV budget and 4K+1K comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-scan", type=Path, required=True)
    parser.add_argument("--long-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def label(row: dict[str, object]) -> str:
    mode = str(row["requested_mode"])
    if mode == "resident":
        return "resident"
    return f"{mode} {row['requested_budget_mib']} MiB"


def main() -> None:
    args = parse_args()
    scan = json.loads(args.budget_scan.read_text(encoding="utf-8"))["measurements"]
    long_rows = json.loads(args.long_run.read_text(encoding="utf-8"))["measurements"]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    budgets = [int(row["requested_budget_mib"]) for row in scan]
    throughput = [float(row["decode_tokens_per_s"]) for row in scan]
    managed = [float(row["managed_kv_gpu_bytes"]) / 2**20 for row in scan]
    effective = [str(row["effective_mode"]) for row in scan]

    axes[0, 0].plot(budgets, throughput, marker="o", color="#4C78A8")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_title("KV budget scan (512 + 64 tokens)")
    axes[0, 0].set_xlabel("Requested KV GPU budget (MiB)")
    axes[0, 0].set_ylabel("Decode token/s")
    axes[0, 0].grid(alpha=0.25)
    for x, y, mode in zip(budgets, throughput, effective):
        axes[0, 0].annotate(mode, (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)

    axes[0, 1].plot(budgets, managed, marker="o", color="#F58518")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_title("Requested vs actually managed GPU KV")
    axes[0, 1].set_xlabel("Requested budget (MiB)")
    axes[0, 1].set_ylabel("Managed GPU KV (MiB)")
    axes[0, 1].grid(alpha=0.25)

    long_labels = [label(row) for row in long_rows]
    colors = ["#59A14F", "#E45756", "#4C78A8"]
    long_throughput = [float(row["decode_tokens_per_s"]) for row in long_rows]
    bars = axes[1, 0].bar(long_labels, long_throughput, color=colors)
    axes[1, 0].set_title("4K prompt + 1K generation")
    axes[1, 0].set_ylabel("Decode token/s")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, long_throughput):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    long_gpu = [float(row["managed_kv_gpu_bytes"]) / 2**20 for row in long_rows]
    bars = axes[1, 1].bar(long_labels, long_gpu, color=colors)
    axes[1, 1].set_title("Managed GPU KV at 5120-token capacity")
    axes[1, 1].set_ylabel("MiB")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, long_gpu):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)

    figure.suptitle("Qwen3-14B INT8: exact KV-cache offloading")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
