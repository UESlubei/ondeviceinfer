#!/usr/bin/env python3
"""Plot layer budget versus throughput, memory, and H2D traffic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))["summaries"]
    budgets = [row["gpu_layer_budget"] for row in rows]
    throughput = [row["median_decode_tokens_per_s"] for row in rows]
    memory = [row["max_peak_allocated_bytes"] / 2**30 for row in rows]
    h2d = [row["weight_h2d_bytes_per_run"] / 2**30 for row in rows]

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(budgets, throughput, marker="o")
    axes[0].set(xlabel="GPU layer budget", ylabel="Decode tokens/s", title="Throughput")
    axes[1].plot(budgets, memory, marker="o", color="tab:orange")
    axes[1].set(xlabel="GPU layer budget", ylabel="Peak allocated GiB", title="GPU memory")
    axes[2].plot(budgets, h2d, marker="o", color="tab:green")
    axes[2].set(xlabel="GPU layer budget", ylabel="H2D GiB / 32-token run", title="Weight traffic")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.set_xticks(budgets)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
