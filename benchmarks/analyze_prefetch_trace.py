#!/usr/bin/env python3
"""Calculate CUDA memcpy/kernel interval overlap from a Chrome trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(intervals: list[tuple[float, float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def total(intervals: list[list[float]]) -> float:
    return sum(end - start for start, end in intervals)


def intersection(a: list[list[float]], b: list[list[float]]) -> float:
    i = j = 0
    result = 0.0
    while i < len(a) and j < len(b):
        result += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = json.loads(args.trace.read_text(encoding="utf-8"))["traceEvents"]
    copies = merge(
        [
            (event["ts"], event["ts"] + event["dur"])
            for event in events
            if event.get("ph") == "X"
            and event.get("cat") == "gpu_memcpy"
            and "Pinned -> Device" in event.get("name", "")
        ]
    )
    kernels = merge(
        [
            (event["ts"], event["ts"] + event["dur"])
            for event in events
            if event.get("ph") == "X" and event.get("cat") == "kernel"
        ]
    )
    copy_us = total(copies)
    kernel_us = total(kernels)
    overlap_us = intersection(copies, kernels)
    result = {
        "copy_ms": copy_us / 1000,
        "kernel_ms": kernel_us / 1000,
        "overlap_ms": overlap_us / 1000,
        "copy_overlap_ratio": overlap_us / copy_us if copy_us else 0.0,
        "kernel_overlap_ratio": overlap_us / kernel_us if kernel_us else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
