#!/usr/bin/env python3
"""Summarize CUDA overlap from an Nsight Systems SQLite export.

The script deliberately reports both summed work and merged wall-clock
occupancy.  CUDA streams can overlap, so simply adding operation durations
overstates their contribution to end-to-end latency.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable


Interval = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def duration_ns(intervals: Iterable[Interval]) -> int:
    return sum(end - start for start, end in intervals)


def intersection_ns(left: list[Interval], right: list[Interval]) -> int:
    left = merge_intervals(left)
    right = merge_intervals(right)
    left_index = 0
    right_index = 0
    overlap = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def fetch_intervals(
    connection: sqlite3.Connection, query: str, parameters: tuple[object, ...] = ()
) -> list[Interval]:
    return [
        (int(start), int(end))
        for start, end in connection.execute(query, parameters)
    ]


def copy_summary(
    connection: sqlite3.Connection,
    *,
    copy_kind: int,
    kernel_union: list[Interval],
) -> dict[str, object]:
    intervals = fetch_intervals(
        connection,
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = ?",
        (copy_kind,),
    )
    union = merge_intervals(intervals)
    work = duration_ns(intervals)
    occupied = duration_ns(union)
    overlap = intersection_ns(union, kernel_union)
    count, byte_count = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(bytes), 0) "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = ?",
        (copy_kind,),
    ).fetchone()
    return {
        "copies": int(count),
        "bytes": int(byte_count),
        "work_ms": work / 1e6,
        "union_ms": occupied / 1e6,
        "achieved_gbps_from_work": (
            float(byte_count) / work if work else None
        ),
        "kernel_overlap_ms": overlap / 1e6,
        "copy_hidden_ratio": overlap / occupied if occupied else None,
    }


def main() -> None:
    args = parse_args()
    connection = sqlite3.connect(args.sqlite)
    try:
        kernels = fetch_intervals(
            connection, "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
        all_copies = fetch_intervals(
            connection, "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        )
        kernel_union = merge_intervals(kernels)
        busy_union = merge_intervals([*kernels, *all_copies])
        all_intervals = [*kernels, *all_copies]
        if not all_intervals:
            raise RuntimeError("SQLite export contains no CUDA kernel or memcpy rows")
        span_start = min(start for start, _ in all_intervals)
        span_end = max(end for _, end in all_intervals)
        span_ns = span_end - span_start
        kernel_occupied_ns = duration_ns(kernel_union)
        report = {
            "schema_version": 1,
            "source": str(args.sqlite.resolve()),
            "profile_span_ms": span_ns / 1e6,
            "kernel": {
                "launches": len(kernels),
                "work_ms": duration_ns(kernels) / 1e6,
                "union_ms": kernel_occupied_ns / 1e6,
            },
            "h2d": copy_summary(
                connection, copy_kind=1, kernel_union=kernel_union
            ),
            "d2h": copy_summary(
                connection, copy_kind=2, kernel_union=kernel_union
            ),
            "gpu_busy_union_ms": duration_ns(busy_union) / 1e6,
            "gpu_idle_ratio_within_activity_span": (
                1.0 - duration_ns(busy_union) / span_ns if span_ns else None
            ),
        }
    finally:
        connection.close()

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
