#!/usr/bin/env python3
"""Merge independently completed Stage 7 modes without altering measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    reference = sources[-1]["manifest"]
    controlled = (
        "model",
        "source_revision",
        "weight_runtime",
        "compute_dtype",
        "input_tokens",
        "max_new_tokens",
        "max_seq_len",
        "ignore_eos",
        "input_ids_sha256",
    )
    for source in sources[:-1]:
        mismatched = [
            key
            for key in controlled
            if source["manifest"].get(key) != reference.get(key)
        ]
        if mismatched:
            raise ValueError(f"Controlled manifest fields differ: {mismatched}")

    rows_by_mode: dict[tuple[object, object], dict[str, object]] = {}
    for source in sources:
        for row in source["measurements"]:
            key = (row["requested_mode"], row["requested_budget_mib"])
            # Later inputs deliberately replace an interrupted run's earlier
            # implementation of the same mode while preserving other modes.
            rows_by_mode[key] = row
    rows = list(rows_by_mode.values())
    mode_order = {"resident": 0, "sync": 1, "prefetch": 2}
    rows.sort(key=lambda row: mode_order[str(row["requested_mode"])])
    expected_ids = next(
        row["output_token_ids"]
        for row in rows
        if row["requested_mode"] == "resident"
    )
    for index, row in enumerate(rows):
        row["run_index"] = index
        actual = row["output_token_ids"]
        mismatch = next(
            (
                token_idx
                for token_idx, (left, right) in enumerate(zip(actual, expected_ids))
                if left != right
            ),
            None,
        )
        if mismatch is None and len(actual) != len(expected_ids):
            mismatch = min(len(actual), len(expected_ids))
        row["first_mismatch_vs_first_run"] = mismatch

    manifest = dict(reference)
    manifest["runs"] = [
        {
            "mode": row["requested_mode"],
            "budget_mib": row["requested_budget_mib"],
        }
        for row in rows
    ]
    manifest["merged_from"] = [str(path) for path in args.inputs]
    merged = {
        "manifest": manifest,
        "measurements": rows,
        "summary": {
            "all_token_sequences_equal": all(
                row["first_mismatch_vs_first_run"] is None for row in rows
            ),
            "completed_run_count": len(rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(merged["summary"], indent=2))


if __name__ == "__main__":
    main()
