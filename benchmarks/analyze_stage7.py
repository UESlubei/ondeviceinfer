#!/usr/bin/env python3
"""Validate Stage 7 byte accounting and derive the final comparison summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    manifest = raw["manifest"]
    rows = raw["measurements"]
    input_tokens = int(manifest["input_tokens"])
    output_tokens = int(manifest["max_new_tokens"])
    capacity = int(manifest["max_seq_len"])

    # Qwen3-14B: 40 layers, K+V, 8 KV heads, head_dim 128, FP16.
    layers = 40
    per_layer_per_token_bytes = 2 * 8 * 128 * 2
    per_model_per_token_bytes = layers * per_layer_per_token_bytes
    expected_final_length = input_tokens + output_tokens - 1
    expected_d2h = expected_final_length * per_model_per_token_bytes
    expected_h2d = (
        sum(range(input_tokens, input_tokens + output_tokens - 1))
        * per_model_per_token_bytes
    )
    expected_resident_gpu = capacity * per_model_per_token_bytes
    expected_layer_capacity = capacity * per_layer_per_token_bytes

    reference_ids = rows[0]["output_token_ids"]
    validations: list[dict[str, object]] = []
    for row in rows:
        mode = row["effective_mode"]
        is_resident = mode == "resident"
        expected_row_h2d = 0 if is_resident else expected_h2d
        expected_row_d2h = 0 if is_resident else expected_d2h
        expected_gpu = (
            expected_resident_gpu
            if is_resident
            else int(row["slot_count"]) * expected_layer_capacity
        )
        checks = {
            "final_cache_length": row["final_cache_length"] == expected_final_length,
            "kv_h2d_bytes": row["kv_h2d_bytes"] == expected_row_h2d,
            "kv_d2h_bytes": row["kv_d2h_bytes"] == expected_row_d2h,
            "managed_kv_gpu_bytes": row["managed_kv_gpu_bytes"] == expected_gpu,
            "output_tokens": len(row["output_token_ids"]) == output_tokens,
            "token_ids": row["output_token_ids"] == reference_ids,
        }
        layer_lengths = row.get("all_layer_cache_lengths")
        if layer_lengths is not None:
            checks["all_layer_cache_lengths"] = (
                len(layer_lengths) == layers
                and all(length == expected_final_length for length in layer_lengths)
            )
        validations.append(
            {
                "requested_mode": row["requested_mode"],
                "effective_mode": mode,
                "all_layer_lengths_recorded": layer_lengths is not None,
                "checks": checks,
                "all_checks_passed": all(checks.values()),
            }
        )

    resident = next(row for row in rows if row["effective_mode"] == "resident")
    sync = next(row for row in rows if row["requested_mode"] == "sync")
    prefetch = next(
        row
        for row in rows
        if row["requested_mode"] == "prefetch"
        and row["effective_mode"] == "prefetch"
    )
    summary = {
        "schema_version": 1,
        "theory": {
            "per_layer_per_token_bytes": per_layer_per_token_bytes,
            "per_model_per_token_bytes": per_model_per_token_bytes,
            "per_model_per_token_kib": per_model_per_token_bytes / 2**10,
            "capacity_tokens": capacity,
            "resident_kv_bytes": expected_resident_gpu,
            "one_layer_slot_bytes": expected_layer_capacity,
            "final_cache_length": expected_final_length,
            "offload_expected_h2d_bytes": expected_h2d,
            "offload_expected_d2h_bytes": expected_d2h,
        },
        "validations": validations,
        "all_validations_passed": all(
            validation["all_checks_passed"] for validation in validations
        ),
        "comparisons": {
            "resident_decode_tokens_per_s": resident["decode_tokens_per_s"],
            "sync_decode_tokens_per_s": sync["decode_tokens_per_s"],
            "prefetch_decode_tokens_per_s": prefetch["decode_tokens_per_s"],
            "sync_speed_ratio_vs_resident": sync["decode_tokens_per_s"]
            / resident["decode_tokens_per_s"],
            "prefetch_speed_ratio_vs_resident": prefetch["decode_tokens_per_s"]
            / resident["decode_tokens_per_s"],
            "prefetch_speedup_vs_sync": prefetch["decode_tokens_per_s"]
            / sync["decode_tokens_per_s"],
            "sync_managed_gpu_reduction": 1
            - sync["managed_kv_gpu_bytes"] / resident["managed_kv_gpu_bytes"],
            "prefetch_managed_gpu_reduction": 1
            - prefetch["managed_kv_gpu_bytes"] / resident["managed_kv_gpu_bytes"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_validations_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
