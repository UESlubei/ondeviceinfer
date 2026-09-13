#!/usr/bin/env python3
"""Compare Qwen3-14B FP16 and INT8 logits and greedy continuations."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ondeviceinfer.generation import greedy_generate
from ondeviceinfer.offload_model import OffloadedQwen3ForCausalLM
from ondeviceinfer.quantized_model import QuantizedQwen3ForCausalLM
from ondeviceinfer.runtime.offload import PrefetchOffloadRuntime, load_cpu_weight_store
from ondeviceinfer.runtime.quantized import (
    QuantizedResidentRuntime,
    load_quantized_cpu_weight_store,
)


DEFAULT_PROMPTS = [
    {"id": "zh_city", "text": "你好，请用一句话介绍南京。"},
    {"id": "zh_math", "text": "请计算 17 乘以 23，并简要说明过程。"},
    {"id": "en_fact", "text": "In one sentence, explain why the sky appears blue."},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", type=Path, required=True)
    parser.add_argument("--int8-model", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--fp16-gpu-layer-budget", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def longest_common_prefix(first: list[int], second: list[int]) -> int:
    count = 0
    for left, right in zip(first, second):
        if left != right:
            break
        count += 1
    return count


def logits_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, object]:
    reference = reference.float()
    actual = actual.float()
    difference = actual - reference
    return {
        "mae": difference.abs().mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "max_abs_error": difference.abs().max().item(),
        "relative_l2": (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-30)
        ).item(),
        "cosine_similarity": F.cosine_similarity(
            reference.flatten(), actual.flatten(), dim=0
        ).item(),
        "fp16_top1": int(reference.argmax().item()),
        "int8_top1": int(actual.argmax().item()),
        "top1_agreement": bool(reference.argmax() == actual.argmax()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        args.fp16_model, local_files_only=True
    )
    encoded = [
        tokenizer(row["text"], return_tensors="pt").input_ids
        for row in DEFAULT_PROMPTS
    ]

    fp16_load_start = time.perf_counter()
    fp16_store = load_cpu_weight_store(
        args.fp16_model, dtype=torch.float16, pin_memory=True
    )
    fp16_runtime = PrefetchOffloadRuntime(
        fp16_store,
        device=device,
        gpu_layer_budget=args.fp16_gpu_layer_budget,
    )
    fp16_model = OffloadedQwen3ForCausalLM(fp16_store, fp16_runtime, device=device)
    fp16_load_seconds = time.perf_counter() - fp16_load_start
    references: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats(device)
    for prompt, ids_cpu in zip(DEFAULT_PROMPTS, encoded):
        ids = ids_cpu.to(device)
        logits_cache = fp16_model.make_resident_cache(
            batch_size=1, max_seq_len=ids.shape[1]
        )
        start = time.perf_counter()
        logits = fp16_model(ids, cache=logits_cache, logits_to_keep=1).logits
        torch.cuda.synchronize(device)
        logits_seconds = time.perf_counter() - start
        generation_cache = fp16_model.make_resident_cache(
            batch_size=1,
            max_seq_len=ids.shape[1] + args.max_new_tokens,
        )
        generated = greedy_generate(
            fp16_model,
            ids,
            max_new_tokens=args.max_new_tokens,
            cache=generation_cache,
            eos_token_id=None,
        )
        references.append(
            {
                "id": prompt["id"],
                "input_ids": ids_cpu[0].tolist(),
                "logits": logits[0, -1].float().cpu(),
                "logits_seconds": logits_seconds,
                "generated_ids": generated.generated_ids[0].tolist(),
                "prefill_ms": generated.prefill_seconds * 1000,
                "decode_tokens_per_s": generated.decode_tokens_per_second,
            }
        )
        print(f"FP16 reference complete: {prompt['id']}", flush=True)
    fp16_peak = torch.cuda.max_memory_allocated(device)
    torch.cuda.synchronize(device)
    del fp16_model, fp16_runtime, fp16_store, logits_cache, generation_cache, generated
    gc.collect()
    torch.cuda.empty_cache()

    int8_load_start = time.perf_counter()
    int8_store = load_quantized_cpu_weight_store(
        args.int8_model, pin_memory=True, progress=True
    )
    int8_runtime = QuantizedResidentRuntime(int8_store, device=device)
    int8_model = QuantizedQwen3ForCausalLM(int8_store, int8_runtime, device=device)
    int8_load_seconds = time.perf_counter() - int8_load_start
    rows: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats(device)
    for prompt, ids_cpu, reference in zip(DEFAULT_PROMPTS, encoded, references):
        ids = ids_cpu.to(device)
        logits_cache = int8_model.make_resident_cache(
            batch_size=1, max_seq_len=ids.shape[1]
        )
        start = time.perf_counter()
        logits = int8_model(ids, cache=logits_cache, logits_to_keep=1).logits
        torch.cuda.synchronize(device)
        logits_seconds = time.perf_counter() - start
        generation_cache = int8_model.make_resident_cache(
            batch_size=1,
            max_seq_len=ids.shape[1] + args.max_new_tokens,
        )
        generated = greedy_generate(
            int8_model,
            ids,
            max_new_tokens=args.max_new_tokens,
            cache=generation_cache,
            eos_token_id=None,
        )
        int8_ids = generated.generated_ids[0].tolist()
        fp16_ids = reference["generated_ids"]
        row = {
            "id": prompt["id"],
            "prompt": prompt["text"],
            "input_tokens": int(ids.shape[1]),
            "logits": logits_metrics(reference["logits"], logits[0, -1].cpu()),
            "fp16_generated_ids": fp16_ids,
            "int8_generated_ids": int8_ids,
            "longest_common_prefix_tokens": longest_common_prefix(fp16_ids, int8_ids),
            "fp16_output_text": tokenizer.decode(
                fp16_ids, skip_special_tokens=False
            ),
            "int8_output_text": tokenizer.decode(
                int8_ids, skip_special_tokens=False
            ),
            "fp16_logits_ms": float(reference["logits_seconds"]) * 1000,
            "int8_logits_ms": logits_seconds * 1000,
            "fp16_prefill_ms": reference["prefill_ms"],
            "int8_prefill_ms": generated.prefill_seconds * 1000,
            "fp16_decode_tokens_per_s": reference["decode_tokens_per_s"],
            "int8_decode_tokens_per_s": generated.decode_tokens_per_second,
        }
        rows.append(row)
        print(
            f"INT8 quality complete: {prompt['id']}, "
            f"LCP={row['longest_common_prefix_tokens']}/{args.max_new_tokens}",
            flush=True,
        )
    int8_peak = torch.cuda.max_memory_allocated(device)

    report = {
        "manifest": {
            "schema_version": 1,
            "fp16_model": str(args.fp16_model.resolve()),
            "int8_model": str(args.int8_model.resolve()),
            "source_revision": int8_store.manifest.get("source_revision"),
            "max_new_tokens": args.max_new_tokens,
            "fp16_runtime": "prefetch",
            "fp16_gpu_layer_budget": args.fp16_gpu_layer_budget,
            "int8_runtime": "all_decoder_layers_resident",
            "compute_dtype": "float16",
            "fp16_load_seconds": fp16_load_seconds,
            "int8_load_seconds": int8_load_seconds,
            "fp16_peak_allocated_bytes": fp16_peak,
            "int8_peak_allocated_bytes": int8_peak,
        },
        "summary": {
            "prompt_count": len(rows),
            "top1_agreement_count": sum(
                bool(row["logits"]["top1_agreement"]) for row in rows
            ),
            "mean_logits_mae": sum(float(row["logits"]["mae"]) for row in rows)
            / len(rows),
            "mean_cosine_similarity": sum(
                float(row["logits"]["cosine_similarity"]) for row in rows
            )
            / len(rows),
            "mean_lcp_tokens": sum(
                int(row["longest_common_prefix_tokens"]) for row in rows
            )
            / len(rows),
        },
        "quality_comparison": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
