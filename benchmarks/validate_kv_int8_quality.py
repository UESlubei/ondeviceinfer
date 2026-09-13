#!/usr/bin/env python3
"""Compare FP16-resident and HQQ-INT8-offloaded KV on one fixed token stream."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ondeviceinfer.kv_offload import OffloadedKVCache
from ondeviceinfer.quantized_model import QuantizedQwen3ForCausalLM
from ondeviceinfer.runtime.quantized import (
    QuantizedResidentRuntime,
    load_quantized_cpu_weight_store,
)


DEFAULT_TEXT = (
    "南京是一座历史文化名城，也是现代化创新城市。请结合地理、历史、文化、"
    "科技和日常生活介绍这座城市，并保持叙述连贯。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--kv-budget-mib", type=int, default=64)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--seed-text", default=DEFAULT_TEXT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def exact_length_ids(tokenizer: object, text: str, length: int) -> torch.Tensor:
    seed = tokenizer(text, add_special_tokens=True).input_ids
    if not seed:
        raise ValueError("seed text produced no tokens")
    repeated = (seed * math.ceil(length / len(seed)))[:length]
    return torch.tensor([repeated], dtype=torch.long)


def ids_hash(input_ids: torch.Tensor) -> str:
    payload = ",".join(str(value) for value in input_ids[0].tolist())
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def make_int8_cache(
    model: QuantizedQwen3ForCausalLM,
    *,
    capacity: int,
    budget_mib: int,
    group_size: int,
) -> OffloadedKVCache:
    return OffloadedKVCache(
        num_layers=model.config.num_hidden_layers,
        batch_size=1,
        num_key_value_heads=model.config.num_key_value_heads,
        max_seq_len=capacity,
        head_dim=model.config.head_dim,
        device=model.device,
        dtype=model.dtype,
        mode="prefetch",
        gpu_budget_bytes=budget_mib * 2**20,
        pin_memory=True,
        storage_format="int8_hqq",
        kv_group_size=group_size,
    )


@torch.inference_mode()
def reference_greedy(
    model: QuantizedQwen3ForCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
) -> tuple[list[torch.Tensor], list[int], object, float]:
    capacity = input_ids.shape[1] + max_new_tokens
    cache = model.make_resident_cache(batch_size=1, max_seq_len=capacity)
    start = time.perf_counter()
    output = model(input_ids, cache=cache, logits_to_keep=1)
    logits = [output.logits[:, -1].float().cpu()]
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    tokens = [int(next_token.item())]
    for _ in range(1, max_new_tokens):
        output = model(next_token, cache=cache, logits_to_keep=1)
        logits.append(output.logits[:, -1].float().cpu())
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(int(next_token.item()))
    torch.cuda.synchronize(input_ids.device)
    seconds = time.perf_counter() - start
    return logits, tokens, cache, seconds


def compare_logits(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, object]:
    reference = reference.float()
    actual = actual.float().cpu()
    reference_log_probs = F.log_softmax(reference, dim=-1)
    actual_log_probs = F.log_softmax(actual, dim=-1)
    kl = torch.sum(
        reference_log_probs.exp() * (reference_log_probs - actual_log_probs), dim=-1
    ).mean()
    # KL is non-negative in exact arithmetic.  FP32 log-softmax/reduction can
    # produce tiny negative values (around 1e-7), so avoid reporting numerical
    # round-off as a physically meaningful negative divergence.
    kl_value = max(0.0, float(kl.item()))
    return {
        "cosine": float(
            F.cosine_similarity(reference.flatten(), actual.flatten(), dim=0).item()
        ),
        "mse": float(torch.mean((reference - actual) ** 2).item()),
        "kl_reference_to_int8": kl_value,
        "reference_top1": int(reference.argmax(dim=-1).item()),
        "int8_top1": int(actual.argmax(dim=-1).item()),
        "top1_equal": bool(
            reference.argmax(dim=-1).item() == actual.argmax(dim=-1).item()
        ),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


@torch.inference_mode()
def teacher_forced_int8(
    model: QuantizedQwen3ForCausalLM,
    input_ids: torch.Tensor,
    *,
    reference_logits: list[torch.Tensor],
    reference_tokens: list[int],
    budget_mib: int,
    group_size: int,
) -> tuple[list[dict[str, object]], list[float], OffloadedKVCache, float]:
    cache = make_int8_cache(
        model,
        capacity=input_ids.shape[1] + len(reference_tokens),
        budget_mib=budget_mib,
        group_size=group_size,
    )
    start = time.perf_counter()
    output = model(input_ids, cache=cache, logits_to_keep=1)
    comparisons = [compare_logits(reference_logits[0], output.logits[:, -1])]
    nll_values = [
        -float(
            F.log_softmax(output.logits[:, -1].float(), dim=-1)[
                0, reference_tokens[0]
            ].item()
        )
    ]
    for step in range(1, len(reference_tokens)):
        token = torch.tensor(
            [[reference_tokens[step - 1]]], dtype=torch.long, device=input_ids.device
        )
        output = model(token, cache=cache, logits_to_keep=1)
        comparisons.append(compare_logits(reference_logits[step], output.logits[:, -1]))
        nll_values.append(
            -float(
                F.log_softmax(output.logits[:, -1].float(), dim=-1)[
                    0, reference_tokens[step]
                ].item()
            )
        )
    cache.synchronize()
    seconds = time.perf_counter() - start
    return comparisons, nll_values, cache, seconds


@torch.inference_mode()
def greedy_int8(
    model: QuantizedQwen3ForCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    budget_mib: int,
    group_size: int,
) -> tuple[list[int], OffloadedKVCache, float]:
    cache = make_int8_cache(
        model,
        capacity=input_ids.shape[1] + max_new_tokens,
        budget_mib=budget_mib,
        group_size=group_size,
    )
    start = time.perf_counter()
    output = model(input_ids, cache=cache, logits_to_keep=1)
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    tokens = [int(next_token.item())]
    for _ in range(1, max_new_tokens):
        output = model(next_token, cache=cache, logits_to_keep=1)
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(int(next_token.item()))
    cache.synchronize()
    seconds = time.perf_counter() - start
    return tokens, cache, seconds


def sequence_nll(logits: list[torch.Tensor], targets: list[int]) -> float:
    losses = [
        -float(F.log_softmax(row.float(), dim=-1)[0, target].item())
        for row, target in zip(logits, targets)
    ]
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.input_tokens <= 0 or args.max_new_tokens < 2:
        raise ValueError("input-tokens must be positive and max-new-tokens at least 2")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(20260912)
    store = load_quantized_cpu_weight_store(args.model, pin_memory=True, progress=True)
    runtime = QuantizedResidentRuntime(store, device=device)
    model = QuantizedQwen3ForCausalLM(store, runtime, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids_cpu = exact_length_ids(tokenizer, args.seed_text, args.input_tokens)
    input_ids = input_ids_cpu.to(device)

    # Warm all kernels before collecting either numerical path.
    warmup_cache = model.make_resident_cache(batch_size=1, max_seq_len=4)
    model(input_ids[:, :4], cache=warmup_cache, logits_to_keep=1)
    torch.cuda.synchronize(device)
    del warmup_cache

    reference_logits, reference_tokens, reference_cache, reference_seconds = (
        reference_greedy(model, input_ids, max_new_tokens=args.max_new_tokens)
    )
    reference_lengths = list(reference_cache.lengths)
    del reference_cache
    gc.collect()
    torch.cuda.empty_cache()

    comparisons, int8_nll_values, teacher_cache, teacher_seconds = teacher_forced_int8(
        model,
        input_ids,
        reference_logits=reference_logits,
        reference_tokens=reference_tokens,
        budget_mib=args.kv_budget_mib,
        group_size=args.kv_group_size,
    )
    teacher_lengths = list(teacher_cache.lengths)
    teacher_storage = {
        "managed_cpu_bytes": teacher_cache.managed_cpu_bytes,
        "managed_cpu_data_bytes": teacher_cache.managed_cpu_data_bytes,
        "managed_cpu_metadata_bytes": teacher_cache.managed_cpu_metadata_bytes,
        "managed_gpu_bytes": teacher_cache.managed_gpu_bytes,
        "host_compression_ratio": teacher_cache.host_compression_ratio,
        "effective_mode": teacher_cache.effective_mode,
        "slot_count": teacher_cache.slot_count,
        "host_is_pinned": teacher_cache.host_is_pinned,
    }
    del teacher_cache
    gc.collect()
    torch.cuda.empty_cache()

    int8_tokens, greedy_cache, greedy_seconds = greedy_int8(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        budget_mib=args.kv_budget_mib,
        group_size=args.kv_group_size,
    )
    greedy_lengths = list(greedy_cache.lengths)
    del greedy_cache

    cosines = [float(row["cosine"]) for row in comparisons]
    mses = [float(row["mse"]) for row in comparisons]
    kls = [float(row["kl_reference_to_int8"]) for row in comparisons]
    top1_equal = [bool(row["top1_equal"]) for row in comparisons]
    reference_nll = sequence_nll(reference_logits, reference_tokens)
    int8_nll = sum(int8_nll_values) / len(int8_nll_values)
    reference_ppl = math.exp(reference_nll)
    int8_ppl = math.exp(int8_nll)
    token_matches = sum(a == b for a, b in zip(reference_tokens, int8_tokens))
    first_mismatch = next(
        (index for index, pair in enumerate(zip(reference_tokens, int8_tokens)) if pair[0] != pair[1]),
        None,
    )
    expected_length = args.input_tokens + args.max_new_tokens - 1
    properties = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "configuration": {
            "input_tokens": args.input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "kv_storage_format": "int8_hqq",
            "kv_group_size": args.kv_group_size,
            "hqq_optimize": False,
            "hqq_runtime_backend": "torch",
            "kv_budget_mib": args.kv_budget_mib,
            "input_ids_sha256": ids_hash(input_ids_cpu),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": properties.name,
        },
        "timing_seconds": {
            "resident_reference": reference_seconds,
            "int8_teacher_forced": teacher_seconds,
            "int8_greedy": greedy_seconds,
        },
        "logits": {
            "steps": len(comparisons),
            "cosine_mean": sum(cosines) / len(cosines),
            "cosine_min": min(cosines),
            "mse_mean": sum(mses) / len(mses),
            "mse_max": max(mses),
            "kl_mean": sum(kls) / len(kls),
            "kl_max": max(kls),
            "teacher_forced_top1_agreement": sum(top1_equal) / len(top1_equal),
            "all_finite": all(bool(row["finite"]) for row in comparisons),
            "per_step": comparisons,
        },
        "perplexity_on_reference_greedy_stream": {
            "resident_fp16_kv": reference_ppl,
            "int8_hqq_kv": int8_ppl,
            "relative_change": int8_ppl / reference_ppl - 1,
        },
        "greedy": {
            "token_agreement": token_matches / len(reference_tokens),
            "first_mismatch": first_mismatch,
            "reference_token_ids": reference_tokens,
            "int8_token_ids": int8_tokens,
            "reference_text": tokenizer.decode(reference_tokens),
            "int8_text": tokenizer.decode(int8_tokens),
        },
        "cache": {
            **teacher_storage,
            "expected_final_length": expected_length,
            "reference_all_lengths_valid": all(
                length == expected_length for length in reference_lengths
            ),
            "teacher_int8_all_lengths_valid": all(
                length == expected_length for length in teacher_lengths
            ),
            "greedy_int8_all_lengths_valid": all(
                length == expected_length for length in greedy_lengths
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cosine_min": report["logits"]["cosine_min"],
                "top1_agreement": report["logits"]["teacher_forced_top1_agreement"],
                "greedy_agreement": report["greedy"]["token_agreement"],
                "ppl_relative_change": report["perplexity_on_reference_greedy_stream"]["relative_change"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
