#!/usr/bin/env python3
"""Transformers correctness reference with all weights CPU-offloaded.

This is deliberately a validation-only program.  It does not use
``model.generate()`` and is never imported by the production runtime.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from accelerate import cpu_offload
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="你好，请用一句话介绍南京。")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    cpu_offload(model, execution_device=torch.device("cuda:0"))
    load_seconds = time.perf_counter() - load_start

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = model(input_ids=input_ids, use_cache=True)
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    past_key_values = output.past_key_values
    generated = [next_token]
    for _ in range(1, args.max_new_tokens):
        output = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    generated_ids = torch.cat(generated, dim=1)[0].tolist()
    report = {
        "model": str(args.model.resolve()),
        "runtime": "transformers_accelerate_cpu_offload_reference",
        "dtype": "float16",
        "attention": "sdpa",
        "prompt": args.prompt,
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": len(generated_ids),
        "generated_ids": generated_ids,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
