#!/usr/bin/env python3
"""Compare the custom Qwen3 implementation with a Transformers reference.

Transformers is intentionally imported only in this validation program.  The
production model and generation loop do not depend on a Transformers model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ondeviceinfer.generation import greedy_generate
from ondeviceinfer.weights import load_resident_model


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="你好，请用一句话介绍南京。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def reference_greedy(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(input_ids=input_ids, use_cache=True)
    prefill_logits = output.logits[:, -1].float()
    past_key_values = output.past_key_values
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    for _ in range(1, max_new_tokens):
        if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
            break
        output = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
    return prefill_logits, torch.cat(generated, dim=1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-model correctness check")
    dtype = DTYPES[args.dtype]
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    custom = load_resident_model(args.model, device=device, dtype=dtype)
    custom_cache = custom.make_resident_cache(
        batch_size=1,
        max_seq_len=input_ids.shape[1] + args.max_new_tokens,
    )
    custom_prefill = custom(input_ids, cache=custom_cache, logits_to_keep=1).logits[
        :, -1
    ].float()
    custom_cache.reset()
    custom_result = greedy_generate(
        custom,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        cache=custom_cache,
        eos_token_id=custom.config.eos_token_id,
    )

    reference = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).eval().to(device)
    reference_prefill, reference_ids = reference_greedy(
        reference,
        input_ids,
        args.max_new_tokens,
        custom.config.eos_token_id,
    )

    common = min(custom_result.generated_ids.shape[1], reference_ids.shape[1])
    equal_prefix = custom_result.generated_ids[:, :common] == reference_ids[:, :common]
    mismatch_positions = (~equal_prefix).nonzero(as_tuple=False)
    first_mismatch = (
        int(mismatch_positions[0, 1]) if mismatch_positions.numel() else None
    )
    difference = (custom_prefill - reference_prefill).abs()
    report = {
        "model": str(args.model.resolve()),
        "dtype": args.dtype,
        "prompt": args.prompt,
        "input_tokens": int(input_ids.shape[1]),
        "requested_new_tokens": args.max_new_tokens,
        "custom_new_tokens": int(custom_result.generated_ids.shape[1]),
        "reference_new_tokens": int(reference_ids.shape[1]),
        "matching_prefix_tokens": common if first_mismatch is None else first_mismatch,
        "first_mismatch": first_mismatch,
        "prefill_top1_match": bool(
            custom_prefill.argmax(dim=-1).eq(reference_prefill.argmax(dim=-1)).all()
        ),
        "prefill_logits_max_abs_error": float(difference.max().item()),
        "prefill_logits_mean_abs_error": float(difference.mean().item()),
        "custom_generated_ids": custom_result.generated_ids[0].tolist(),
        "reference_generated_ids": reference_ids[0].tolist(),
        "custom_text": tokenizer.decode(
            custom_result.generated_ids[0], skip_special_tokens=True
        ),
        "reference_text": tokenizer.decode(reference_ids[0], skip_special_tokens=True),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if first_mismatch is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
