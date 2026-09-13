"""Prefill and greedy decode loop implemented without model.generate()."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from .cache import KVCache
from .model import Qwen3ForCausalLM


@dataclass
class GenerationResult:
    token_ids: Tensor
    generated_ids: Tensor
    prefill_seconds: float
    decode_seconds: float

    @property
    def decode_tokens_per_second(self) -> float:
        decode_tokens = max(0, self.generated_ids.shape[1] - 1)
        return decode_tokens / self.decode_seconds if self.decode_seconds > 0 else 0.0


@torch.inference_mode()
def greedy_generate(
    model: Qwen3ForCausalLM,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    cache: KVCache,
    eos_token_id: int | None = None,
) -> GenerationResult:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[0] != 1:
        raise NotImplementedError("The first learning baseline supports batch size 1")

    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = model(input_ids, cache=cache, logits_to_keep=1)
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - start

    generated = [next_token]
    decode_start = time.perf_counter()
    for _ in range(1, max_new_tokens):
        if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
            break
        output = model(next_token, cache=cache, logits_to_keep=1)
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_start

    generated_ids = torch.cat(generated, dim=1)
    return GenerationResult(
        token_ids=torch.cat((input_ids, generated_ids), dim=1),
        generated_ids=generated_ids,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
    )
