"""Qwen3 execution over per-output-channel INT8 decoder weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.profiler import record_function

from .attention import build_causal_mask
from .cache import KVCache, ResidentKVCache
from .functional import rms_norm
from .layers import RotaryEmbedding, apply_rotary_pos_emb
from .model import CausalLMOutput
from .profiling import nvtx_range
from .runtime.quantized import (
    LINEAR_WEIGHT_NAMES,
    QuantizedCpuWeightStore,
    qweight_name,
    scale_name,
)


@dataclass
class QuantizedComputeStats:
    """Optional CUDA-event timing for dequantization and FP16 GEMM."""

    enabled: bool = False
    dequant_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )
    gemm_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )
    dequantized_weight_bytes: int = 0
    linear_calls: int = 0

    def reset(self) -> None:
        self.dequant_events.clear()
        self.gemm_events.clear()
        self.dequantized_weight_bytes = 0
        self.linear_calls = 0

    @staticmethod
    def _seconds(
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> float:
        if not events:
            return 0.0
        torch.cuda.synchronize()
        return sum(start.elapsed_time(end) for start, end in events) / 1000.0

    def dequant_seconds(self) -> float:
        return self._seconds(self.dequant_events)

    def gemm_seconds(self) -> float:
        return self._seconds(self.gemm_events)


def _recorded_cuda_call(
    function: object,
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> Tensor:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = function()  # type: ignore[operator]
    end.record()
    events.append((start, end))
    return result


def int8_weight_only_linear(
    inputs: Tensor,
    weights: Mapping[str, Tensor],
    weight_name: str,
    scratch: Tensor,
    *,
    stats: QuantizedComputeStats | None = None,
) -> Tensor:
    """Dequantize one matrix into a reused FP16 scratch view, then GEMM.

    All operations are queued on the current compute stream.  Reusing the same
    scratch for consecutive matrices is safe because CUDA stream order keeps
    the GEMM read ahead of the next overwrite.
    """
    if weight_name not in LINEAR_WEIGHT_NAMES:
        raise KeyError(f"Unsupported quantized linear tensor {weight_name!r}")
    qweight = weights[qweight_name(weight_name)]
    scale = weights[scale_name(weight_name)]
    if qweight.device != inputs.device or scale.device != inputs.device:
        raise ValueError("INT8 weights, scales, and activations must share a device")
    if qweight.numel() > scratch.numel():
        raise ValueError(
            f"Scratch has {scratch.numel()} elements, but {weight_name} needs "
            f"{qweight.numel()}"
        )
    dequantized = scratch[: qweight.numel()].view(qweight.shape)

    def dequantize() -> Tensor:
        dequantized.copy_(qweight)
        dequantized.mul_(scale.unsqueeze(1))
        return dequantized

    if stats is not None:
        stats.linear_calls += 1
        stats.dequantized_weight_bytes += dequantized.numel() * dequantized.element_size()
    with nvtx_range(f"int8_dequant::{weight_name}"):
        if stats is not None and stats.enabled and inputs.device.type == "cuda":
            _recorded_cuda_call(dequantize, stats.dequant_events)
        else:
            dequantize()
    with nvtx_range(f"fp16_gemm::{weight_name}"):
        if stats is not None and stats.enabled and inputs.device.type == "cuda":
            return _recorded_cuda_call(
                lambda: F.linear(inputs, dequantized), stats.gemm_events
            )
        return F.linear(inputs, dequantized)


def quantized_decoder_layer_forward(
    hidden_states: Tensor,
    weights: Mapping[str, Tensor],
    config: object,
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    scratch: Tensor,
    *,
    cache: KVCache | None = None,
    stats: QuantizedComputeStats | None = None,
) -> Tensor:
    """Execute one Qwen3 decoder layer using INT8 weight-only linears."""
    # Config is kept structural here so tiny synthetic configurations can use
    # the same function without importing a model implementation.
    residual = hidden_states
    states = rms_norm(
        hidden_states,
        weights["input_layernorm.weight"],
        config.rms_norm_eps,  # type: ignore[attr-defined]
    )
    batch_size, query_length, _ = states.shape

    query = int8_weight_only_linear(
        states, weights, "self_attn.q_proj.weight", scratch, stats=stats
    ).view(
        batch_size,
        query_length,
        config.num_attention_heads,  # type: ignore[attr-defined]
        config.head_dim,  # type: ignore[attr-defined]
    )
    key = int8_weight_only_linear(
        states, weights, "self_attn.k_proj.weight", scratch, stats=stats
    ).view(
        batch_size,
        query_length,
        config.num_key_value_heads,  # type: ignore[attr-defined]
        config.head_dim,  # type: ignore[attr-defined]
    )
    value = int8_weight_only_linear(
        states, weights, "self_attn.v_proj.weight", scratch, stats=stats
    ).view(
        batch_size,
        query_length,
        config.num_key_value_heads,  # type: ignore[attr-defined]
        config.head_dim,  # type: ignore[attr-defined]
    )
    query = rms_norm(
        query, weights["self_attn.q_norm.weight"], config.rms_norm_eps  # type: ignore[attr-defined]
    ).transpose(1, 2)
    key = rms_norm(
        key, weights["self_attn.k_norm.weight"], config.rms_norm_eps  # type: ignore[attr-defined]
    ).transpose(1, 2)
    value = value.transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if cache is not None:
        key, value = cache.update(layer_idx, key, value)
    key_length = key.shape[-2]
    attention_mask = None
    is_causal = query_length > 1 and key_length == query_length
    if query_length > 1 and key_length != query_length:
        attention_mask = build_causal_mask(
            query_length, key_length, device=query.device
        )

    with nvtx_range(f"kv_attention::layer_{layer_idx}"):
        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=True,
        )
    attention_output = attention_output.transpose(1, 2).contiguous().view(
        batch_size,
        query_length,
        config.num_attention_heads * config.head_dim,  # type: ignore[attr-defined]
    )
    attention_output = int8_weight_only_linear(
        attention_output,
        weights,
        "self_attn.o_proj.weight",
        scratch,
        stats=stats,
    )
    hidden_states = residual + attention_output

    residual = hidden_states
    states = rms_norm(
        hidden_states,
        weights["post_attention_layernorm.weight"],
        config.rms_norm_eps,  # type: ignore[attr-defined]
    )
    gate = int8_weight_only_linear(
        states, weights, "mlp.gate_proj.weight", scratch, stats=stats
    )
    up = int8_weight_only_linear(
        states, weights, "mlp.up_proj.weight", scratch, stats=stats
    )
    mlp_output = int8_weight_only_linear(
        F.silu(gate) * up,
        weights,
        "mlp.down_proj.weight",
        scratch,
        stats=stats,
    )
    return residual + mlp_output


class QuantizedQwen3ForCausalLM:
    """Qwen3 model whose decoder is INT8 and supplied by a placement runtime."""

    def __init__(
        self,
        store: QuantizedCpuWeightStore,
        runtime: object,
        *,
        device: str | torch.device,
        profile_compute: bool = False,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.config = store.config
        self.device = torch.device(device)
        self.dtype = store.compute_dtype
        self.final_norm = store.final_norm.to(self.device)
        self.lm_head = store.lm_head.to(self.device)
        self.rotary_emb = RotaryEmbedding(
            self.config.head_dim, self.config.rope_theta
        ).to(self.device)
        self.dequant_scratch = torch.empty(
            store.max_dequant_elements, device=self.device, dtype=self.dtype
        )
        self.compute_stats = QuantizedComputeStats(enabled=profile_compute)

    def make_resident_cache(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        device: str | torch.device | None = None,
    ) -> ResidentKVCache:
        return ResidentKVCache(
            num_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            num_key_value_heads=self.config.num_key_value_heads,
            max_seq_len=max_seq_len,
            head_dim=self.config.head_dim,
            device=device or self.device,
            dtype=self.dtype,
        )

    @torch.inference_mode()
    def __call__(
        self,
        input_ids: Tensor,
        *,
        cache: KVCache | None = None,
        position_ids: Tensor | None = None,
        logits_to_keep: int = 0,
        **_: object,
    ) -> CausalLMOutput:
        ids_cpu = input_ids.detach().to(device="cpu")
        hidden_cpu = F.embedding(ids_cpu, self.store.embed_tokens)
        hidden_states = hidden_cpu.to(self.device)

        past_length = cache.get_seq_length(0) if cache is not None else 0
        query_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + query_length,
                device=self.device,
            ).expand(input_ids.shape[0], -1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        if cache is not None and hasattr(cache, "begin_forward"):
            cache.begin_forward(query_length=query_length)  # type: ignore[attr-defined]
        self.runtime.begin_forward()  # type: ignore[attr-defined]
        for layer_idx in range(self.config.num_hidden_layers):
            if cache is not None and hasattr(cache, "before_layer"):
                cache.before_layer(layer_idx)  # type: ignore[attr-defined]
            with nvtx_range(f"decoder_layer::{layer_idx}"):
                with record_function(f"quantized_decoder_layer_{layer_idx}"):
                    weights = self.runtime.weights_for(layer_idx)  # type: ignore[attr-defined]
                    hidden_states = quantized_decoder_layer_forward(
                        hidden_states,
                        weights,
                        self.config,
                        layer_idx,
                        cos,
                        sin,
                        self.dequant_scratch,
                        cache=cache,
                        stats=self.compute_stats,
                    )
            if cache is not None and hasattr(cache, "after_layer"):
                cache.after_layer(layer_idx)  # type: ignore[attr-defined]
            self.runtime.after_layer(layer_idx)  # type: ignore[attr-defined]
        self.runtime.end_forward()  # type: ignore[attr-defined]
        if cache is not None and hasattr(cache, "end_forward"):
            cache.end_forward()  # type: ignore[attr-defined]

        hidden_states = rms_norm(
            hidden_states, self.final_norm, self.config.rms_norm_eps
        )
        if logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = F.linear(hidden_states, self.lm_head)
        return CausalLMOutput(logits=logits, cache=cache)
