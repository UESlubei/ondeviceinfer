"""Stateless Qwen3 decoder math for resident/offloaded weight buffers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .attention import cached_scaled_dot_product_attention
from .cache import KVCache
from .config import Qwen3Config
from .layers import apply_rotary_pos_emb


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
    input_dtype = hidden_states.dtype
    states_fp32 = hidden_states.float()
    variance = states_fp32.square().mean(dim=-1, keepdim=True)
    normalized = states_fp32 * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def decoder_layer_forward(
    hidden_states: Tensor,
    weights: Mapping[str, Tensor],
    config: Qwen3Config,
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    *,
    cache: KVCache | None = None,
) -> Tensor:
    """Execute one dense Qwen3 decoder layer from explicit weight tensors."""
    residual = hidden_states
    states = rms_norm(
        hidden_states, weights["input_layernorm.weight"], config.rms_norm_eps
    )
    batch_size, query_length, _ = states.shape

    query = F.linear(states, weights["self_attn.q_proj.weight"]).view(
        batch_size, query_length, config.num_attention_heads, config.head_dim
    )
    key = F.linear(states, weights["self_attn.k_proj.weight"]).view(
        batch_size, query_length, config.num_key_value_heads, config.head_dim
    )
    value = F.linear(states, weights["self_attn.v_proj.weight"]).view(
        batch_size, query_length, config.num_key_value_heads, config.head_dim
    )
    query = rms_norm(
        query, weights["self_attn.q_norm.weight"], config.rms_norm_eps
    ).transpose(1, 2)
    key = rms_norm(
        key, weights["self_attn.k_norm.weight"], config.rms_norm_eps
    ).transpose(1, 2)
    value = value.transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    attention_output = cached_scaled_dot_product_attention(
        query,
        key,
        value,
        layer_idx=layer_idx,
        cache=cache,
        attention_mask=None,
        dropout_p=0.0,
    )
    attention_output = attention_output.transpose(1, 2).contiguous().view(
        batch_size,
        query_length,
        config.num_attention_heads * config.head_dim,
    )
    attention_output = F.linear(
        attention_output, weights["self_attn.o_proj.weight"]
    )
    hidden_states = residual + attention_output

    residual = hidden_states
    states = rms_norm(
        hidden_states,
        weights["post_attention_layernorm.weight"],
        config.rms_norm_eps,
    )
    gate = F.linear(states, weights["mlp.gate_proj.weight"])
    up = F.linear(states, weights["mlp.up_proj.weight"])
    mlp_output = F.linear(
        F.silu(gate) * up, weights["mlp.down_proj.weight"]
    )
    return residual + mlp_output
