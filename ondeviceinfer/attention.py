"""Qwen3 grouped-query self-attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cache import KVCache
from .config import Qwen3Config
from .layers import RMSNorm, apply_rotary_pos_emb


def build_causal_mask(
    query_length: int,
    key_length: int,
    *,
    device: torch.device,
) -> Tensor:
    """Build an absolute-position boolean mask for cached attention.

    PyTorch SDPA interprets a boolean ``True`` as "participates in
    attention".  The final query tokens correspond to the final
    ``query_length`` absolute positions in the key sequence.
    """
    if query_length > key_length:
        raise ValueError("query_length cannot exceed key_length")
    past_length = key_length - query_length
    query_positions = torch.arange(
        past_length, key_length, device=device
    ).unsqueeze(-1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    return key_positions <= query_positions


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout
        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        # Qwen3 normalizes each attention head after projection.
        self.q_norm = RMSNorm(
            config.head_dim, config.rms_norm_eps, **factory_kwargs
        )
        self.k_norm = RMSNorm(
            config.head_dim, config.rms_norm_eps, **factory_kwargs
        )

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        cache: KVCache | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, query_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_attention_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )

        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if cache is not None:
            key, value = cache.update(self.layer_idx, key, value)

        key_length = key.shape[-2]
        if attention_mask is None and query_length > 1:
            # An explicit absolute-position mask is also correct for non-square
            # cached attention.  For uncached prefill, is_causal=True lets SDPA
            # select its fastest causal kernel.
            if key_length == query_length:
                is_causal = True
            else:
                attention_mask = build_causal_mask(
                    query_length, key_length, device=query.device
                )
                is_causal = False
        else:
            # A one-token decode query is allowed to attend to all cached keys.
            is_causal = False

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=True,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, query_length, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attention_output)
