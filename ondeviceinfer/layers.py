"""Foundational Qwen3 layers implemented with basic PyTorch operations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by Qwen3.

    Qwen3 has no mean subtraction and no bias.  The reduction is performed in
    float32 to avoid excessive rounding error when activations use FP16/BF16.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        states_fp32 = hidden_states.float()
        variance = states_fp32.square().mean(dim=-1, keepdim=True)
        normalized = states_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Construct Qwen3's RoPE cosine and sine tables for requested positions."""

    def __init__(self, head_dim: int, theta: float = 1_000_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")

        # inv_freq is model state, but not a trainable parameter.
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        inv_freq = 1.0 / (theta**exponent)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return cos/sin shaped ``[batch, seq_len, head_dim]``.

        ``x`` is used only to select the output device and dtype.  Computing
        the angles in float32 keeps long-position phase calculations stable.
        """
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, seq_len]")

        positions = position_ids.to(device=x.device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=x.device)
        frequencies = positions.unsqueeze(-1) * inv_freq
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Map ``[x1, x2]`` to ``[-x2, x1]`` along the head dimension."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE to Q and K tensors shaped ``[batch, heads, seq, dim]``."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_rotated = query * cos + rotate_half(query) * sin
    key_rotated = key * cos + rotate_half(key) * sin
    return query_rotated, key_rotated


class SwiGLU(nn.Module):
    """Qwen3 feed-forward block: down(silu(gate(x)) * up(x))."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, **factory_kwargs
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, **factory_kwargs
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, **factory_kwargs
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )
