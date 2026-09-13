"""KV-cache implementations shared by resident and offloaded runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class KVCache(Protocol):
    """Minimal interface consumed by the attention implementation."""

    def get_seq_length(self, layer_idx: int = 0) -> int: ...

    def update(
        self, layer_idx: int, key_states: Tensor, value_states: Tensor
    ) -> tuple[Tensor, Tensor]: ...

    def reset(self) -> None: ...


@dataclass
class CacheStats:
    cpu_to_gpu_bytes: int = 0
    gpu_to_cpu_bytes: int = 0


class ResidentKVCache:
    """Preallocated KV cache that remains on one device.

    Storage layout is ``[batch, kv_heads, max_seq_len, head_dim]``.  Each
    decoder layer owns an independent logical length because layers update the
    cache one after another during a single model forward.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_key_value_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        shape = (batch_size, num_key_value_heads, max_seq_len, head_dim)
        self.key_cache = [torch.empty(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.value_cache = [torch.empty(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.lengths = [0] * num_layers
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.stats = CacheStats()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.lengths[layer_idx]

    def update(
        self, layer_idx: int, key_states: Tensor, value_states: Tensor
    ) -> tuple[Tensor, Tensor]:
        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have the same shape")
        if key_states.shape[0] != self.batch_size:
            raise ValueError("KV cache batch size cannot change after allocation")

        start = self.lengths[layer_idx]
        end = start + key_states.shape[-2]
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache capacity exceeded: requested {end}, capacity {self.max_seq_len}"
            )

        self.key_cache[layer_idx][:, :, start:end].copy_(key_states)
        self.value_cache[layer_idx][:, :, start:end].copy_(value_states)
        self.lengths[layer_idx] = end
        return (
            self.key_cache[layer_idx][:, :, :end],
            self.value_cache[layer_idx][:, :, :end],
        )

    def reset(self) -> None:
        self.lengths[:] = [0] * len(self.lengths)
        self.stats = CacheStats()

    @property
    def storage_bytes(self) -> int:
        tensors = self.key_cache + self.value_cache
        return sum(t.numel() * t.element_size() for t in tensors)
