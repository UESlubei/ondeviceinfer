"""Dense Qwen3 model whose decoder weights are supplied by an offload runtime."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.profiler import record_function

from .cache import KVCache, ResidentKVCache
from .functional import decoder_layer_forward, rms_norm
from .layers import RotaryEmbedding
from .model import CausalLMOutput
from .runtime.offload import CpuWeightStore


class OffloadedQwen3ForCausalLM:
    def __init__(self, store: CpuWeightStore, runtime: object, *, device: str | torch.device) -> None:
        self.store = store
        self.runtime = runtime
        self.config = store.config
        self.device = torch.device(device)
        self.dtype = store.dtype
        self.final_norm = store.final_norm.to(self.device)
        self.lm_head = store.lm_head.to(self.device)
        self.rotary_emb = RotaryEmbedding(
            self.config.head_dim, self.config.rope_theta
        ).to(self.device)

    def make_resident_cache(
        self, *, batch_size: int, max_seq_len: int, device: str | torch.device | None = None
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
        # The activation is tiny compared with model weights; a synchronous
        # copy gives it an unambiguous lifetime before decoder execution.
        hidden_states = hidden_cpu.to(self.device)

        past_length = cache.get_seq_length(0) if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + input_ids.shape[1],
                device=self.device,
            ).expand(input_ids.shape[0], -1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        self.runtime.begin_forward()
        for layer_idx in range(self.config.num_hidden_layers):
            with record_function(f"decoder_layer_{layer_idx}"):
                weights = self.runtime.weights_for(layer_idx)
                hidden_states = decoder_layer_forward(
                    hidden_states,
                    weights,
                    self.config,
                    layer_idx,
                    cos,
                    sin,
                    cache=cache,
                )
            self.runtime.after_layer(layer_idx)
        self.runtime.end_forward()

        hidden_states = rms_norm(
            hidden_states, self.final_norm, self.config.rms_norm_eps
        )
        if logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = F.linear(hidden_states, self.lm_head)
        return CausalLMOutput(logits=logits, cache=cache)
