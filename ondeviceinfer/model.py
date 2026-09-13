"""Dense Qwen3 model assembled only from basic PyTorch components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .attention import Qwen3Attention
from .cache import KVCache, ResidentKVCache
from .config import Qwen3Config
from .layers import RMSNorm, RotaryEmbedding, SwiGLU


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = Qwen3Attention(
            config, layer_idx, **factory_kwargs
        )
        self.mlp = SwiGLU(
            config.hidden_size, config.intermediate_size, **factory_kwargs
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, **factory_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, **factory_kwargs
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos,
            sin,
            cache=cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            **factory_kwargs,
        )
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, layer_idx, **factory_kwargs)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, **factory_kwargs
        )
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: KVCache | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        past_length = cache.get_seq_length(0) if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + input_ids.shape[1],
                device=input_ids.device,
            ).expand(input_ids.shape[0], -1)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos,
                sin,
                cache=cache,
                attention_mask=attention_mask,
            )
        return self.norm(hidden_states)


@dataclass
class CausalLMOutput:
    logits: Tensor
    cache: KVCache | None = None


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: KVCache | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> CausalLMOutput:
        hidden_states = self.model(
            input_ids,
            cache=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        if logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        return CausalLMOutput(logits=self.lm_head(hidden_states), cache=cache)

    def make_resident_cache(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> ResidentKVCache:
        parameter = next(self.parameters())
        return ResidentKVCache(
            num_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            num_key_value_heads=self.config.num_key_value_heads,
            max_seq_len=max_seq_len,
            head_dim=self.config.head_dim,
            device=device or parameter.device,
            dtype=parameter.dtype,
        )
