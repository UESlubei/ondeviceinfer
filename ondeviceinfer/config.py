"""Configuration used by our Qwen3 implementation.

This module deliberately parses ``config.json`` itself instead of importing a
Transformers model class.  Keeping the configuration small also makes every
architectural assumption in our implementation visible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3Config":
        """Load and validate the architecture fields needed by the runtime."""
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        if raw.get("model_type") != "qwen3":
            raise ValueError(
                f"Expected a Qwen3 config, got model_type={raw.get('model_type')!r}"
            )

        config = cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get(
                "head_dim", raw["hidden_size"] // raw["num_attention_heads"]
            ),
            max_position_embeddings=raw["max_position_embeddings"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            tie_word_embeddings=raw["tie_word_embeddings"],
            attention_bias=raw.get("attention_bias", False),
            attention_dropout=raw.get("attention_dropout", 0.0),
            bos_token_id=raw.get("bos_token_id", 151643),
            eos_token_id=raw.get("eos_token_id", 151645),
            pad_token_id=raw.get("pad_token_id"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("Q heads must be divisible by KV heads for GQA")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")

    @property
    def num_key_value_groups(self) -> int:
        """Number of query heads sharing one key/value head."""
        return self.num_attention_heads // self.num_key_value_heads
