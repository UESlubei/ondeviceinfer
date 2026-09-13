"""Safetensors checkpoint indexing and strict model loading."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .config import Qwen3Config
from .model import Qwen3ForCausalLM


class SafetensorIndex:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map: dict[str, str] = raw["weight_map"]
            self.total_size = int(raw.get("metadata", {}).get("total_size", 0))
        else:
            files = sorted(self.model_path.glob("*.safetensors"))
            if len(files) != 1:
                raise FileNotFoundError(f"No safetensors index found in {self.model_path}")
            state = load_file(files[0], device="cpu")
            self.weight_map = {name: files[0].name for name in state}
            self.total_size = sum(t.numel() * t.element_size() for t in state.values())

    @property
    def shard_names(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    def load_state_dict(self, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for shard_name in self.shard_names:
            shard = load_file(self.model_path / shard_name, device="cpu")
            for name, tensor in shard.items():
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                state[name] = tensor
        return state

    def names_for_prefix(self, prefix: str) -> list[str]:
        return sorted(name for name in self.weight_map if name.startswith(prefix))

    def get_tensor(self, name: str) -> torch.Tensor:
        try:
            shard_name = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"Tensor {name!r} is not present in the checkpoint") from error
        with safe_open(
            self.model_path / shard_name, framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(name)


def load_resident_model(
    model_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3ForCausalLM:
    """Strictly load a Qwen3 checkpoint into the custom model."""
    model_path = Path(model_path)
    config = Qwen3Config.from_json(model_path / "config.json")
    model = Qwen3ForCausalLM(config, device="meta", dtype=dtype)
    index = SafetensorIndex(model_path)
    state = index.load_state_dict(dtype=dtype)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    del state

    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model = model.to(device=device)
    model.eval().requires_grad_(False)
    return model
