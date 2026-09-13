"""Pinned CPU weights with synchronous or double-buffered CUDA offloading."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torch.profiler import record_function

from ..config import Qwen3Config
from ..weights import SafetensorIndex


LAYER_TENSOR_NAMES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def layer_bytes(layer: dict[str, Tensor]) -> int:
    return sum(tensor_bytes(tensor) for tensor in layer.values())


@dataclass
class CpuWeightStore:
    config: Qwen3Config
    layers: list[dict[str, Tensor]]
    embed_tokens: Tensor
    final_norm: Tensor
    lm_head: Tensor
    dtype: torch.dtype

    @property
    def decoder_bytes(self) -> int:
        return sum(layer_bytes(layer) for layer in self.layers)

    @property
    def total_bytes(self) -> int:
        return (
            self.decoder_bytes
            + tensor_bytes(self.embed_tokens)
            + tensor_bytes(self.final_norm)
            + tensor_bytes(self.lm_head)
        )

    @property
    def all_pinned(self) -> bool:
        tensors = [self.embed_tokens, self.final_norm, self.lm_head]
        tensors.extend(t for layer in self.layers for t in layer.values())
        return all(tensor.is_pinned() for tensor in tensors)


def _convert_cpu_tensor(
    tensor: Tensor, *, dtype: torch.dtype, pin_memory: bool
) -> Tensor:
    tensor = tensor.to(dtype=dtype).contiguous()
    return tensor.pin_memory() if pin_memory else tensor


def load_cpu_weight_store(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.float16,
    pin_memory: bool = True,
    progress: bool = True,
) -> CpuWeightStore:
    """Load Qwen3 weights once into immutable CPU (optionally pinned) tensors."""
    model_path = Path(model_path)
    config = Qwen3Config.from_json(model_path / "config.json")
    index = SafetensorIndex(model_path)
    layers: list[dict[str, Tensor]] = []
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}."
        weights = {
            suffix: _convert_cpu_tensor(
                index.get_tensor(prefix + suffix), dtype=dtype, pin_memory=pin_memory
            )
            for suffix in LAYER_TENSOR_NAMES
        }
        layers.append(weights)
        if progress:
            print(
                f"loaded CPU layer {layer_idx + 1}/{config.num_hidden_layers} "
                f"({layer_bytes(weights) / 2**20:.2f} MiB)",
                flush=True,
            )

    return CpuWeightStore(
        config=config,
        layers=layers,
        embed_tokens=_convert_cpu_tensor(
            index.get_tensor("model.embed_tokens.weight"),
            dtype=dtype,
            pin_memory=pin_memory,
        ),
        final_norm=_convert_cpu_tensor(
            index.get_tensor("model.norm.weight"),
            dtype=dtype,
            pin_memory=pin_memory,
        ),
        lm_head=_convert_cpu_tensor(
            index.get_tensor("lm_head.weight"),
            dtype=dtype,
            pin_memory=pin_memory,
        ),
        dtype=dtype,
    )


@dataclass
class RuntimeStats:
    startup_h2d_bytes: int = 0
    weight_h2d_bytes: int = 0
    synchronous_copy_seconds: float = 0.0
    async_copy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )

    def reset_forward_counters(self) -> None:
        self.weight_h2d_bytes = 0
        self.synchronous_copy_seconds = 0.0
        self.async_copy_events.clear()

    def async_copy_seconds(self) -> float:
        if not self.async_copy_events:
            return 0.0
        torch.cuda.synchronize()
        return sum(start.elapsed_time(end) for start, end in self.async_copy_events) / 1000


def allocate_layer_like(layer: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        name: torch.empty_like(tensor, device=device)
        for name, tensor in layer.items()
    }


def copy_layer(
    destination: dict[str, Tensor],
    source: dict[str, Tensor],
    *,
    non_blocking: bool,
) -> None:
    for name in LAYER_TENSOR_NAMES:
        destination[name].copy_(source[name], non_blocking=non_blocking)


class SyncOffloadRuntime:
    """Static prefix plus one synchronously refilled layer slot."""

    def __init__(
        self,
        store: CpuWeightStore,
        *,
        device: torch.device | str,
        gpu_layer_budget: int,
    ) -> None:
        if not 1 <= gpu_layer_budget <= store.config.num_hidden_layers:
            raise ValueError("gpu_layer_budget must be between 1 and num_hidden_layers")
        self.store = store
        self.device = torch.device(device)
        self.gpu_layer_budget = gpu_layer_budget
        self.resident_count = min(
            store.config.num_hidden_layers, max(0, gpu_layer_budget - 1)
        )
        self.stats = RuntimeStats()
        self.resident = [
            {name: tensor.to(self.device) for name, tensor in store.layers[i].items()}
            for i in range(self.resident_count)
        ]
        self.stats.startup_h2d_bytes = sum(layer_bytes(layer) for layer in self.resident)
        self.slot = (
            allocate_layer_like(store.layers[self.resident_count], self.device)
            if self.resident_count < store.config.num_hidden_layers
            else None
        )

    def begin_forward(self) -> None:
        pass

    def weights_for(self, layer_idx: int) -> dict[str, Tensor]:
        if layer_idx < self.resident_count:
            return self.resident[layer_idx]
        if self.slot is None:
            raise RuntimeError("No dynamic layer slot allocated")
        start = time.perf_counter()
        copy_layer(self.slot, self.store.layers[layer_idx], non_blocking=False)
        self.stats.synchronous_copy_seconds += time.perf_counter() - start
        self.stats.weight_h2d_bytes += layer_bytes(self.store.layers[layer_idx])
        return self.slot

    def after_layer(self, layer_idx: int) -> None:
        pass

    def end_forward(self) -> None:
        pass


@dataclass
class _AsyncSlot:
    weights: dict[str, Tensor]
    ready: torch.cuda.Event
    compute_done: torch.cuda.Event
    layer_idx: int | None = None
    has_compute_event: bool = False


class PrefetchOffloadRuntime:
    """Static prefix plus two event-protected asynchronous layer slots."""

    def __init__(
        self,
        store: CpuWeightStore,
        *,
        device: torch.device | str,
        gpu_layer_budget: int,
    ) -> None:
        if gpu_layer_budget < 2:
            raise ValueError("Prefetch requires gpu_layer_budget >= 2")
        if gpu_layer_budget > store.config.num_hidden_layers:
            raise ValueError("gpu_layer_budget cannot exceed num_hidden_layers")
        self.store = store
        self.device = torch.device(device)
        self.gpu_layer_budget = gpu_layer_budget
        self.resident_count = min(
            store.config.num_hidden_layers, max(0, gpu_layer_budget - 2)
        )
        self.stats = RuntimeStats()
        self.resident = [
            {name: tensor.to(self.device) for name, tensor in store.layers[i].items()}
            for i in range(self.resident_count)
        ]
        self.stats.startup_h2d_bytes = sum(layer_bytes(layer) for layer in self.resident)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        template_idx = min(self.resident_count, store.config.num_hidden_layers - 1)
        self.slots = [
            _AsyncSlot(
                weights=allocate_layer_like(store.layers[template_idx], self.device),
                ready=torch.cuda.Event(enable_timing=False),
                compute_done=torch.cuda.Event(enable_timing=False),
            )
            for _ in range(2)
        ]
        self.first_dynamic = self.resident_count

    def _slot_for(self, layer_idx: int) -> _AsyncSlot:
        return self.slots[(layer_idx - self.first_dynamic) % 2]

    def _schedule(self, layer_idx: int) -> None:
        if layer_idx >= self.store.config.num_hidden_layers:
            return
        slot = self._slot_for(layer_idx)
        with torch.cuda.stream(self.copy_stream):
            if slot.has_compute_event:
                self.copy_stream.wait_event(slot.compute_done)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self.copy_stream)
            with record_function(f"prefetch_layer_{layer_idx}"):
                copy_layer(
                    slot.weights,
                    self.store.layers[layer_idx],
                    non_blocking=True,
                )
            end.record(self.copy_stream)
            slot.ready.record(self.copy_stream)
        slot.layer_idx = layer_idx
        self.stats.async_copy_events.append((start, end))
        self.stats.weight_h2d_bytes += layer_bytes(self.store.layers[layer_idx])

    def begin_forward(self) -> None:
        if self.first_dynamic < self.store.config.num_hidden_layers:
            self._schedule(self.first_dynamic)

    def weights_for(self, layer_idx: int) -> dict[str, Tensor]:
        if layer_idx < self.resident_count:
            return self.resident[layer_idx]
        slot = self._slot_for(layer_idx)
        if slot.layer_idx != layer_idx:
            raise RuntimeError(
                f"Layer {layer_idx} was not prefetched (slot has {slot.layer_idx})"
            )
        torch.cuda.current_stream(self.device).wait_event(slot.ready)
        return slot.weights

    def after_layer(self, layer_idx: int) -> None:
        if layer_idx >= self.resident_count:
            slot = self._slot_for(layer_idx)
            slot.compute_done.record(torch.cuda.current_stream(self.device))
            slot.has_compute_event = True
            self._schedule(layer_idx + 1)

    def end_forward(self) -> None:
        pass


def estimate_gpu_weight_bytes(
    store: CpuWeightStore, gpu_layer_budget: int
) -> int:
    """Estimate decoder slots plus the resident final norm and LM head."""
    if not 1 <= gpu_layer_budget <= store.config.num_hidden_layers:
        raise ValueError("invalid gpu_layer_budget")
    representative = max(layer_bytes(layer) for layer in store.layers)
    return (
        representative * gpu_layer_budget
        + tensor_bytes(store.final_norm)
        + tensor_bytes(store.lm_head)
    )
