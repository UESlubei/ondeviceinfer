"""INT8 weight-only checkpoint loading and GPU layer staging runtimes.

Decoder matrix weights are stored as signed INT8 values plus one FP16 scale
per output channel.  Norm weights and the global embedding/final norm/LM head
remain FP16.  The runtime only owns placement and transfer; dequantization is
performed by :mod:`ondeviceinfer.quantized_model` immediately before GEMM.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor
from torch.profiler import record_function

from ..config import Qwen3Config
from ..profiling import nvtx_range
from .offload import RuntimeStats, tensor_bytes


LINEAR_WEIGHT_NAMES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

NORM_WEIGHT_NAMES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)


def qweight_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected a .weight suffix, got {weight_name!r}")
    return weight_name.removesuffix(".weight") + ".qweight"


def scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected a .weight suffix, got {weight_name!r}")
    return weight_name.removesuffix(".weight") + ".scale"


def quantized_layer_bytes(layer: dict[str, Tensor]) -> int:
    return sum(tensor_bytes(tensor) for tensor in layer.values())


@dataclass
class QuantizedCpuWeightStore:
    config: Qwen3Config
    layers: list[dict[str, Tensor]]
    embed_tokens: Tensor
    final_norm: Tensor
    lm_head: Tensor
    compute_dtype: torch.dtype
    manifest: dict[str, object]

    @property
    def decoder_bytes(self) -> int:
        return sum(quantized_layer_bytes(layer) for layer in self.layers)

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
        tensors.extend(tensor for layer in self.layers for tensor in layer.values())
        return all(tensor.is_pinned() for tensor in tensors)

    @property
    def max_dequant_elements(self) -> int:
        return max(
            layer[qweight_name(name)].numel()
            for layer in self.layers
            for name in LINEAR_WEIGHT_NAMES
        )


def _prepare_cpu_tensor(tensor: Tensor, *, pin_memory: bool) -> Tensor:
    tensor = tensor.contiguous()
    return tensor.pin_memory() if pin_memory else tensor


def _validate_layer(
    layer: dict[str, Tensor], layer_idx: int, config: Qwen3Config
) -> None:
    expected = set(NORM_WEIGHT_NAMES)
    for name in LINEAR_WEIGHT_NAMES:
        expected.add(qweight_name(name))
        expected.add(scale_name(name))
    missing = sorted(expected.difference(layer))
    unexpected = sorted(set(layer).difference(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"Invalid quantized layer {layer_idx}: missing={missing}, "
            f"unexpected={unexpected}"
        )
    for name in NORM_WEIGHT_NAMES:
        if layer[name].dtype != torch.float16:
            raise TypeError(f"{name} must be FP16, got {layer[name].dtype}")
    expected_norm_shapes = {
        "input_layernorm.weight": (config.hidden_size,),
        "post_attention_layernorm.weight": (config.hidden_size,),
        "self_attn.q_norm.weight": (config.head_dim,),
        "self_attn.k_norm.weight": (config.head_dim,),
    }
    for name, shape in expected_norm_shapes.items():
        if tuple(layer[name].shape) != shape:
            raise ValueError(
                f"Invalid shape for layer {layer_idx} {name}: "
                f"expected {shape}, got {tuple(layer[name].shape)}"
            )
    kv_width = config.num_key_value_heads * config.head_dim
    expected_linear_shapes = {
        "self_attn.q_proj.weight": (config.hidden_size, config.hidden_size),
        "self_attn.k_proj.weight": (kv_width, config.hidden_size),
        "self_attn.v_proj.weight": (kv_width, config.hidden_size),
        "self_attn.o_proj.weight": (config.hidden_size, config.hidden_size),
        "mlp.gate_proj.weight": (config.intermediate_size, config.hidden_size),
        "mlp.up_proj.weight": (config.intermediate_size, config.hidden_size),
        "mlp.down_proj.weight": (config.hidden_size, config.intermediate_size),
    }
    for name in LINEAR_WEIGHT_NAMES:
        qweight = layer[qweight_name(name)]
        scale = layer[scale_name(name)]
        if qweight.dtype != torch.int8:
            raise TypeError(f"{qweight_name(name)} must be INT8")
        if scale.dtype != torch.float16:
            raise TypeError(f"{scale_name(name)} must be FP16")
        if qweight.ndim != 2 or scale.shape != (qweight.shape[0],):
            raise ValueError(
                f"Invalid qweight/scale shapes for {name}: "
                f"{tuple(qweight.shape)} and {tuple(scale.shape)}"
            )
        if not bool(torch.all(torch.isfinite(scale))):
            raise ValueError(f"Non-finite scale in layer {layer_idx} {name}")
        if not bool(torch.all(scale > 0)):
            raise ValueError(f"Non-positive scale in layer {layer_idx} {name}")
        if tuple(qweight.shape) != expected_linear_shapes[name]:
            raise ValueError(
                f"Invalid shape for layer {layer_idx} {name}: expected "
                f"{expected_linear_shapes[name]}, got {tuple(qweight.shape)}"
            )


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_quantized_cpu_weight_store(
    checkpoint_path: str | Path,
    *,
    pin_memory: bool = True,
    progress: bool = True,
    verify_checksums: bool = False,
) -> QuantizedCpuWeightStore:
    """Load the repository's per-layer INT8 checkpoint into immutable CPU RAM."""
    checkpoint_path = Path(checkpoint_path)
    config = Qwen3Config.from_json(checkpoint_path / "config.json")
    manifest_path = checkpoint_path / "quantization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "qwen3-int8-weight-only-v1":
        raise ValueError(f"Unsupported quantized format in {manifest_path}")
    if verify_checksums:
        for entry in manifest.get("tensor_files", []):
            tensor_path = checkpoint_path / entry["name"]
            if tensor_path.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"Size mismatch for {tensor_path}")
            if _sha256_file(tensor_path) != entry["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch for {tensor_path}")

    layers: list[dict[str, Tensor]] = []
    for layer_idx in range(config.num_hidden_layers):
        path = checkpoint_path / f"layer-{layer_idx:03d}.safetensors"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = load_file(path, device="cpu")
        layer = {
            name: _prepare_cpu_tensor(tensor, pin_memory=pin_memory)
            for name, tensor in raw.items()
        }
        _validate_layer(layer, layer_idx, config)
        layers.append(layer)
        if progress:
            print(
                f"loaded INT8 CPU layer {layer_idx + 1}/{config.num_hidden_layers} "
                f"({quantized_layer_bytes(layer) / 2**20:.2f} MiB)",
                flush=True,
            )

    globals_path = checkpoint_path / "global.safetensors"
    global_weights = load_file(globals_path, device="cpu")
    expected_globals = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    if set(global_weights) != expected_globals:
        raise RuntimeError(
            f"Invalid global tensors: expected {sorted(expected_globals)}, "
            f"got {sorted(global_weights)}"
        )
    globals_fp16 = {
        name: _prepare_cpu_tensor(tensor.to(torch.float16), pin_memory=pin_memory)
        for name, tensor in global_weights.items()
    }
    expected_global_shapes = {
        "model.embed_tokens.weight": (config.vocab_size, config.hidden_size),
        "model.norm.weight": (config.hidden_size,),
        "lm_head.weight": (config.vocab_size, config.hidden_size),
    }
    for name, expected_shape in expected_global_shapes.items():
        if tuple(globals_fp16[name].shape) != expected_shape:
            raise ValueError(
                f"Invalid global shape for {name}: expected {expected_shape}, "
                f"got {tuple(globals_fp16[name].shape)}"
            )
    return QuantizedCpuWeightStore(
        config=config,
        layers=layers,
        embed_tokens=globals_fp16["model.embed_tokens.weight"],
        final_norm=globals_fp16["model.norm.weight"],
        lm_head=globals_fp16["lm_head.weight"],
        compute_dtype=torch.float16,
        manifest=manifest,
    )


def allocate_quantized_layer_like(
    layer: dict[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        name: torch.empty_like(tensor, device=device)
        for name, tensor in layer.items()
    }


def copy_quantized_layer(
    destination: dict[str, Tensor],
    source: dict[str, Tensor],
    *,
    non_blocking: bool,
) -> None:
    for name in sorted(source):
        destination[name].copy_(source[name], non_blocking=non_blocking)


class QuantizedSyncOffloadRuntime:
    """Static INT8 prefix plus one synchronously refilled INT8 layer slot."""

    def __init__(
        self,
        store: QuantizedCpuWeightStore,
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
        self.stats.startup_h2d_bytes = sum(
            quantized_layer_bytes(layer) for layer in self.resident
        )
        self.slot = (
            allocate_quantized_layer_like(
                store.layers[self.resident_count], self.device
            )
            if self.resident_count < store.config.num_hidden_layers
            else None
        )

    def begin_forward(self) -> None:
        pass

    def weights_for(self, layer_idx: int) -> dict[str, Tensor]:
        if layer_idx < self.resident_count:
            return self.resident[layer_idx]
        if self.slot is None:
            raise RuntimeError("No dynamic INT8 layer slot allocated")
        start = time.perf_counter()
        copy_quantized_layer(
            self.slot, self.store.layers[layer_idx], non_blocking=False
        )
        self.stats.synchronous_copy_seconds += time.perf_counter() - start
        self.stats.weight_h2d_bytes += quantized_layer_bytes(
            self.store.layers[layer_idx]
        )
        return self.slot

    def after_layer(self, layer_idx: int) -> None:
        pass

    def end_forward(self) -> None:
        pass


@dataclass
class _QuantizedAsyncSlot:
    weights: dict[str, Tensor]
    ready: torch.cuda.Event
    compute_done: torch.cuda.Event
    layer_idx: int | None = None
    has_compute_event: bool = False


class QuantizedPrefetchOffloadRuntime:
    """Static INT8 prefix plus two event-protected asynchronous slots."""

    def __init__(
        self,
        store: QuantizedCpuWeightStore,
        *,
        device: torch.device | str,
        gpu_layer_budget: int,
    ) -> None:
        if gpu_layer_budget < 2:
            raise ValueError("INT8 prefetch requires gpu_layer_budget >= 2")
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
        self.stats.startup_h2d_bytes = sum(
            quantized_layer_bytes(layer) for layer in self.resident
        )
        self.copy_stream = torch.cuda.Stream(device=self.device)
        template_idx = min(self.resident_count, store.config.num_hidden_layers - 1)
        self.slots = [
            _QuantizedAsyncSlot(
                weights=allocate_quantized_layer_like(
                    store.layers[template_idx], self.device
                ),
                ready=torch.cuda.Event(enable_timing=False),
                compute_done=torch.cuda.Event(enable_timing=False),
            )
            for _ in range(2)
        ]
        self.first_dynamic = self.resident_count

    def _slot_for(self, layer_idx: int) -> _QuantizedAsyncSlot:
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
            with nvtx_range(f"weight_h2d::layer_{layer_idx}"):
                with record_function(f"prefetch_int8_layer_{layer_idx}"):
                    copy_quantized_layer(
                        slot.weights,
                        self.store.layers[layer_idx],
                        non_blocking=True,
                    )
            end.record(self.copy_stream)
            slot.ready.record(self.copy_stream)
        slot.layer_idx = layer_idx
        self.stats.async_copy_events.append((start, end))
        self.stats.weight_h2d_bytes += quantized_layer_bytes(
            self.store.layers[layer_idx]
        )

    def begin_forward(self) -> None:
        if self.first_dynamic < self.store.config.num_hidden_layers:
            self._schedule(self.first_dynamic)

    def weights_for(self, layer_idx: int) -> dict[str, Tensor]:
        if layer_idx < self.resident_count:
            return self.resident[layer_idx]
        slot = self._slot_for(layer_idx)
        if slot.layer_idx != layer_idx:
            raise RuntimeError(
                f"INT8 layer {layer_idx} was not prefetched "
                f"(slot has {slot.layer_idx})"
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


class QuantizedResidentRuntime:
    """All quantized decoder tensors permanently resident on one GPU."""

    def __init__(
        self,
        store: QuantizedCpuWeightStore,
        *,
        device: torch.device | str,
    ) -> None:
        self.store = store
        self.device = torch.device(device)
        self.gpu_layer_budget = store.config.num_hidden_layers
        self.resident_count = store.config.num_hidden_layers
        self.stats = RuntimeStats()
        self.resident = [
            {name: tensor.to(self.device) for name, tensor in layer.items()}
            for layer in store.layers
        ]
        self.stats.startup_h2d_bytes = sum(
            quantized_layer_bytes(layer) for layer in self.resident
        )

    def begin_forward(self) -> None:
        pass

    def weights_for(self, layer_idx: int) -> dict[str, Tensor]:
        return self.resident[layer_idx]

    def after_layer(self, layer_idx: int) -> None:
        pass

    def end_forward(self) -> None:
        pass


def estimate_quantized_gpu_weight_bytes(
    store: QuantizedCpuWeightStore,
    gpu_layer_budget: int,
    *,
    include_dequant_scratch: bool = True,
) -> int:
    """Estimate staged INT8 tensors, LM head, norm, and FP16 scratch."""
    if not 1 <= gpu_layer_budget <= store.config.num_hidden_layers:
        raise ValueError("invalid gpu_layer_budget")
    representative = max(quantized_layer_bytes(layer) for layer in store.layers)
    scratch = (
        store.max_dequant_elements * torch.tensor([], dtype=torch.float16).element_size()
        if include_dequant_scratch
        else 0
    )
    return (
        representative * gpu_layer_budget
        + tensor_bytes(store.final_norm)
        + tensor_bytes(store.lm_head)
        + scratch
    )
