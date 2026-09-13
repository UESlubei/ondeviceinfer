"""Layer-wise KV-cache offloading with synchronous and prefetched runtimes.

The cache keeps the authoritative history in host memory using the physical
layout ``[batch, sequence, kv_heads, head_dim]``.  A layer is materialized in
one of a small number of device working slots and exposed to attention as the
usual ``[batch, kv_heads, sequence, head_dim]`` view.

The four lifecycle hooks are optional.  Calling them around the decoder loop
allows the prefetch backend to overlap layer ``i + 1`` H2D with layer ``i``
compute.  ``update`` remains compatible with the minimal ``KVCache`` protocol
and falls back to conservative implicit lifecycle management when hooks are
not used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from .kv_quantization import (
    EncodedTensorBuffer,
    KVStorageFormat,
    make_kv_codec,
    tensor_bytes,
)
from .profiling import nvtx_range


KVOffloadMode = Literal["sync", "prefetch"]


def _copy_encoded(
    source: EncodedTensorBuffer, destination: EncodedTensorBuffer
) -> None:
    """Copy one encoded tensor without changing its storage representation."""
    if source.data.shape != destination.data.shape:
        raise ValueError("encoded data copy shape mismatch")
    destination.data.copy_(source.data, non_blocking=True)
    for name in ("scale", "zero"):
        source_meta = getattr(source, name)
        destination_meta = getattr(destination, name)
        if (source_meta is None) != (destination_meta is None):
            raise ValueError(f"encoded {name} presence mismatch")
        if source_meta is not None:
            assert destination_meta is not None
            if source_meta.shape != destination_meta.shape:
                raise ValueError(f"encoded {name} copy shape mismatch")
            destination_meta.copy_(source_meta, non_blocking=True)


@dataclass
class KVOffloadStats:
    """Cumulative payload and operation counters.

    On the CPU fallback backend, H2D/D2H counters describe the logically
    equivalent offload payload copied between host storage and the working
    slot.  This keeps byte-accounting tests deterministic on machines without
    CUDA; ``cuda_enabled`` distinguishes emulation from physical PCIe traffic.
    """

    cpu_to_gpu_bytes: int = 0
    gpu_to_cpu_bytes: int = 0
    cpu_to_gpu_copies: int = 0
    gpu_to_cpu_copies: int = 0
    h2d_data_bytes: int = 0
    h2d_metadata_bytes: int = 0
    d2h_data_bytes: int = 0
    d2h_metadata_bytes: int = 0
    quantize_calls: int = 0
    dequantize_calls: int = 0
    update_calls: int = 0
    completed_forwards: int = 0
    synchronous_h2d_seconds: float = 0.0
    synchronous_d2h_seconds: float = 0.0
    synchronous_quantize_seconds: float = 0.0
    synchronous_dequantize_seconds: float = 0.0
    accumulated_async_h2d_seconds: float = 0.0
    accumulated_async_d2h_seconds: float = 0.0
    accumulated_async_quantize_seconds: float = 0.0
    accumulated_async_dequantize_seconds: float = 0.0
    async_h2d_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )
    async_d2h_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )
    async_quantize_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )
    async_dequantize_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )

    @property
    def h2d_bytes(self) -> int:
        return self.cpu_to_gpu_bytes

    @property
    def d2h_bytes(self) -> int:
        return self.gpu_to_cpu_bytes

    @staticmethod
    def _event_seconds(
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> float:
        return sum(start.elapsed_time(end) for start, end in events) / 1000.0

    def h2d_seconds(self) -> float:
        return (
            self.synchronous_h2d_seconds
            + self.accumulated_async_h2d_seconds
            + self._event_seconds(self.async_h2d_events)
        )

    def d2h_seconds(self) -> float:
        return (
            self.synchronous_d2h_seconds
            + self.accumulated_async_d2h_seconds
            + self._event_seconds(self.async_d2h_events)
        )

    def quantize_seconds(self) -> float:
        return (
            self.synchronous_quantize_seconds
            + self.accumulated_async_quantize_seconds
            + self._event_seconds(self.async_quantize_events)
        )

    def dequantize_seconds(self) -> float:
        return (
            self.synchronous_dequantize_seconds
            + self.accumulated_async_dequantize_seconds
            + self._event_seconds(self.async_dequantize_events)
        )


@dataclass
class _WorkingSlot:
    key: Tensor
    value: Tensor
    encoded_key: EncodedTensorBuffer | None = None
    encoded_value: EncodedTensorBuffer | None = None
    ready: torch.cuda.Event | None = None
    append_ready: torch.cuda.Event | None = None
    compute_done: torch.cuda.Event | None = None
    writeback_done: torch.cuda.Event | None = None
    tag: tuple[int, int] | None = None
    loaded_length: int = 0
    has_compute_event: bool = False
    has_writeback_event: bool = False


class OffloadedKVCache:
    """Preallocated, exact layer-wise KV-cache offloading.

    Parameters
    ----------
    mode:
        ``"sync"`` uses one working slot. ``"prefetch"`` uses two CUDA
        slots and an independent transfer stream.  Prefetch automatically
        falls back to sync on CPU, or when the supplied budget can hold only
        one complete layer slot.
    gpu_budget_bytes:
        Budget for managed KV working slots.  It must hold at least one full
        layer at ``max_seq_len``.  The first implementation intentionally does
        not use excess budget for a cache of resident layers: sequential
        layer scans would otherwise thrash a normal FIFO/LRU cache.

    Notes
    -----
    No historical token is discarded.  With ``storage_format="fp16"`` the
    values are exact copies; ``storage_format="int8_hqq"`` retains the full
    history with lossy group-wise affine encoding.  Only the newly appended
    K/V slice is copied back to the host after an update; the full active
    history is copied host-to-device when a layer is acquired.
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
        mode: KVOffloadMode = "sync",
        gpu_budget_bytes: int | None = None,
        pin_memory: bool = True,
        track_transfer_timing: bool = False,
        storage_format: KVStorageFormat = "fp16",
        kv_group_size: int = 64,
        hqq_optimize: bool = False,
        hqq_runtime_backend: str = "torch",
    ) -> None:
        dimensions = {
            "num_layers": num_layers,
            "batch_size": batch_size,
            "num_key_value_heads": num_key_value_heads,
            "max_seq_len": max_seq_len,
            "head_dim": head_dim,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if mode not in ("sync", "prefetch"):
            raise ValueError("mode must be 'sync' or 'prefetch'")

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_key_value_heads = num_key_value_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.storage_format = storage_format
        self.kv_group_size = kv_group_size
        self.hqq_optimize = hqq_optimize
        self.hqq_runtime_backend = hqq_runtime_backend
        self.codec = make_kv_codec(
            storage_format,
            compute_dtype=dtype,
            head_dim=head_dim,
            group_size=kv_group_size,
            hqq_optimize=hqq_optimize,
            hqq_runtime_backend=hqq_runtime_backend,
        )
        self.requested_mode: KVOffloadMode = mode
        self.track_transfer_timing = track_transfer_timing
        self.cuda_enabled = self.device.type == "cuda"
        if self.cuda_enabled and not torch.cuda.is_available():
            raise RuntimeError("A CUDA KV-cache device was requested, but CUDA is unavailable")

        element_size = torch.empty((), dtype=dtype).element_size()
        self.attention_layer_capacity_bytes = (
            2
            * batch_size
            * max_seq_len
            * num_key_value_heads
            * head_dim
            * element_size
        )
        host_shape = (
            batch_size,
            max_seq_len,
            num_key_value_heads,
            head_dim,
        )
        self.encoded_layer_capacity_bytes = 2 * self.codec.bytes_for_shape(host_shape)
        self.staging_layer_capacity_bytes = (
            self.encoded_layer_capacity_bytes
            if self.storage_format == "int8_hqq"
            else 0
        )
        self.slot_capacity_bytes = (
            self.attention_layer_capacity_bytes + self.staging_layer_capacity_bytes
        )
        # Backward-compatible name used by the stage-7 budget tests.  For the
        # INT8 path this now correctly includes compressed staging in addition
        # to the FP16 attention materialization slot.
        self.layer_capacity_bytes = self.slot_capacity_bytes
        if gpu_budget_bytes is None:
            requested_slots = 2 if mode == "prefetch" else 1
            gpu_budget_bytes = requested_slots * self.slot_capacity_bytes
        if isinstance(gpu_budget_bytes, bool) or not isinstance(gpu_budget_bytes, int):
            raise TypeError("gpu_budget_bytes must be an integer number of bytes")
        if gpu_budget_bytes < self.layer_capacity_bytes:
            raise ValueError(
                "gpu_budget_bytes cannot hold one complete layer KV slot: "
                f"need {self.layer_capacity_bytes}, got {gpu_budget_bytes}"
            )
        self.gpu_budget_bytes = gpu_budget_bytes

        budget_slots = gpu_budget_bytes // self.slot_capacity_bytes
        self.fallback_reason: str | None = None
        if mode == "prefetch" and budget_slots < 2:
            self.effective_mode: KVOffloadMode = "sync"
            self.fallback_reason = "prefetch requires two complete layer slots"
        elif mode == "prefetch" and not self.cuda_enabled:
            self.effective_mode = "sync"
            self.fallback_reason = "CUDA unavailable; using synchronous CPU emulation"
        else:
            self.effective_mode = mode

        if self.cuda_enabled and not pin_memory:
            raise ValueError("CUDA KV offloading requires pinned host memory")
        use_pinned_host = self.cuda_enabled and pin_memory
        self._key_host_buffers = [
            self.codec.allocate(
                host_shape, device="cpu", pin_memory=use_pinned_host
            )
            for _ in range(num_layers)
        ]
        self._value_host_buffers = [
            self.codec.allocate(
                host_shape, device="cpu", pin_memory=use_pinned_host
            )
            for _ in range(num_layers)
        ]
        # ``key_host``/``value_host`` remain available for the exact FP16
        # backend and expose qdata for INT8 diagnostics.
        self.key_host = [buffer.data for buffer in self._key_host_buffers]
        self.value_host = [buffer.data for buffer in self._value_host_buffers]
        self.key_q_host = self.key_host if storage_format == "int8_hqq" else []
        self.value_q_host = self.value_host if storage_format == "int8_hqq" else []
        self.key_scale_host = [
            buffer.scale for buffer in self._key_host_buffers if buffer.scale is not None
        ]
        self.value_scale_host = [
            buffer.scale for buffer in self._value_host_buffers if buffer.scale is not None
        ]
        self.key_zero_host = [
            buffer.zero for buffer in self._key_host_buffers if buffer.zero is not None
        ]
        self.value_zero_host = [
            buffer.zero for buffer in self._value_host_buffers if buffer.zero is not None
        ]
        self.host_is_pinned = all(
            buffer.is_pinned
            for buffer in self._key_host_buffers + self._value_host_buffers
        )

        self.slot_count = 2 if self.effective_mode == "prefetch" else 1
        self._slots = []
        for _ in range(self.slot_count):
            encoded_key = None
            encoded_value = None
            if self.storage_format == "int8_hqq":
                encoded_key = self.codec.allocate(host_shape, device=self.device)
                encoded_value = self.codec.allocate(host_shape, device=self.device)
            self._slots.append(
                _WorkingSlot(
                    key=torch.empty(host_shape, dtype=dtype, device=self.device),
                    value=torch.empty(host_shape, dtype=dtype, device=self.device),
                    encoded_key=encoded_key,
                    encoded_value=encoded_value,
                )
            )
        self.transfer_stream: torch.cuda.Stream | None = None
        self._sync_copy_start: torch.cuda.Event | None = None
        self._sync_copy_end: torch.cuda.Event | None = None
        self._sync_codec_start: torch.cuda.Event | None = None
        self._sync_codec_end: torch.cuda.Event | None = None
        self._sync_done: torch.cuda.Event | None = None
        if self.effective_mode == "prefetch":
            self.transfer_stream = torch.cuda.Stream(device=self.device)
            allocation_ready = torch.cuda.Event(enable_timing=False)
            allocation_ready.record(torch.cuda.current_stream(self.device))
            self.transfer_stream.wait_event(allocation_ready)
            for slot in self._slots:
                slot.ready = torch.cuda.Event(enable_timing=False)
                slot.append_ready = torch.cuda.Event(enable_timing=False)
                slot.compute_done = torch.cuda.Event(enable_timing=False)
                slot.writeback_done = torch.cuda.Event(enable_timing=False)
        elif self.cuda_enabled:
            # Sync mode serializes every timed copy, so one reusable pair is
            # sufficient and avoids creating O(layers * tokens) Event objects.
            self._sync_copy_start = torch.cuda.Event(enable_timing=True)
            self._sync_copy_end = torch.cuda.Event(enable_timing=True)
            self._sync_codec_start = torch.cuda.Event(enable_timing=True)
            self._sync_codec_end = torch.cuda.Event(enable_timing=True)
            self._sync_done = torch.cuda.Event(enable_timing=False)

        self.lengths = [0] * num_layers
        self.stats = KVOffloadStats()
        self._epoch = -1
        self._in_forward = False
        self._active_layer: int | None = None
        self._active_slot: _WorkingSlot | None = None
        self._last_layer_seen: int | None = None

    @property
    def managed_cpu_bytes(self) -> int:
        return sum(
            buffer.storage_bytes
            for buffer in self._key_host_buffers + self._value_host_buffers
        )

    @property
    def managed_cpu_data_bytes(self) -> int:
        return sum(tensor_bytes(tensor) for tensor in self.key_host + self.value_host)

    @property
    def managed_cpu_metadata_bytes(self) -> int:
        return self.managed_cpu_bytes - self.managed_cpu_data_bytes

    @property
    def managed_device_bytes(self) -> int:
        total = 0
        for slot in self._slots:
            total += tensor_bytes(slot.key) + tensor_bytes(slot.value)
            if slot.encoded_key is not None:
                total += slot.encoded_key.storage_bytes
            if slot.encoded_value is not None:
                total += slot.encoded_value.storage_bytes
        return total

    @property
    def managed_gpu_bytes(self) -> int:
        """Physical GPU bytes, or zero for the CPU emulation backend."""
        return self.managed_device_bytes if self.cuda_enabled else 0

    @property
    def storage_bytes(self) -> int:
        return self.managed_cpu_bytes + self.managed_device_bytes

    @property
    def host_compression_ratio(self) -> float:
        fp_bytes = self.num_layers * self.attention_layer_capacity_bytes
        return self.managed_cpu_bytes / fp_bytes

    @property
    def hqq_version(self) -> str | None:
        return getattr(self.codec, "library_version", None)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        self._validate_layer_idx(layer_idx)
        return self.lengths[layer_idx]

    def begin_forward(self, *, query_length: int | None = None) -> None:
        """Start one model forward and, when possible, prefetch layer zero."""
        if query_length is not None and query_length <= 0:
            raise ValueError("query_length must be positive")
        if self._in_forward:
            raise RuntimeError("begin_forward called before the previous forward ended")
        self._epoch += 1
        self._in_forward = True
        self._last_layer_seen = None
        if self.effective_mode == "prefetch":
            self._schedule_prefetch(0)

    def before_layer(self, layer_idx: int) -> None:
        """Acquire the layer's working slot before decoder computation."""
        self._validate_layer_idx(layer_idx)
        if self._active_layer == layer_idx:
            return
        if self._active_layer is not None:
            # Implicit compatibility path: by the time the next layer reaches
            # cache.update(), all previous-layer consumers have been enqueued.
            self.after_layer(self._active_layer)

        if self._in_forward and self._last_layer_seen is not None:
            if layer_idx <= self._last_layer_seen:
                self.end_forward()
        if not self._in_forward:
            self.begin_forward()

        if self.effective_mode == "prefetch":
            slot = self._slot_for(layer_idx)
            expected_tag = (self._epoch, layer_idx)
            if slot.tag != expected_tag:
                self._schedule_prefetch(layer_idx)
            assert slot.ready is not None
            torch.cuda.current_stream(self.device).wait_event(slot.ready)
        else:
            slot = self._slots[0]
            self._load_sync(slot, layer_idx)

        self._active_layer = layer_idx
        self._active_slot = slot
        self._last_layer_seen = layer_idx

    def update(
        self, layer_idx: int, key_states: Tensor, value_states: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Append K/V and return the complete device history in attention layout."""
        self._validate_update(layer_idx, key_states, value_states)
        self.before_layer(layer_idx)
        assert self._active_slot is not None
        slot = self._active_slot

        start = self.lengths[layer_idx]
        query_length = key_states.shape[-2]
        end = start + query_length
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache capacity exceeded: requested {end}, capacity {self.max_seq_len}"
            )

        key_token_major = key_states.transpose(1, 2)
        value_token_major = value_states.transpose(1, 2)
        slot.key[:, start:end].copy_(key_token_major)
        slot.value[:, start:end].copy_(value_token_major)
        self.lengths[layer_idx] = end
        slot.loaded_length = end
        self.stats.update_calls += 1

        key_host_slice, value_host_slice = self._host_slices(layer_idx, start, end)
        data_bytes, metadata_bytes = self._encoded_pair_byte_breakdown(
            key_host_slice, value_host_slice
        )
        self.stats.gpu_to_cpu_bytes += data_bytes + metadata_bytes
        self.stats.d2h_data_bytes += data_bytes
        self.stats.d2h_metadata_bytes += metadata_bytes
        self.stats.gpu_to_cpu_copies += 1
        if self.effective_mode == "prefetch":
            self._writeback_async(slot, layer_idx, start, end)
            # At this point the current layer's full K/V is ready and its
            # attention kernel will be enqueued immediately after update()
            # returns.  Queueing the next layer here (rather than waiting for
            # after_layer) creates the intended copy/compute overlap.
            if layer_idx + 1 < self.num_layers:
                self._schedule_prefetch(layer_idx + 1)
        else:
            # CUDA sync mode intentionally blocks here.  On CPU this is the
            # deterministic emulation of the same device-to-host payload.
            self._writeback_sync(slot, layer_idx, start, end)

        return (
            slot.key[:, :end].transpose(1, 2),
            slot.value[:, :end].transpose(1, 2),
        )

    def after_layer(self, layer_idx: int) -> None:
        """Protect the active slot and launch next-layer prefetch."""
        self._validate_layer_idx(layer_idx)
        if self._active_layer != layer_idx or self._active_slot is None:
            raise RuntimeError(
                f"after_layer({layer_idx}) does not match active layer "
                f"{self._active_layer}"
            )
        slot = self._active_slot
        if self.effective_mode == "prefetch":
            assert slot.compute_done is not None
            slot.compute_done.record(torch.cuda.current_stream(self.device))
            slot.has_compute_event = True

        self._active_layer = None
        self._active_slot = None
        next_layer = layer_idx + 1
        if self.effective_mode == "prefetch" and next_layer < self.num_layers:
            self._schedule_prefetch(next_layer)

    def end_forward(self) -> None:
        """Close a model forward without introducing a host synchronization."""
        if not self._in_forward:
            return
        if self._active_layer is not None:
            self.after_layer(self._active_layer)
        self._in_forward = False
        self.stats.completed_forwards += 1

    def synchronize(self) -> None:
        """Finish pending slot use and make all host writebacks observable."""
        if self._active_layer is not None:
            self.after_layer(self._active_layer)
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)

    def reset_stats(self) -> None:
        self.stats = KVOffloadStats()

    def drain_timing_events(self) -> None:
        """Fold completed CUDA timings into scalars and release Event objects.

        Long generations otherwise retain two event pairs for every layer and
        token until the very end.  Benchmark windows call this after a device
        synchronization, bounding host/driver resources without changing the
        byte counters.
        """
        if self.stats.async_h2d_events:
            self.stats.accumulated_async_h2d_seconds += self.stats._event_seconds(
                self.stats.async_h2d_events
            )
            self.stats.async_h2d_events.clear()
        if self.stats.async_d2h_events:
            self.stats.accumulated_async_d2h_seconds += self.stats._event_seconds(
                self.stats.async_d2h_events
            )
            self.stats.async_d2h_events.clear()
        if self.stats.async_quantize_events:
            self.stats.accumulated_async_quantize_seconds += (
                self.stats._event_seconds(self.stats.async_quantize_events)
            )
            self.stats.async_quantize_events.clear()
        if self.stats.async_dequantize_events:
            self.stats.accumulated_async_dequantize_seconds += (
                self.stats._event_seconds(self.stats.async_dequantize_events)
            )
            self.stats.async_dequantize_events.clear()

    def reset(self) -> None:
        self.synchronize()
        self.lengths[:] = [0] * self.num_layers
        self.stats = KVOffloadStats()
        self._in_forward = False
        self._active_layer = None
        self._active_slot = None
        self._last_layer_seen = None
        for slot in self._slots:
            slot.tag = None
            slot.loaded_length = 0
            slot.has_compute_event = False
            slot.has_writeback_event = False

    def _validate_layer_idx(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}"
            )

    def _validate_update(
        self, layer_idx: int, key_states: Tensor, value_states: Tensor
    ) -> None:
        self._validate_layer_idx(layer_idx)
        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have the same shape")
        if key_states.ndim != 4:
            raise ValueError("K/V states must have shape [batch, kv_heads, seq, head_dim]")
        expected = (
            self.batch_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        actual = (key_states.shape[0], key_states.shape[1], key_states.shape[3])
        if actual != expected:
            raise ValueError(
                "K/V shape mismatch: expected batch/heads/head_dim "
                f"{expected}, got {actual}"
            )
        if key_states.device != self.device or value_states.device != self.device:
            raise ValueError(f"K/V states must be on cache device {self.device}")
        if key_states.dtype != self.dtype or value_states.dtype != self.dtype:
            raise ValueError(f"K/V states must use cache dtype {self.dtype}")

    def _slot_for(self, layer_idx: int) -> _WorkingSlot:
        return self._slots[layer_idx % self.slot_count]

    def _host_slices(
        self, layer_idx: int, start: int, end: int
    ) -> tuple[EncodedTensorBuffer, EncodedTensorBuffer]:
        return (
            self._key_host_buffers[layer_idx].token_slice(start, end),
            self._value_host_buffers[layer_idx].token_slice(start, end),
        )

    @staticmethod
    def _slot_encoded_slices(
        slot: _WorkingSlot, start: int, end: int
    ) -> tuple[EncodedTensorBuffer, EncodedTensorBuffer]:
        if slot.encoded_key is None or slot.encoded_value is None:
            raise RuntimeError("INT8 slot staging buffers are unavailable")
        return (
            slot.encoded_key.token_slice(start, end),
            slot.encoded_value.token_slice(start, end),
        )

    @staticmethod
    def _encoded_pair_byte_breakdown(
        key: EncodedTensorBuffer, value: EncodedTensorBuffer
    ) -> tuple[int, int]:
        data_bytes = tensor_bytes(key.data) + tensor_bytes(value.data)
        metadata_bytes = key.storage_bytes + value.storage_bytes - data_bytes
        return data_bytes, metadata_bytes

    def _load_sync(self, slot: _WorkingSlot, layer_idx: int) -> None:
        length = self.lengths[layer_idx]
        if length:
            key_host, value_host = self._host_slices(layer_idx, 0, length)
            key_output = slot.key[:, :length]
            value_output = slot.value[:, :length]
            if self.cuda_enabled:
                assert self._sync_copy_start is not None
                assert self._sync_copy_end is not None
                assert self._sync_codec_start is not None
                assert self._sync_codec_end is not None
                assert self._sync_done is not None
                stream = torch.cuda.current_stream(self.device)
                if self.track_transfer_timing:
                    self._sync_copy_start.record(stream)
                if self.storage_format == "int8_hqq":
                    key_staging, value_staging = self._slot_encoded_slices(
                        slot, 0, length
                    )
                    with nvtx_range(f"kv_int8_h2d::layer_{layer_idx}"):
                        _copy_encoded(key_host, key_staging)
                        _copy_encoded(value_host, value_staging)
                else:
                    with nvtx_range(f"kv_h2d::layer_{layer_idx}"):
                        self.codec.decode_into(key_host, key_output)
                        self.codec.decode_into(value_host, value_output)
                if self.track_transfer_timing:
                    self._sync_copy_end.record(stream)
                if self.storage_format == "int8_hqq":
                    if self.track_transfer_timing:
                        self._sync_codec_start.record(stream)
                    with nvtx_range(f"kv_int8_dequant::layer_{layer_idx}"):
                        self.codec.decode_into(key_staging, key_output)
                        self.codec.decode_into(value_staging, value_output)
                    if self.track_transfer_timing:
                        self._sync_codec_end.record(stream)
                    self.stats.dequantize_calls += 2
                self._sync_done.record(stream)
                self._sync_done.synchronize()
                if self.track_transfer_timing:
                    self.stats.synchronous_h2d_seconds += (
                        self._sync_copy_start.elapsed_time(self._sync_copy_end)
                        / 1000.0
                    )
                    if self.storage_format == "int8_hqq":
                        self.stats.synchronous_dequantize_seconds += (
                            self._sync_codec_start.elapsed_time(self._sync_codec_end)
                            / 1000.0
                        )
            else:
                copy_start = time.perf_counter()
                if self.storage_format == "int8_hqq":
                    key_staging, value_staging = self._slot_encoded_slices(
                        slot, 0, length
                    )
                    _copy_encoded(key_host, key_staging)
                    _copy_encoded(value_host, value_staging)
                else:
                    self.codec.decode_into(key_host, key_output)
                    self.codec.decode_into(value_host, value_output)
                self.stats.synchronous_h2d_seconds += (
                    time.perf_counter() - copy_start
                )
                if self.storage_format == "int8_hqq":
                    codec_start = time.perf_counter()
                    self.codec.decode_into(key_staging, key_output)
                    self.codec.decode_into(value_staging, value_output)
                    self.stats.synchronous_dequantize_seconds += (
                        time.perf_counter() - codec_start
                    )
                    self.stats.dequantize_calls += 2
            data_bytes, metadata_bytes = self._encoded_pair_byte_breakdown(
                key_host, value_host
            )
            self.stats.cpu_to_gpu_bytes += data_bytes + metadata_bytes
            self.stats.h2d_data_bytes += data_bytes
            self.stats.h2d_metadata_bytes += metadata_bytes
            self.stats.cpu_to_gpu_copies += 1
        slot.tag = (self._epoch, layer_idx)
        slot.loaded_length = length

    def _writeback_sync(
        self, slot: _WorkingSlot, layer_idx: int, start: int, end: int
    ) -> None:
        key_host, value_host = self._host_slices(layer_idx, start, end)
        key_source = slot.key[:, start:end]
        value_source = slot.value[:, start:end]
        if self.cuda_enabled:
            assert self._sync_copy_start is not None
            assert self._sync_copy_end is not None
            assert self._sync_codec_start is not None
            assert self._sync_codec_end is not None
            assert self._sync_done is not None
            stream = torch.cuda.current_stream(self.device)
            if self.storage_format == "int8_hqq":
                key_staging, value_staging = self._slot_encoded_slices(
                    slot, start, end
                )
                if self.track_transfer_timing:
                    self._sync_codec_start.record(stream)
                with nvtx_range(f"kv_int8_quant::layer_{layer_idx}"):
                    self.codec.encode_into(key_source, key_staging)
                    self.codec.encode_into(value_source, value_staging)
                if self.track_transfer_timing:
                    self._sync_codec_end.record(stream)
                self.stats.quantize_calls += 2
            if self.track_transfer_timing:
                self._sync_copy_start.record(stream)
            if self.storage_format == "int8_hqq":
                with nvtx_range(f"kv_int8_d2h::layer_{layer_idx}"):
                    _copy_encoded(key_staging, key_host)
                    _copy_encoded(value_staging, value_host)
            else:
                with nvtx_range(f"kv_d2h::layer_{layer_idx}"):
                    self.codec.encode_into(key_source, key_host)
                    self.codec.encode_into(value_source, value_host)
            if self.track_transfer_timing:
                self._sync_copy_end.record(stream)
            self._sync_done.record(stream)
            self._sync_done.synchronize()
            if self.track_transfer_timing:
                self.stats.synchronous_d2h_seconds += (
                    self._sync_copy_start.elapsed_time(self._sync_copy_end) / 1000.0
                )
                if self.storage_format == "int8_hqq":
                    self.stats.synchronous_quantize_seconds += (
                        self._sync_codec_start.elapsed_time(self._sync_codec_end)
                        / 1000.0
                    )
        else:
            if self.storage_format == "int8_hqq":
                key_staging, value_staging = self._slot_encoded_slices(
                    slot, start, end
                )
                codec_start = time.perf_counter()
                self.codec.encode_into(key_source, key_staging)
                self.codec.encode_into(value_source, value_staging)
                self.stats.synchronous_quantize_seconds += (
                    time.perf_counter() - codec_start
                )
                self.stats.quantize_calls += 2
            copy_start = time.perf_counter()
            if self.storage_format == "int8_hqq":
                _copy_encoded(key_staging, key_host)
                _copy_encoded(value_staging, value_host)
            else:
                self.codec.encode_into(key_source, key_host)
                self.codec.encode_into(value_source, value_host)
            self.stats.synchronous_d2h_seconds += time.perf_counter() - copy_start

    def _schedule_prefetch(self, layer_idx: int) -> None:
        assert self.transfer_stream is not None
        slot = self._slot_for(layer_idx)
        tag = (self._epoch, layer_idx)
        if slot.tag == tag:
            return
        length = self.lengths[layer_idx]
        key_host, value_host = self._host_slices(layer_idx, 0, length)
        with torch.cuda.stream(self.transfer_stream):
            if slot.has_compute_event:
                assert slot.compute_done is not None
                self.transfer_stream.wait_event(slot.compute_done)
            if slot.has_writeback_event:
                assert slot.writeback_done is not None
                self.transfer_stream.wait_event(slot.writeback_done)
            if length and self.track_transfer_timing:
                copy_start = torch.cuda.Event(enable_timing=True)
                copy_end = torch.cuda.Event(enable_timing=True)
                copy_start.record(self.transfer_stream)
            if length:
                if self.storage_format == "int8_hqq":
                    key_staging, value_staging = self._slot_encoded_slices(
                        slot, 0, length
                    )
                    with nvtx_range(f"kv_int8_h2d::layer_{layer_idx}"):
                        _copy_encoded(key_host, key_staging)
                        _copy_encoded(value_host, value_staging)
                else:
                    with nvtx_range(f"kv_h2d::layer_{layer_idx}"):
                        self.codec.decode_into(key_host, slot.key[:, :length])
                        self.codec.decode_into(value_host, slot.value[:, :length])
                if self.track_transfer_timing:
                    copy_end.record(self.transfer_stream)
                    self.stats.async_h2d_events.append((copy_start, copy_end))
                if self.storage_format == "int8_hqq":
                    if self.track_transfer_timing:
                        codec_start = torch.cuda.Event(enable_timing=True)
                        codec_end = torch.cuda.Event(enable_timing=True)
                        codec_start.record(self.transfer_stream)
                    with nvtx_range(f"kv_int8_dequant::layer_{layer_idx}"):
                        self.codec.decode_into(key_staging, slot.key[:, :length])
                        self.codec.decode_into(value_staging, slot.value[:, :length])
                    if self.track_transfer_timing:
                        codec_end.record(self.transfer_stream)
                        self.stats.async_dequantize_events.append(
                            (codec_start, codec_end)
                        )
                    self.stats.dequantize_calls += 2
            assert slot.ready is not None
            slot.ready.record(self.transfer_stream)
        if length:
            data_bytes, metadata_bytes = self._encoded_pair_byte_breakdown(
                key_host, value_host
            )
            self.stats.cpu_to_gpu_bytes += data_bytes + metadata_bytes
            self.stats.h2d_data_bytes += data_bytes
            self.stats.h2d_metadata_bytes += metadata_bytes
            self.stats.cpu_to_gpu_copies += 1
        slot.tag = tag
        slot.loaded_length = length

    def _writeback_async(
        self, slot: _WorkingSlot, layer_idx: int, start: int, end: int
    ) -> None:
        assert self.transfer_stream is not None
        assert slot.append_ready is not None
        assert slot.writeback_done is not None
        compute_stream = torch.cuda.current_stream(self.device)
        slot.append_ready.record(compute_stream)
        key_host, value_host = self._host_slices(layer_idx, start, end)
        key_source = slot.key[:, start:end]
        value_source = slot.value[:, start:end]
        with torch.cuda.stream(self.transfer_stream):
            self.transfer_stream.wait_event(slot.append_ready)
            if self.storage_format == "int8_hqq":
                key_staging, value_staging = self._slot_encoded_slices(
                    slot, start, end
                )
                if self.track_transfer_timing:
                    quant_start = torch.cuda.Event(enable_timing=True)
                    quant_end = torch.cuda.Event(enable_timing=True)
                    quant_start.record(self.transfer_stream)
                with nvtx_range(f"kv_int8_quant::layer_{layer_idx}"):
                    self.codec.encode_into(key_source, key_staging)
                    self.codec.encode_into(value_source, value_staging)
                if self.track_transfer_timing:
                    quant_end.record(self.transfer_stream)
                    self.stats.async_quantize_events.append(
                        (quant_start, quant_end)
                    )
                self.stats.quantize_calls += 2
            if self.track_transfer_timing:
                copy_start = torch.cuda.Event(enable_timing=True)
                copy_end = torch.cuda.Event(enable_timing=True)
                copy_start.record(self.transfer_stream)
            if self.storage_format == "int8_hqq":
                with nvtx_range(f"kv_int8_d2h::layer_{layer_idx}"):
                    _copy_encoded(key_staging, key_host)
                    _copy_encoded(value_staging, value_host)
            else:
                with nvtx_range(f"kv_d2h::layer_{layer_idx}"):
                    self.codec.encode_into(key_source, key_host)
                    self.codec.encode_into(value_source, value_host)
            if self.track_transfer_timing:
                copy_end.record(self.transfer_stream)
            slot.writeback_done.record(self.transfer_stream)
        if self.track_transfer_timing:
            self.stats.async_d2h_events.append((copy_start, copy_end))
        slot.has_writeback_event = True


class SyncKVOffloadCache(OffloadedKVCache):
    """Convenience constructor for the one-slot synchronous backend."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(mode="sync", **kwargs)


class PrefetchKVOffloadCache(OffloadedKVCache):
    """Convenience constructor for the requested two-slot prefetch backend."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(mode="prefetch", **kwargs)
