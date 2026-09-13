"""Storage codecs used by the layer-wise KV-cache offload runtime.

The INT8 codec deliberately quantizes storage only.  Attention still consumes
the cache's normal floating-point working slots, which keeps the numerical
change isolated from the SDPA implementation and makes FP16/INT8 experiments
directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Literal, Protocol

import torch
from torch import Tensor


KVStorageFormat = Literal["fp16", "int8_hqq"]
HQQRuntimeBackend = Literal["torch", "hqq_reference"]


def tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@dataclass
class EncodedTensorBuffer:
    """Preallocated data and optional affine metadata for one K or V tensor."""

    data: Tensor
    scale: Tensor | None = None
    zero: Tensor | None = None

    @property
    def storage_bytes(self) -> int:
        tensors = (self.data, self.scale, self.zero)
        return sum(tensor_bytes(tensor) for tensor in tensors if tensor is not None)

    @property
    def is_pinned(self) -> bool:
        tensors = (self.data, self.scale, self.zero)
        return all(
            tensor.is_pinned() for tensor in tensors if tensor is not None
        )

    def token_slice(self, start: int, end: int) -> "EncodedTensorBuffer":
        return EncodedTensorBuffer(
            data=self.data[:, start:end],
            scale=None if self.scale is None else self.scale[:, start:end],
            zero=None if self.zero is None else self.zero[:, start:end],
        )


class KVCodec(Protocol):
    """Interface between cache placement and the stored numerical format."""

    storage_format: KVStorageFormat
    compute_dtype: torch.dtype
    head_dim: int

    def allocate(
        self,
        shape: tuple[int, int, int, int],
        *,
        device: torch.device | str,
        pin_memory: bool = False,
    ) -> EncodedTensorBuffer: ...

    def encode_into(self, source: Tensor, destination: EncodedTensorBuffer) -> None: ...

    def decode_into(self, source: EncodedTensorBuffer, destination: Tensor) -> None: ...

    def bytes_for_shape(self, shape: tuple[int, int, int, int]) -> int: ...


class FP16KVCodec:
    """Identity codec preserving the original floating-point offload path."""

    storage_format: KVStorageFormat = "fp16"

    def __init__(self, *, compute_dtype: torch.dtype, head_dim: int) -> None:
        self.compute_dtype = compute_dtype
        self.head_dim = head_dim

    def allocate(
        self,
        shape: tuple[int, int, int, int],
        *,
        device: torch.device | str,
        pin_memory: bool = False,
    ) -> EncodedTensorBuffer:
        return EncodedTensorBuffer(
            data=torch.empty(
                shape,
                dtype=self.compute_dtype,
                device=device,
                pin_memory=pin_memory,
            )
        )

    def encode_into(self, source: Tensor, destination: EncodedTensorBuffer) -> None:
        if destination.scale is not None or destination.zero is not None:
            raise ValueError("FP16 storage must not contain scale/zero metadata")
        if source.shape != destination.data.shape:
            raise ValueError(
                f"FP16 encode shape mismatch: {source.shape} != {destination.data.shape}"
            )
        destination.data.copy_(source, non_blocking=True)

    def decode_into(self, source: EncodedTensorBuffer, destination: Tensor) -> None:
        if source.scale is not None or source.zero is not None:
            raise ValueError("FP16 storage must not contain scale/zero metadata")
        if source.data.shape != destination.shape:
            raise ValueError(
                f"FP16 decode shape mismatch: {source.data.shape} != {destination.shape}"
            )
        destination.copy_(source.data, non_blocking=True)

    def bytes_for_shape(self, shape: tuple[int, int, int, int]) -> int:
        elements = 1
        for dimension in shape:
            elements *= dimension
        return elements * torch.empty((), dtype=self.compute_dtype).element_size()


class HQQInt8KVCodec:
    """HQQ-compatible group-wise affine UINT8 storage for online K/V tensors.

    HQQ's public ``Quantizer`` implements the quantization itself.  Iterative
    half-quadratic optimization is disabled by default because cache encoding
    runs on every decoder layer/token; the dynamic per-group affine path is the
    online configuration also evaluated by this project.  Scale and zero are
    persisted as FP16 exactly as specified by the experiment design.
    """

    storage_format: KVStorageFormat = "int8_hqq"
    nbits = 8

    def __init__(
        self,
        *,
        compute_dtype: torch.dtype,
        head_dim: int,
        group_size: int = 64,
        optimize: bool = False,
        metadata_dtype: torch.dtype = torch.float16,
        runtime_backend: HQQRuntimeBackend = "torch",
    ) -> None:
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if head_dim % group_size:
            raise ValueError(
                f"head_dim {head_dim} must be divisible by group_size {group_size}"
            )
        if not compute_dtype.is_floating_point:
            raise TypeError("compute_dtype must be floating point")
        if metadata_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("metadata_dtype must be float16, bfloat16, or float32")
        if runtime_backend not in ("torch", "hqq_reference"):
            raise ValueError("runtime_backend must be 'torch' or 'hqq_reference'")
        if optimize and runtime_backend != "hqq_reference":
            raise ValueError("HQQ iterative optimization requires hqq_reference backend")
        try:
            from hqq.core.quantize import Quantizer
        except ImportError as error:
            raise ImportError(
                "storage_format='int8_hqq' requires hqq==0.2.8.post1; "
                "install it with `python -m pip install hqq==0.2.8.post1`"
            ) from error

        self.compute_dtype = compute_dtype
        self.head_dim = head_dim
        self.group_size = group_size
        self.groups_per_head = head_dim // group_size
        self.optimize = optimize
        self.metadata_dtype = metadata_dtype
        self.runtime_backend = runtime_backend
        self.quantizer = Quantizer
        self.library_version = metadata.version("hqq")

    def allocate(
        self,
        shape: tuple[int, int, int, int],
        *,
        device: torch.device | str,
        pin_memory: bool = False,
    ) -> EncodedTensorBuffer:
        self._validate_shape(shape)
        metadata_shape = shape[:-1] + (self.groups_per_head,)
        return EncodedTensorBuffer(
            data=torch.empty(
                shape,
                dtype=torch.uint8,
                device=device,
                pin_memory=pin_memory,
            ),
            scale=torch.empty(
                metadata_shape,
                dtype=self.metadata_dtype,
                device=device,
                pin_memory=pin_memory,
            ),
            zero=torch.empty(
                metadata_shape,
                dtype=self.metadata_dtype,
                device=device,
                pin_memory=pin_memory,
            ),
        )

    def encode_into(self, source: Tensor, destination: EncodedTensorBuffer) -> None:
        self._validate_buffer_pair(source, destination)
        if self.runtime_backend == "torch":
            self._encode_torch(source, destination)
            return
        self._encode_hqq_reference(source, destination)

    def _encode_hqq_reference(
        self, source: Tensor, destination: EncodedTensorBuffer
    ) -> None:
        rows = source.reshape(-1, self.head_dim)
        quantized, meta = self.quantizer.quantize(
            rows,
            nbits=self.nbits,
            channel_wise=True,
            group_size=self.group_size,
            optimize=self.optimize,
            round_zero=False,
            axis=1,
            bitpack=True,
            compute_dtype=self.compute_dtype,
            view_as_float=False,
            device=str(source.device),
        )
        assert destination.scale is not None
        assert destination.zero is not None
        destination.data.copy_(quantized.reshape_as(destination.data))
        destination.scale.copy_(
            meta["scale"].reshape_as(destination.scale).to(self.metadata_dtype)
        )
        destination.zero.copy_(
            meta["zero"].reshape_as(destination.zero).to(self.metadata_dtype)
        )

    def decode_into(self, source: EncodedTensorBuffer, destination: Tensor) -> None:
        self._validate_buffer_pair(destination, source)
        if self.runtime_backend == "torch":
            self._decode_torch(source, destination)
            return
        self._decode_hqq_reference(source, destination)

    def _decode_hqq_reference(
        self, source: EncodedTensorBuffer, destination: Tensor
    ) -> None:
        assert source.scale is not None
        assert source.zero is not None
        packed_group_shape = (-1, self.group_size)
        meta = {
            "nbits": self.nbits,
            "group_size": self.group_size,
            "shape": torch.Size((destination.numel() // self.head_dim, self.head_dim)),
            "scale": source.scale.reshape(-1, 1),
            "zero": source.zero.reshape(-1, 1),
            "axis": 1,
            "packing": "8bit_u8",
            "unpack_view_dtype": torch.uint8,
            "view_as_float": False,
            "compute_dtype": self.compute_dtype,
        }
        decoded = self.quantizer.dequantize(
            source.data.reshape(packed_group_shape), meta
        )
        destination.copy_(decoded.reshape_as(destination))

    def _encode_torch(
        self, source: Tensor, destination: EncodedTensorBuffer
    ) -> None:
        """Allocation-reusing online form of HQQ's optimize=False equations."""
        assert destination.scale is not None
        assert destination.zero is not None
        groups = source.float().reshape(-1, self.group_size)
        minimum = groups.amin(dim=1, keepdim=True)
        maximum = groups.amax(dim=1, keepdim=True)
        denominator = maximum - minimum
        inverse_scale = torch.where(
            denominator.abs() <= 1e-4,
            torch.ones_like(denominator),
            255.0 / denominator,
        ).clamp_(max=2e4)
        zero = -minimum * inverse_scale
        quantized = (
            (groups * inverse_scale + zero)
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)
        )
        destination.data.copy_(quantized.reshape_as(destination.data))
        destination.scale.copy_(
            (1.0 / inverse_scale).reshape_as(destination.scale)
        )
        destination.zero.copy_(zero.reshape_as(destination.zero))

    def _decode_torch(
        self, source: EncodedTensorBuffer, destination: Tensor
    ) -> None:
        """Dequantize directly into the reusable FP16 attention slot."""
        assert source.scale is not None
        assert source.zero is not None
        output_groups = destination.reshape(-1, self.group_size)
        qgroups = source.data.reshape(-1, self.group_size)
        scale = source.scale.reshape(-1, 1)
        zero = source.zero.reshape(-1, 1)
        torch.sub(qgroups, zero, out=output_groups)
        output_groups.mul_(scale)

    def bytes_for_shape(self, shape: tuple[int, int, int, int]) -> int:
        self._validate_shape(shape)
        elements = 1
        for dimension in shape:
            elements *= dimension
        groups = elements // self.group_size
        q_bytes = elements * torch.empty((), dtype=torch.uint8).element_size()
        metadata_element_size = torch.empty((), dtype=self.metadata_dtype).element_size()
        return q_bytes + 2 * groups * metadata_element_size

    def _validate_shape(self, shape: tuple[int, int, int, int]) -> None:
        if len(shape) != 4:
            raise ValueError("encoded K/V shape must be [batch, sequence, heads, head_dim]")
        if shape[-1] != self.head_dim:
            raise ValueError(
                f"last dimension must be head_dim {self.head_dim}, got {shape[-1]}"
            )

    def _validate_buffer_pair(
        self, floating: Tensor, encoded: EncodedTensorBuffer
    ) -> None:
        self._validate_shape(tuple(floating.shape))
        if floating.dtype != self.compute_dtype:
            raise ValueError(
                f"floating K/V must use {self.compute_dtype}, got {floating.dtype}"
            )
        if encoded.data.shape != floating.shape or encoded.data.dtype != torch.uint8:
            raise ValueError("INT8 data buffer shape or dtype mismatch")
        if encoded.scale is None or encoded.zero is None:
            raise ValueError("INT8 storage requires scale and zero buffers")
        expected_metadata_shape = floating.shape[:-1] + (self.groups_per_head,)
        if encoded.scale.shape != expected_metadata_shape:
            raise ValueError("INT8 scale buffer shape mismatch")
        if encoded.zero.shape != expected_metadata_shape:
            raise ValueError("INT8 zero buffer shape mismatch")
        if encoded.scale.dtype != self.metadata_dtype:
            raise ValueError("INT8 scale buffer dtype mismatch")
        if encoded.zero.dtype != self.metadata_dtype:
            raise ValueError("INT8 zero buffer dtype mismatch")


def make_kv_codec(
    storage_format: KVStorageFormat,
    *,
    compute_dtype: torch.dtype,
    head_dim: int,
    group_size: int = 64,
    hqq_optimize: bool = False,
    hqq_runtime_backend: HQQRuntimeBackend = "torch",
) -> KVCodec:
    if storage_format == "fp16":
        return FP16KVCodec(compute_dtype=compute_dtype, head_dim=head_dim)
    if storage_format == "int8_hqq":
        return HQQInt8KVCodec(
            compute_dtype=compute_dtype,
            head_dim=head_dim,
            group_size=group_size,
            optimize=hqq_optimize,
            runtime_backend=hqq_runtime_backend,
        )
    raise ValueError("storage_format must be 'fp16' or 'int8_hqq'")
