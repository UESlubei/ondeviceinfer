"""Inference-only symmetric INT8 quantization for linear weights.

The checkpoint format used here keeps one scale for every output channel
(row) of a linear weight.  Runtime execution deliberately dequantizes into a
caller-owned FP16 scratch buffer before calling :func:`torch.nn.functional.linear`.
This is not an INT8 GEMM; it is the simple, auditable memory-saving baseline
used before introducing custom quantized kernels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import prod
from typing import Any, Self

import torch
import torch.nn.functional as F
from torch import Tensor


QUANTIZATION_FORMAT = "int8_per_output_channel_symmetric"
QUANTIZATION_VERSION = 1
QMIN = -127
QMAX = 127


def _state_key(prefix: str, suffix: str) -> str:
    prefix = prefix.rstrip(".")
    return f"{prefix}.{suffix}" if prefix else suffix


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True, slots=True)
class QuantizedLinearWeight:
    """A two-dimensional linear weight quantized independently by row.

    ``qweight`` has shape ``[out_features, in_features]`` and dtype INT8.
    ``scale`` has shape ``[out_features]`` and contains strictly positive
    floating-point values.  A row is reconstructed as
    ``qweight[row].to(fp16) * scale[row]``; symmetric quantization therefore
    has an implicit zero point of zero.
    """

    qweight: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        if self.qweight.ndim != 2:
            raise ValueError("qweight must have shape [out_features, in_features]")
        if self.qweight.dtype != torch.int8:
            raise TypeError("qweight must have dtype torch.int8")
        if self.scale.ndim != 1 or self.scale.shape[0] != self.qweight.shape[0]:
            raise ValueError("scale must have shape [out_features]")
        if not self.scale.is_floating_point():
            raise TypeError("scale must use a floating-point dtype")
        if self.qweight.device != self.scale.device:
            raise ValueError("qweight and scale must be on the same device")
        if not bool(torch.isfinite(self.scale).all()):
            raise ValueError("scale must contain only finite values")
        if not bool((self.scale > 0).all()):
            raise ValueError("scale values must be strictly positive")

    @property
    def shape(self) -> torch.Size:
        return self.qweight.shape

    @property
    def out_features(self) -> int:
        return self.qweight.shape[0]

    @property
    def in_features(self) -> int:
        return self.qweight.shape[1]

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    @property
    def storage_nbytes(self) -> int:
        """Bytes occupied by the persistent INT8 weight and its scales."""
        return self.qweight.nbytes + self.scale.nbytes

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """Copy both persistent tensors to ``device`` without changing dtypes."""
        return type(self)(
            qweight=self.qweight.to(device=device, non_blocking=non_blocking),
            scale=self.scale.to(device=device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> Self:
        """Return a pinned-CPU copy suitable for asynchronous H2D transfer."""
        if self.device.type != "cpu":
            raise ValueError("only CPU quantized weights can be pinned")
        return type(self)(
            qweight=self.qweight.pin_memory(),
            scale=self.scale.pin_memory(),
        )

    def tensor_dict(self, prefix: str = "") -> dict[str, Tensor]:
        """Return the two tensors under stable checkpoint/manifest keys."""
        return {
            _state_key(prefix, "qweight"): self.qweight,
            _state_key(prefix, "scale"): self.scale,
        }

    @classmethod
    def from_tensor_dict(
        cls,
        tensors: Mapping[str, Tensor],
        prefix: str = "",
    ) -> Self:
        """Reconstruct an instance from :meth:`tensor_dict` output."""
        return cls(
            qweight=tensors[_state_key(prefix, "qweight")],
            scale=tensors[_state_key(prefix, "scale")],
        )

    def manifest_entry(self, prefix: str = "") -> dict[str, Any]:
        """Build a JSON-serializable description of this quantized tensor."""
        return {
            "format": QUANTIZATION_FORMAT,
            "version": QUANTIZATION_VERSION,
            "axis": 0,
            "zero_point": 0,
            "quant_min": QMIN,
            "quant_max": QMAX,
            "shape": list(self.shape),
            "qweight_key": _state_key(prefix, "qweight"),
            "scale_key": _state_key(prefix, "scale"),
            "qweight_dtype": _dtype_name(self.qweight.dtype),
            "scale_dtype": _dtype_name(self.scale.dtype),
        }

    def dequantize(
        self,
        scratch: FP16DequantScratch | Tensor,
    ) -> Tensor:
        """Dequantize into caller-owned FP16 storage."""
        return dequantize_linear_weight(self, scratch)

    def linear(
        self,
        input: Tensor,
        scratch: FP16DequantScratch | Tensor,
        bias: Tensor | None = None,
    ) -> Tensor:
        """Dequantize into ``scratch`` and apply an FP16 linear projection."""
        return quantized_linear(input, self, scratch, bias=bias)


@dataclass(slots=True)
class FP16DequantScratch:
    """Reusable flat FP16 storage for one dequantized weight at a time.

    One buffer sized for the largest matrix can serve differently shaped
    projections sequentially.  On CUDA, reuse is safe on the same stream;
    cross-stream users must add their own stream/event synchronization.
    """

    buffer: Tensor

    def __post_init__(self) -> None:
        if self.buffer.dtype != torch.float16:
            raise TypeError("dequantization scratch must have dtype torch.float16")
        if self.buffer.ndim != 1:
            raise ValueError("dequantization scratch must be a flat tensor")
        if not self.buffer.is_contiguous():
            raise ValueError("dequantization scratch must be contiguous")

    @classmethod
    def allocate(
        cls,
        capacity: int,
        *,
        device: torch.device | str,
    ) -> Self:
        if capacity <= 0:
            raise ValueError("scratch capacity must be positive")
        return cls(torch.empty(capacity, dtype=torch.float16, device=device))

    @classmethod
    def for_weights(
        cls,
        weights: Iterable[QuantizedLinearWeight],
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        """Allocate once for the largest matrix in ``weights``."""
        weights = tuple(weights)
        if not weights:
            raise ValueError("at least one weight is required")
        target_device = torch.device(device) if device is not None else weights[0].device
        return cls.allocate(
            max(weight.qweight.numel() for weight in weights),
            device=target_device,
        )

    @property
    def capacity(self) -> int:
        return self.buffer.numel()

    @property
    def device(self) -> torch.device:
        return self.buffer.device

    def view_for(self, shape: Iterable[int]) -> Tensor:
        shape = tuple(shape)
        required = prod(shape)
        if required > self.capacity:
            raise ValueError(
                f"scratch capacity {self.capacity} is smaller than required {required}"
            )
        return self.buffer[:required].view(shape)


@torch.no_grad()
def quantize_linear_weight(
    weight: Tensor,
    *,
    row_block_size: int = 256,
    scale_dtype: torch.dtype = torch.float16,
) -> QuantizedLinearWeight:
    """Quantize ``[out, in]`` weight rows in bounded-memory blocks.

    Quantization arithmetic is performed in FP32.  The scale actually stored
    in ``scale_dtype`` is also the scale used to produce the integer values,
    so serialization and later dequantization reproduce the intended weight.
    Zero rows receive scale 1 and an all-zero integer row.  If a nonzero scale
    underflows in the chosen scale dtype, it is replaced by that dtype's
    smallest positive representable value.
    """
    if weight.ndim != 2:
        raise ValueError("linear weight must have shape [out_features, in_features]")
    if not weight.is_floating_point():
        raise TypeError("linear weight must use a floating-point dtype")
    if weight.shape[0] == 0 or weight.shape[1] == 0:
        raise ValueError("linear weight dimensions must be nonzero")
    if row_block_size <= 0:
        raise ValueError("row_block_size must be positive")
    if not torch.empty((), dtype=scale_dtype).is_floating_point():
        raise TypeError("scale_dtype must be floating point")

    out_features = weight.shape[0]
    qweight = torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
    scales = torch.empty(out_features, dtype=scale_dtype, device=weight.device)

    for row_start in range(0, out_features, row_block_size):
        row_end = min(row_start + row_block_size, out_features)
        block = weight[row_start:row_end].to(torch.float32)
        if not bool(torch.isfinite(block).all()):
            raise ValueError("linear weight contains NaN or infinity")

        absmax = block.abs().amax(dim=1)
        raw_scale = absmax / QMAX
        raw_scale = torch.where(absmax == 0, torch.ones_like(raw_scale), raw_scale)
        stored_scale = raw_scale.to(scale_dtype)

        # FP16 can underflow for unusually tiny rows.  Keep every scale valid
        # and nonzero; such a row will safely quantize to zero if appropriate.
        underflow = (absmax != 0) & (stored_scale == 0)
        if bool(underflow.any()):
            zero = torch.zeros((), dtype=scale_dtype, device=weight.device)
            one = torch.ones((), dtype=scale_dtype, device=weight.device)
            smallest_positive = torch.nextafter(zero, one)
            stored_scale = torch.where(underflow, smallest_positive, stored_scale)
        if not bool(torch.isfinite(stored_scale).all()):
            raise OverflowError("row scale is not representable in scale_dtype")

        scale_fp32 = stored_scale.to(torch.float32)
        quantized = torch.round(block / scale_fp32[:, None])
        quantized.clamp_(QMIN, QMAX)
        qweight[row_start:row_end].copy_(quantized.to(torch.int8))
        scales[row_start:row_end].copy_(stored_scale)

    return QuantizedLinearWeight(qweight=qweight.contiguous(), scale=scales.contiguous())


def quantize_per_output_channel_int8(
    weight: Tensor,
    *,
    row_block_size: int = 256,
    scale_dtype: torch.dtype = torch.float16,
) -> QuantizedLinearWeight:
    """Explicitly named alias for :func:`quantize_linear_weight`."""
    return quantize_linear_weight(
        weight,
        row_block_size=row_block_size,
        scale_dtype=scale_dtype,
    )


def _scratch_view(
    scratch: FP16DequantScratch | Tensor,
    shape: torch.Size,
) -> Tensor:
    if isinstance(scratch, FP16DequantScratch):
        return scratch.view_for(shape)
    if scratch.dtype != torch.float16:
        raise TypeError("dequantization scratch must have dtype torch.float16")
    if not scratch.is_contiguous():
        raise ValueError("dequantization scratch must be contiguous")
    required = prod(shape)
    if scratch.numel() < required:
        raise ValueError(
            f"scratch capacity {scratch.numel()} is smaller than required {required}"
        )
    return scratch.view(-1)[:required].view(shape)


@torch.no_grad()
def dequantize_linear_weight(
    weight: QuantizedLinearWeight,
    scratch: FP16DequantScratch | Tensor,
) -> Tensor:
    """Reconstruct ``weight`` directly into reusable FP16 ``scratch``."""
    output = _scratch_view(scratch, weight.shape)
    if output.device != weight.device:
        raise ValueError("scratch and quantized weight must be on the same device")
    torch.mul(weight.qweight, weight.scale[:, None], out=output)
    return output


@torch.no_grad()
def quantized_linear(
    input: Tensor,
    weight: QuantizedLinearWeight,
    scratch: FP16DequantScratch | Tensor,
    *,
    bias: Tensor | None = None,
) -> Tensor:
    """Apply a quantized weight using scratch dequantization plus ``F.linear``."""
    if input.dtype != torch.float16:
        raise TypeError("quantized_linear expects FP16 input activations")
    if input.device != weight.device:
        raise ValueError("input and quantized weight must be on the same device")
    if input.shape[-1] != weight.in_features:
        raise ValueError(
            f"input has {input.shape[-1]} features, expected {weight.in_features}"
        )
    if bias is not None and (bias.dtype != torch.float16 or bias.device != input.device):
        raise ValueError("bias must be FP16 and on the input device")
    dequantized = dequantize_linear_weight(weight, scratch)
    return F.linear(input, dequantized, bias)


__all__ = [
    "FP16DequantScratch",
    "QMAX",
    "QMIN",
    "QUANTIZATION_FORMAT",
    "QUANTIZATION_VERSION",
    "QuantizedLinearWeight",
    "dequantize_linear_weight",
    "quantize_linear_weight",
    "quantize_per_output_channel_int8",
    "quantized_linear",
]
