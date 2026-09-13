import pytest
import torch
import torch.nn.functional as F

from ondeviceinfer.quantization import (
    FP16DequantScratch,
    QUANTIZATION_FORMAT,
    QuantizedLinearWeight,
    dequantize_linear_weight,
    quantize_linear_weight,
    quantized_linear,
)


DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


@pytest.mark.parametrize("device", DEVICES, ids=lambda device: device.type)
def test_per_output_channel_error_and_row_block_invariance(device):
    generator = torch.Generator(device=device).manual_seed(7)
    weight = torch.randn(9, 37, generator=generator, device=device) * torch.linspace(
        0.02, 2.0, 9, device=device
    )[:, None]

    blocked = quantize_linear_weight(weight, row_block_size=2)
    unblocked = quantize_linear_weight(weight, row_block_size=weight.shape[0])

    assert torch.equal(blocked.qweight, unblocked.qweight)
    assert torch.equal(blocked.scale, unblocked.scale)
    assert blocked.qweight.min().item() >= -127
    assert blocked.qweight.max().item() <= 127

    scratch = FP16DequantScratch.allocate(weight.numel(), device=device)
    reconstructed = blocked.dequantize(scratch).float()
    per_row_error = (weight - reconstructed).abs().amax(dim=1)
    # Half-precision scale rounding adds a small amount above the ideal scale/2.
    assert torch.all(per_row_error <= blocked.scale.float() * 0.6 + 2e-6)


@pytest.mark.parametrize("device", DEVICES, ids=lambda device: device.type)
def test_zero_and_tiny_rows_have_safe_nonzero_scales(device):
    weight = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1e-10, -1e-10, 0.0, 1e-10],
            [1.0, -0.5, 0.25, 0.0],
        ],
        device=device,
    )
    quantized = quantize_linear_weight(weight, row_block_size=1)
    scratch = FP16DequantScratch.allocate(weight.numel(), device=device)
    reconstructed = dequantize_linear_weight(quantized, scratch)

    assert torch.count_nonzero(quantized.qweight[0]).item() == 0
    assert torch.count_nonzero(reconstructed[0]).item() == 0
    assert quantized.scale[0].item() == 1.0
    assert torch.isfinite(quantized.scale).all()
    assert torch.all(quantized.scale > 0)
    assert torch.isfinite(reconstructed).all()


@pytest.mark.parametrize("device", DEVICES, ids=lambda device: device.type)
def test_one_flat_scratch_is_reused_across_weight_shapes(device):
    first = quantize_linear_weight(torch.randn(5, 8, device=device))
    second = quantize_linear_weight(torch.randn(3, 8, device=device))
    scratch = FP16DequantScratch.for_weights([first, second], device=device)
    storage_pointer = scratch.buffer.data_ptr()

    first_result = first.dequantize(scratch)
    first_copy = first_result.clone()
    second_result = second.dequantize(scratch)

    assert scratch.buffer.data_ptr() == storage_pointer
    assert first_result.data_ptr() == storage_pointer
    assert second_result.data_ptr() == storage_pointer
    assert torch.equal(first.dequantize(scratch), first_copy)


@pytest.mark.parametrize("device", DEVICES, ids=lambda device: device.type)
def test_quantized_linear_matches_explicit_dequantized_linear(device):
    generator = torch.Generator(device=device).manual_seed(19)
    weight = torch.randn(11, 13, generator=generator, device=device) * 0.2
    input = torch.randn(
        2, 3, 13, generator=generator, device=device, dtype=torch.float16
    )
    bias = torch.randn(11, generator=generator, device=device, dtype=torch.float16)
    quantized = quantize_linear_weight(weight)
    scratch = FP16DequantScratch.allocate(weight.numel(), device=device)

    dequantized = quantized.dequantize(scratch).clone()
    expected = F.linear(input, dequantized, bias)
    actual = quantized_linear(input, quantized, scratch, bias=bias)

    assert torch.equal(actual, expected)
    assert torch.allclose(actual, F.linear(input, weight.half(), bias), atol=2e-2, rtol=2e-2)


def test_tensor_and_manifest_interfaces_round_trip():
    quantized = quantize_linear_weight(torch.randn(4, 6))
    tensors = quantized.tensor_dict("model.layers.0.self_attn.q_proj.weight")
    restored = QuantizedLinearWeight.from_tensor_dict(
        tensors, "model.layers.0.self_attn.q_proj.weight"
    )
    manifest = quantized.manifest_entry("model.layers.0.self_attn.q_proj.weight")

    assert restored.qweight is quantized.qweight
    assert restored.scale is quantized.scale
    assert manifest["format"] == QUANTIZATION_FORMAT
    assert manifest["axis"] == 0
    assert manifest["shape"] == [4, 6]
    assert manifest["qweight_key"].endswith(".qweight")
    assert manifest["scale_key"].endswith(".scale")
