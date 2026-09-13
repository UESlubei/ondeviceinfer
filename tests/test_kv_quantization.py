import pytest
import torch
import torch.nn.functional as F

from ondeviceinfer.kv_offload import OffloadedKVCache
from ondeviceinfer.kv_quantization import HQQInt8KVCodec


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0)
    )


def test_hqq_int8_codec_layout_bytes_and_round_trip():
    torch.manual_seed(7)
    shape = (1, 3, 2, 128)
    codec = HQQInt8KVCodec(
        compute_dtype=torch.float32,
        head_dim=128,
        group_size=64,
    )
    source = torch.randn(shape)
    encoded = codec.allocate(shape, device="cpu")
    reconstructed = torch.empty_like(source)

    codec.encode_into(source, encoded)
    codec.decode_into(encoded, reconstructed)

    assert encoded.data.dtype == torch.uint8
    assert encoded.scale is not None and encoded.scale.dtype == torch.float16
    assert encoded.zero is not None and encoded.zero.dtype == torch.float16
    assert encoded.scale.shape == (1, 3, 2, 2)
    assert encoded.zero.shape == (1, 3, 2, 2)
    # 768 UINT8 values + 12 groups * (FP16 scale + FP16 zero).
    assert encoded.storage_bytes == 768 + 12 * 4
    assert codec.bytes_for_shape(shape) == encoded.storage_bytes
    assert cosine(source, reconstructed) >= 0.999
    assert torch.mean((source - reconstructed) ** 2).item() < 1e-3


def test_hqq_int8_codec_handles_constant_groups_without_nan():
    codec = HQQInt8KVCodec(
        compute_dtype=torch.float32,
        head_dim=128,
        group_size=64,
    )
    source = torch.full((1, 2, 1, 128), 0.125)
    encoded = codec.allocate(tuple(source.shape), device="cpu")
    reconstructed = torch.empty_like(source)

    codec.encode_into(source, encoded)
    codec.decode_into(encoded, reconstructed)

    assert torch.isfinite(reconstructed).all()
    assert torch.allclose(source, reconstructed, atol=2e-4, rtol=0)


def test_online_torch_backend_matches_hqq_reference_quantizer():
    torch.manual_seed(9)
    source = torch.randn(1, 5, 2, 128, dtype=torch.float16)
    online = HQQInt8KVCodec(
        compute_dtype=torch.float16,
        head_dim=128,
        group_size=64,
        runtime_backend="torch",
    )
    reference = HQQInt8KVCodec(
        compute_dtype=torch.float16,
        head_dim=128,
        group_size=64,
        runtime_backend="hqq_reference",
    )
    online_buffer = online.allocate(tuple(source.shape), device="cpu")
    reference_buffer = reference.allocate(tuple(source.shape), device="cpu")
    online.encode_into(source, online_buffer)
    reference.encode_into(source, reference_buffer)

    assert torch.equal(online_buffer.data, reference_buffer.data)
    assert torch.equal(online_buffer.scale, reference_buffer.scale)
    assert torch.equal(online_buffer.zero, reference_buffer.zero)
    online_output = torch.empty_like(source)
    reference_output = torch.empty_like(source)
    online.decode_into(online_buffer, online_output)
    reference.decode_into(reference_buffer, reference_output)
    assert torch.equal(online_output, reference_output)


def test_hqq_int8_codec_rejects_incompatible_group_size():
    with pytest.raises(ValueError, match="must be divisible"):
        HQQInt8KVCodec(
            compute_dtype=torch.float16,
            head_dim=128,
            group_size=96,
        )


def test_int8_sync_offload_round_trip_and_byte_accounting():
    torch.manual_seed(11)
    cache = OffloadedKVCache(
        num_layers=2,
        batch_size=1,
        num_key_value_heads=2,
        max_seq_len=8,
        head_dim=128,
        device="cpu",
        dtype=torch.float32,
        mode="sync",
        pin_memory=False,
        storage_format="int8_hqq",
        kv_group_size=64,
    )
    prefill = []
    cache.begin_forward()
    for layer_idx in range(2):
        key = torch.randn(1, 2, 3, 128)
        value = torch.randn_like(key)
        prefill.append((key.clone(), value.clone()))
        cache.before_layer(layer_idx)
        actual_key, actual_value = cache.update(layer_idx, key, value)
        # Newly produced K/V stays FP32/FP16 for the current attention call.
        assert torch.equal(actual_key, key)
        assert torch.equal(actual_value, value)
        cache.after_layer(layer_idx)
    cache.end_forward()

    # 3 tokens * 2 heads * 128 values, for K and V.  Every 64 values add
    # one FP16 scale and one FP16 zero.
    per_layer_prefill_payload = 2 * (3 * 2 * 128 + 3 * 2 * 2 * 4)
    assert cache.stats.d2h_bytes == 2 * per_layer_prefill_payload
    assert cache.stats.d2h_data_bytes == 2 * (2 * 3 * 2 * 128)
    assert cache.stats.d2h_metadata_bytes == 2 * (2 * 3 * 2 * 2 * 4)
    assert cache.stats.quantize_calls == 4
    assert cache.key_q_host[0].dtype == torch.uint8

    cache.begin_forward()
    for layer_idx in range(2):
        new_key = torch.randn(1, 2, 1, 128)
        new_value = torch.randn_like(new_key)
        cache.before_layer(layer_idx)
        actual_key, actual_value = cache.update(layer_idx, new_key, new_value)
        assert cosine(actual_key[:, :, :3], prefill[layer_idx][0]) >= 0.999
        assert cosine(actual_value[:, :, :3], prefill[layer_idx][1]) >= 0.999
        assert torch.equal(actual_key[:, :, 3:], new_key)
        assert torch.equal(actual_value[:, :, 3:], new_value)
        cache.after_layer(layer_idx)
    cache.end_forward()

    assert cache.lengths == [4, 4]
    assert cache.stats.dequantize_calls == 4
    assert cache.stats.quantize_calls == 8
    assert cache.host_compression_ratio == pytest.approx(0.265625)
    assert cache.slot_capacity_bytes == (
        cache.attention_layer_capacity_bytes + cache.encoded_layer_capacity_bytes
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_int8_prefetch_round_trip_and_slot_events():
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    cache = OffloadedKVCache(
        num_layers=4,
        batch_size=1,
        num_key_value_heads=2,
        max_seq_len=4,
        head_dim=128,
        device=device,
        dtype=torch.float16,
        mode="prefetch",
        pin_memory=True,
        storage_format="int8_hqq",
        kv_group_size=64,
    )
    expected = []
    cache.begin_forward()
    for layer_idx in range(4):
        key = torch.randn(1, 2, 2, 128, device=device, dtype=torch.float16)
        value = torch.randn_like(key)
        expected.append((key.cpu(), value.cpu()))
        cache.before_layer(layer_idx)
        full_key, full_value = cache.update(layer_idx, key, value)
        assert torch.equal(full_key, key)
        assert torch.equal(full_value, value)
        cache.after_layer(layer_idx)
    cache.end_forward()
    cache.synchronize()

    observed_sums = []
    cache.begin_forward()
    for layer_idx in range(4):
        key = torch.randn(1, 2, 1, 128, device=device, dtype=torch.float16)
        value = torch.randn_like(key)
        cache.before_layer(layer_idx)
        full_key, full_value = cache.update(layer_idx, key, value)
        assert cosine(full_key[:, :, :2].cpu(), expected[layer_idx][0]) >= 0.999
        assert cosine(full_value[:, :, :2].cpu(), expected[layer_idx][1]) >= 0.999
        # Keep an asynchronous consumer alive so slot i cannot be overwritten
        # when the transfer stream starts layer i+2.
        observed_sums.append(full_key.sum())
        cache.after_layer(layer_idx)
    cache.end_forward()
    cache.synchronize()

    assert all(torch.isfinite(value) for value in observed_sums)
    assert cache.effective_mode == "prefetch"
    assert cache.stats.quantize_calls == 16
    assert cache.stats.dequantize_calls == 8
    assert cache.stats.h2d_metadata_bytes > 0
    assert cache.stats.d2h_metadata_bytes > 0

    # A new request must not observe compressed pages or slot tags from the
    # previous epoch.  Physical buffers may be reused, but logical lengths and
    # asynchronous dependencies are reset before the next forward.
    cache.reset()
    assert cache.lengths == [0, 0, 0, 0]
    assert cache.stats == type(cache.stats)()
    cache.begin_forward()
    for layer_idx in range(4):
        key = torch.full(
            (1, 2, 1, 128),
            float(layer_idx + 10),
            device=device,
            dtype=torch.float16,
        )
        value = -key
        cache.before_layer(layer_idx)
        full_key, full_value = cache.update(layer_idx, key, value)
        assert full_key.shape[2] == 1
        assert full_value.shape[2] == 1
        assert torch.equal(full_key, key)
        assert torch.equal(full_value, value)
        cache.after_layer(layer_idx)
    cache.end_forward()
    cache.synchronize()
    assert cache.lengths == [1, 1, 1, 1]
