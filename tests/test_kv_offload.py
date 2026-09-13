import pytest
import torch

from ondeviceinfer.cache import ResidentKVCache
from ondeviceinfer.kv_offload import OffloadedKVCache


def make_cache(**overrides):
    kwargs = {
        "num_layers": 2,
        "batch_size": 1,
        "num_key_value_heads": 2,
        "max_seq_len": 8,
        "head_dim": 4,
        "device": "cpu",
        "dtype": torch.float32,
        "mode": "sync",
        "pin_memory": False,
    }
    kwargs.update(overrides)
    return OffloadedKVCache(**kwargs)


def run_forward(cache, per_layer_states):
    outputs = []
    cache.begin_forward()
    for layer_idx, (key, value) in enumerate(per_layer_states):
        cache.before_layer(layer_idx)
        full_key, full_value = cache.update(layer_idx, key, value)
        outputs.append((full_key.clone(), full_value.clone()))
        cache.after_layer(layer_idx)
    cache.end_forward()
    return outputs


def test_sync_offload_matches_resident_cache_contents_and_lengths():
    offloaded = make_cache()
    resident = ResidentKVCache(
        num_layers=2,
        batch_size=1,
        num_key_value_heads=2,
        max_seq_len=8,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
    )

    prefill = [
        (
            torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4) + 100 * layer,
            torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4) + 1000 + 100 * layer,
        )
        for layer in range(2)
    ]
    actual_prefill = run_forward(offloaded, prefill)
    for layer_idx, (key, value) in enumerate(prefill):
        expected_key, expected_value = resident.update(layer_idx, key, value)
        assert torch.equal(actual_prefill[layer_idx][0], expected_key)
        assert torch.equal(actual_prefill[layer_idx][1], expected_value)

    decode = [
        (
            torch.full((1, 2, 1, 4), 2000.0 + layer),
            torch.full((1, 2, 1, 4), 3000.0 + layer),
        )
        for layer in range(2)
    ]
    actual_decode = run_forward(offloaded, decode)
    for layer_idx, (key, value) in enumerate(decode):
        expected_key, expected_value = resident.update(layer_idx, key, value)
        assert torch.equal(actual_decode[layer_idx][0], expected_key)
        assert torch.equal(actual_decode[layer_idx][1], expected_value)

    assert offloaded.lengths == [3, 3]
    assert [offloaded.get_seq_length(i) for i in range(2)] == [3, 3]
    assert offloaded.key_host[0].shape == (1, 8, 2, 4)
    assert torch.equal(
        offloaded.key_host[0][:, :3].transpose(1, 2), actual_decode[0][0]
    )


def test_transfer_and_managed_byte_accounting_counts_only_new_d2h_slices():
    cache = make_cache()
    bytes_per_token_all_kv = 2 * 1 * 2 * 4 * 4

    prefill = [
        (torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))
        for _ in range(2)
    ]
    run_forward(cache, prefill)
    assert cache.stats.h2d_bytes == 0
    assert cache.stats.d2h_bytes == 2 * 2 * bytes_per_token_all_kv

    decode = [
        (torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
        for _ in range(2)
    ]
    run_forward(cache, decode)

    expected_history_h2d = 2 * 2 * bytes_per_token_all_kv
    expected_total_d2h = (2 * 2 + 2 * 1) * bytes_per_token_all_kv
    assert cache.stats.h2d_bytes == expected_history_h2d
    assert cache.stats.d2h_bytes == expected_total_d2h
    assert cache.stats.cpu_to_gpu_copies == 2
    assert cache.stats.gpu_to_cpu_copies == 4
    assert cache.stats.completed_forwards == 2

    expected_layer_capacity = 8 * bytes_per_token_all_kv
    assert cache.layer_capacity_bytes == expected_layer_capacity
    assert cache.managed_cpu_bytes == 2 * expected_layer_capacity
    assert cache.managed_device_bytes == expected_layer_capacity
    assert cache.managed_gpu_bytes == 0
    assert cache.storage_bytes == 3 * expected_layer_capacity


def test_budget_validation_and_prefetch_fallback_are_explicit():
    prototype = make_cache()
    layer_bytes = prototype.layer_capacity_bytes

    with pytest.raises(ValueError, match="one complete layer"):
        make_cache(gpu_budget_bytes=layer_bytes - 1)

    budget_fallback = make_cache(
        mode="prefetch", gpu_budget_bytes=layer_bytes
    )
    assert budget_fallback.requested_mode == "prefetch"
    assert budget_fallback.effective_mode == "sync"
    assert "two complete layer slots" in budget_fallback.fallback_reason
    assert budget_fallback.slot_count == 1

    cpu_fallback = make_cache(
        mode="prefetch", gpu_budget_bytes=2 * layer_bytes
    )
    assert cpu_fallback.effective_mode == "sync"
    assert "CPU emulation" in cpu_fallback.fallback_reason


def test_capacity_shape_and_dtype_validation():
    cache = make_cache(max_seq_len=2)
    good = torch.zeros(1, 2, 2, 4)
    cache.update(0, good, good)

    with pytest.raises(ValueError, match="capacity exceeded"):
        cache.update(0, good[:, :, :1], good[:, :, :1])

    wrong_heads = torch.zeros(1, 1, 1, 4)
    with pytest.raises(ValueError, match="shape mismatch"):
        make_cache().update(0, wrong_heads, wrong_heads)

    wrong_dtype = torch.zeros(1, 2, 1, 4, dtype=torch.float16)
    with pytest.raises(ValueError, match="cache dtype"):
        make_cache().update(0, wrong_dtype, wrong_dtype)


def test_hooks_are_optional_and_reset_clears_lengths_and_stats():
    cache = make_cache()
    for layer_idx in range(2):
        states = torch.full((1, 2, 1, 4), float(layer_idx + 1))
        keys, values = cache.update(layer_idx, states, states)
        assert torch.equal(keys, states)
        assert torch.equal(values, states)

    # The final implicit layer is conservatively completed by synchronize.
    cache.synchronize()
    assert cache.lengths == [1, 1]
    assert cache.stats.update_calls == 2
    cache.reset()
    assert cache.lengths == [0, 0]
    assert cache.stats == type(cache.stats)()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_prefetch_events_prevent_slot_overwrite():
    device = torch.device("cuda")
    num_layers = 4
    max_seq_len = 4
    cache = OffloadedKVCache(
        num_layers=num_layers,
        batch_size=1,
        num_key_value_heads=2,
        max_seq_len=max_seq_len,
        head_dim=4,
        device=device,
        dtype=torch.float32,
        mode="prefetch",
        gpu_budget_bytes=2 * (2 * 1 * max_seq_len * 2 * 4 * 4),
    )
    observed = []
    cache.begin_forward()
    for layer_idx in range(num_layers):
        cache.before_layer(layer_idx)
        key = torch.full((1, 2, 2, 4), float(layer_idx + 1), device=device)
        full_key, _ = cache.update(layer_idx, key, key)
        # This asynchronous consumer must finish before the slot is refilled
        # two layers later.
        observed.append(full_key.sum())
        cache.after_layer(layer_idx)
    cache.end_forward()
    cache.synchronize()

    assert [value.item() for value in observed] == [16.0, 32.0, 48.0, 64.0]
    for layer_idx in range(num_layers):
        expected = torch.full((1, 2, 2, 4), float(layer_idx + 1))
        actual = cache.key_host[layer_idx][:, :2].transpose(1, 2)
        assert torch.equal(actual, expected)
