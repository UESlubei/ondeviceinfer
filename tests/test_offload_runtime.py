import torch

from ondeviceinfer.config import Qwen3Config
from ondeviceinfer.runtime.offload import (
    CpuWeightStore,
    estimate_gpu_weight_bytes,
    layer_bytes,
)


def test_gpu_weight_estimate_counts_layer_slots_and_lm_head():
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10_000,
        tie_word_embeddings=False,
    )
    layer = {"weight": torch.empty(100, dtype=torch.float16)}
    store = CpuWeightStore(
        config=config,
        layers=[layer, layer],
        embed_tokens=torch.empty(16, 8, dtype=torch.float16),
        final_norm=torch.empty(8, dtype=torch.float16),
        lm_head=torch.empty(16, 8, dtype=torch.float16),
        dtype=torch.float16,
    )
    expected = 2 * layer_bytes(layer) + store.final_norm.nbytes + store.lm_head.nbytes
    assert estimate_gpu_weight_bytes(store, 2) == expected
