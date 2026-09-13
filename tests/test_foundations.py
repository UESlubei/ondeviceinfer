import json

import pytest
import torch
import torch.nn.functional as F

from ondeviceinfer.config import Qwen3Config
from ondeviceinfer.layers import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


def test_config_loads_qwen3_and_derives_gqa_groups(tmp_path):
    raw = {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "tie_word_embeddings": True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = Qwen3Config.from_json(path)

    assert config.num_key_value_groups == 2
    assert config.hidden_size == config.num_attention_heads * config.head_dim


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_matches_pytorch(dtype):
    torch.manual_seed(0)
    x = torch.randn(2, 5, 128, dtype=dtype)
    layer = RMSNorm(128, eps=1e-6).to(dtype=dtype)
    layer.weight.data.uniform_(0.5, 1.5)

    actual = layer(x)
    # Match Qwen3's operation order: normalize in FP32, cast activations back,
    # then apply the learned scale in the model dtype.
    expected = F.rms_norm(x.float(), (128,), None, 1e-6).to(dtype) * layer.weight

    assert actual.dtype == dtype
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_rope_position_zero_is_identity_and_preserves_norm():
    torch.manual_seed(0)
    batch, q_heads, kv_heads, seq_len, head_dim = 2, 16, 8, 7, 128
    query = torch.randn(batch, q_heads, seq_len, head_dim)
    key = torch.randn(batch, kv_heads, seq_len, head_dim)
    positions = torch.arange(seq_len).expand(batch, -1)
    rope = RotaryEmbedding(head_dim=head_dim, theta=1_000_000)

    cos, sin = rope(query, positions)
    query_rotated, key_rotated = apply_rotary_pos_emb(query, key, cos, sin)

    assert cos.shape == (batch, seq_len, head_dim)
    assert torch.equal(query_rotated[:, :, 0], query[:, :, 0])
    assert torch.equal(key_rotated[:, :, 0], key[:, :, 0])
    assert torch.allclose(query_rotated.norm(dim=-1), query.norm(dim=-1), atol=1e-5)
    assert torch.allclose(key_rotated.norm(dim=-1), key.norm(dim=-1), atol=1e-5)
