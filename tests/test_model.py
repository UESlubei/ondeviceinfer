import torch
import torch.nn.functional as F

from ondeviceinfer.attention import Qwen3Attention, build_causal_mask
from ondeviceinfer.cache import ResidentKVCache
from ondeviceinfer.config import Qwen3Config
from ondeviceinfer.layers import SwiGLU
from ondeviceinfer.model import Qwen3DecoderLayer, Qwen3ForCausalLM
from ondeviceinfer.functional import decoder_layer_forward
from ondeviceinfer.layers import RotaryEmbedding


def tiny_config(num_hidden_layers=2):
    return Qwen3Config(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        tie_word_embeddings=False,
    )


def test_cached_causal_mask_uses_absolute_query_positions():
    mask = build_causal_mask(2, 5, device=torch.device("cpu"))
    expected = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_sdpa_gqa_matches_explicit_kv_repetition():
    torch.manual_seed(1)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    v = torch.randn(2, 2, 5, 8)

    actual = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    expected = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        is_causal=True,
    )
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_attention_shapes_with_different_q_and_kv_head_counts():
    config = tiny_config(num_hidden_layers=1)
    attention = Qwen3Attention(config, layer_idx=0)
    hidden = torch.randn(2, 5, config.hidden_size)
    positions = torch.arange(5).expand(2, -1)
    # Position zero is sufficient here; RoPE construction is tested separately.
    frequencies = positions.unsqueeze(-1) * torch.ones(config.head_dim // 2)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    output = attention(hidden, angles.cos(), angles.sin())
    assert output.shape == hidden.shape


def test_swiglu_shape_and_formula():
    torch.manual_seed(2)
    layer = SwiGLU(8, 16)
    x = torch.randn(2, 3, 8)
    expected = layer.down_proj(F.silu(layer.gate_proj(x)) * layer.up_proj(x))
    assert torch.equal(layer(x), expected)
    assert layer(x).shape == x.shape


def test_resident_cache_appends_without_reallocation():
    cache = ResidentKVCache(
        num_layers=2,
        batch_size=1,
        num_key_value_heads=2,
        max_seq_len=8,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
    )
    storage_pointer = cache.key_cache[0].data_ptr()
    first = torch.randn(1, 2, 3, 4)
    second = torch.randn(1, 2, 1, 4)

    keys, _ = cache.update(0, first, first)
    keys, _ = cache.update(0, second, second)

    assert cache.get_seq_length(0) == 4
    assert cache.get_seq_length(1) == 0
    assert cache.key_cache[0].data_ptr() == storage_pointer
    assert torch.equal(keys[:, :, :3], first)
    assert torch.equal(keys[:, :, 3:], second)


def test_tiny_model_prefill_then_decode_updates_every_layer_cache():
    torch.manual_seed(3)
    config = tiny_config()
    model = Qwen3ForCausalLM(config).eval()
    cache = model.make_resident_cache(batch_size=1, max_seq_len=16)
    prompt = torch.tensor([[1, 2, 3, 4]])

    prefill = model(prompt, cache=cache, logits_to_keep=1)
    assert prefill.logits.shape == (1, 1, config.vocab_size)
    assert cache.lengths == [4, 4]

    token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    decode = model(token, cache=cache, logits_to_keep=1)
    assert decode.logits.shape == (1, 1, config.vocab_size)
    assert cache.lengths == [5, 5]


def test_functional_decoder_matches_module_decoder():
    torch.manual_seed(4)
    config = tiny_config(num_hidden_layers=1)
    layer = Qwen3DecoderLayer(config, layer_idx=0).eval()
    weights = dict(layer.state_dict())
    hidden = torch.randn(1, 6, config.hidden_size)
    positions = torch.arange(6).unsqueeze(0)
    cos, sin = RotaryEmbedding(config.head_dim, config.rope_theta)(hidden, positions)

    module_output = layer(hidden, cos, sin)
    functional_output = decoder_layer_forward(
        hidden, weights, config, 0, cos, sin
    )

    assert torch.allclose(module_output, functional_output, atol=1e-6, rtol=1e-6)
