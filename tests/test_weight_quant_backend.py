import torch
from types import SimpleNamespace
from ondeviceinfer.weight_quantization.backend import decode_weight, unpack_unsigned, CompressedLinearBackend
from benchmarks.validate_weight_reconstruction import decode_rows


def test_unpack_bit_order_and_axes():
    packed = torch.tensor([[0x76543210]], dtype=torch.int32)
    expected = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    assert torch.equal(unpack_unsigned(packed, 4, 1, 8), expected)
    assert torch.equal(unpack_unsigned(packed, 4, 0, 8), expected.t())


def test_group_zero_point_and_scale_decode():
    # Unsigned nibbles 0..7, two output channels, two groups of four.
    packed = torch.tensor([[0x76543210], [0x76543210]], dtype=torch.int32)
    # Packed output-channel zero points: group0=[0,1], group1=[2,3].
    zero = torch.tensor([[0x10, 0x32]], dtype=torch.int32)
    data = {'mlp.up_proj.weight_packed': packed,
            'mlp.up_proj.weight_shape': torch.tensor([2, 8]),
            'mlp.up_proj.weight_scale': torch.tensor([[1., 2.], [3., 4.]]),
            'mlp.up_proj.weight_zero_point': zero}
    actual = decode_weight(data, 'mlp.up_proj.weight', group_size=4)
    expected = torch.tensor([[0,1,2,3,4,6,8,10], [-3,0,3,6,4,8,12,16]], dtype=torch.float16)
    assert torch.equal(actual, expected)


def test_symmetric_packed_bias_and_int8_channel_scale():
    data = {'self_attn.q_proj.weight_packed': torch.tensor([[0x76543210]], dtype=torch.int32),
            'self_attn.q_proj.weight_shape': torch.tensor([1,8]),
            'self_attn.q_proj.weight_scale': torch.ones((1,2))}
    assert torch.equal(decode_weight(data,'self_attn.q_proj.weight',group_size=4),torch.arange(-8,0,dtype=torch.float16).reshape(1,8))
    data = {'self_attn.q_proj.weight': torch.tensor([[1,-2],[3,-4]],dtype=torch.int8),
            'self_attn.q_proj.weight_scale': torch.tensor([[0.5],[2.]])}
    assert torch.equal(decode_weight(data,'self_attn.q_proj.weight'),torch.tensor([[0.5,-1],[6,-8]],dtype=torch.float16))


def test_sampled_weight_decode_matches_full_decode():
    prefix='mlp.up_proj'
    data={prefix+'.weight_packed':torch.full((8,32),0x76543210,dtype=torch.int32),
          prefix+'.weight_shape':torch.tensor([8,256]),
          prefix+'.weight_scale':torch.arange(1,17,dtype=torch.float16).reshape(8,2)/10,
          prefix+'.weight_zero_point':torch.tensor([[0x76543210,0x01234567]],dtype=torch.int32)}
    rows=torch.tensor([0,3,7])
    assert torch.equal(decode_rows(data,prefix+'.weight',rows),decode_weight(data,prefix+'.weight')[rows])


def test_marlin_safe_row_dispatch_preserves_shape_and_counts():
    backend=object.__new__(CompressedLinearBackend)
    backend.name='marlin';backend.calls=0;backend.gemm_calls=0;backend.marlin_max_rows_per_call=16
    widths=[]
    def fake_gemm(a,output,packed,*args,**kwargs):
        widths.append(a.shape[0])
        assert a.shape[0]<=16
        actual=a@packed.t()
        if output is not None:output.copy_(actual);return output
        return actual
    backend.ops=SimpleNamespace(gptq_marlin_gemm=fake_gemm)
    weight=torch.randn(8,4)
    x=torch.randn(2,12,4)
    weights={'mlp.up_proj._prepared':(weight,None,None,None,None,None,4,8)}
    output=backend.linear(x,weights,'mlp.up_proj.weight')
    assert output.shape==(2,12,8)
    assert torch.allclose(output,x@weight.t())
    assert widths==[16,8]
    assert backend.calls==1 and backend.gemm_calls==2
