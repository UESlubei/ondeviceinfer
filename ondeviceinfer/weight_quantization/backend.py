"""Project forwards over compressed-tensors INT4 and INT8 checkpoints.

vLLM's private CUDA interfaces are confined to this adapter.  Reference
execution is explicit and never a silent fallback from a fused backend.
"""
from __future__ import annotations

import json
import inspect
import importlib.metadata
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ..config import Qwen3Config
from ..profiling import nvtx_range
from ..runtime.offload import RuntimeStats
from ..runtime.quantized import LINEAR_WEIGHT_NAMES, NORM_WEIGHT_NAMES
from ..weights import SafetensorIndex


def unpack_unsigned(packed: torch.Tensor, bits: int, axis: int, length: int) -> torch.Tensor:
    """Recover unsigned values without relying on a compressor version."""
    value = packed.transpose(0, 1) if axis == 0 else packed
    shifts = torch.arange(0, 32, bits, device=value.device, dtype=torch.int32)
    unpacked = ((value.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(value.shape[0], -1)
    unpacked = unpacked[:, :length]
    return unpacked.transpose(0, 1) if axis == 0 else unpacked


def decode_weight(weights: dict, name: str, *, group_size: int = 128) -> torch.Tensor:
    prefix = name.removesuffix('.weight')
    if prefix + '.weight_packed' not in weights:
        q = weights[name]
        scale = weights[prefix + '.weight_scale']
        return (q.float() * scale.float().reshape(-1, 1)).half()
    n, k = (int(x) for x in weights[prefix + '.weight_shape'].tolist())
    q = unpack_unsigned(weights[prefix + '.weight_packed'], 4, 1, k)
    scale = weights[prefix + '.weight_scale'].float()
    zero = weights.get(prefix + '.weight_zero_point')
    if zero is None:
        zero = torch.full_like(scale, 8, dtype=torch.int32)
    else:
        zero = unpack_unsigned(zero, 4, 0, n)
    g_idx = weights.get(prefix + '.weight_g_idx')
    if g_idx is not None and g_idx.numel():
        groups = g_idx.long()
    else:
        groups = torch.arange(k, device=q.device) // group_size
    return ((q.float() - zero[:, groups].float()) * scale[:, groups]).half()


@dataclass
class CompressedStore:
    config: Qwen3Config
    layers: list[dict]
    embed_tokens: torch.Tensor
    final_norm: torch.Tensor
    lm_head: torch.Tensor
    manifest: dict
    compute_dtype: torch.dtype = torch.float16
    max_dequant_elements: int = 0


def load_compressed_store(path: Path) -> CompressedStore:
    index = SafetensorIndex(path)
    cfg = Qwen3Config.from_json(path / 'config.json')
    layers = [{} for _ in range(cfg.num_hidden_layers)]
    globals_ = {}
    for shard in index.shard_names:
        with safe_open(path / shard, framework='pt', device='cpu') as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if name.startswith('model.layers.'):
                    _, _, layer_index, local = name.split('.', 3)
                    layers[int(layer_index)][local] = tensor.contiguous()
                elif name in ('model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'):
                    globals_[name] = tensor.half().contiguous()
    for idx, layer in enumerate(layers):
        for name in NORM_WEIGHT_NAMES:
            if name not in layer:
                raise ValueError(f'missing layer {idx} norm {name}')
            layer[name] = layer[name].half()
            if not bool(torch.isfinite(layer[name]).all()):
                raise ValueError(f'non-finite layer {idx} norm {name}')
        for name in LINEAR_WEIGHT_NAMES:
            prefix = name.removesuffix('.weight')
            if prefix + '.weight_scale' not in layer:
                raise ValueError(f'missing layer {idx} scale {prefix}')
            scale = layer[prefix + '.weight_scale']
            if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                raise ValueError(f'non-finite or non-positive scale: layer {idx} {prefix}')
            hidden, intermediate = cfg.hidden_size, cfg.intermediate_size
            kv = cfg.num_key_value_heads * cfg.head_dim
            expected_shapes = {
                'self_attn.q_proj.weight': (hidden, hidden),
                'self_attn.k_proj.weight': (kv, hidden),
                'self_attn.v_proj.weight': (kv, hidden),
                'self_attn.o_proj.weight': (hidden, hidden),
                'mlp.gate_proj.weight': (intermediate, hidden),
                'mlp.up_proj.weight': (intermediate, hidden),
                'mlp.down_proj.weight': (hidden, intermediate),
            }
            n, k = expected_shapes[name]
            if prefix + '.weight_packed' in layer:
                if tuple(layer[prefix + '.weight_shape'].tolist()) != (n, k):
                    raise ValueError(f'logical shape mismatch: layer {idx} {name}')
                if layer[prefix + '.weight_packed'].dtype != torch.int32 or tuple(layer[prefix + '.weight_packed'].shape) != (n, k//8):
                    raise ValueError(f'packed shape/dtype mismatch: layer {idx} {name}')
                if tuple(scale.shape) != (n, k//128):
                    raise ValueError(f'group scale mismatch: layer {idx} {name}')
                zero=layer.get(prefix+'.weight_zero_point')
                if zero is not None and (zero.dtype!=torch.int32 or tuple(zero.shape)!=(n//8,k//128)):
                    raise ValueError(f'packed zero-point mismatch: layer {idx} {name}')
            elif name not in layer or layer[name].dtype != torch.int8 or tuple(layer[name].shape) != (n,k):
                raise ValueError(f'INT8 shape/dtype mismatch: layer {idx} {name}')
            elif tuple(scale.shape)!=(n,1):
                raise ValueError(f'INT8 channel scale mismatch: layer {idx} {name}')
    manifest_path = path / 'experiment_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest['quantization_config'] = json.loads((path / 'config.json').read_text()).get('quantization_config', {})
    if 'lm_head.weight' not in globals_ and cfg.tie_word_embeddings:
        globals_['lm_head.weight'] = globals_['model.embed_tokens.weight']
    return CompressedStore(cfg, layers, globals_['model.embed_tokens.weight'], globals_['model.norm.weight'], globals_['lm_head.weight'], manifest)


class CompressedResidentRuntime:
    def __init__(self, store: CompressedStore, backend: 'CompressedLinearBackend', device='cuda:0'):
        self.store, self.backend = store, backend
        self.stats = RuntimeStats()
        self.stats.startup_h2d_bytes=sum(value.numel()*value.element_size() for layer in store.layers for value in layer.values() if isinstance(value,torch.Tensor))
        self.gpu_layer_budget = len(store.layers)
        self.resident = []
        for idx, source in enumerate(store.layers):
            layer = {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in source.items()}
            for name in LINEAR_WEIGHT_NAMES:
                backend.prepare(layer, name)
            self.resident.append(layer)
            print(f'prepared layer {idx+1}/{len(store.layers)} for {backend.name}', flush=True)

    @property
    def resident_storage_bytes(self):
        storages={}
        def visit(value):
            if isinstance(value,torch.Tensor) and value.device.type=='cuda':
                storage=value.untyped_storage()
                storages[storage.data_ptr()]=storage.nbytes()
            elif isinstance(value,dict):
                for child in value.values():visit(child)
            elif isinstance(value,(list,tuple)):
                for child in value:visit(child)
        visit(self.resident)
        return sum(storages.values())

    def begin_forward(self):
        pass

    def weights_for(self, layer_idx):
        return self.resident[layer_idx]

    def after_layer(self, layer_idx):
        pass

    def end_forward(self):
        pass


class CompressedLinearBackend:
    def __init__(self, name: str, group_size=128):
        if name not in ('marlin', 'int8_gemm', 'reference_int4', 'reference_w8a8'):
            raise ValueError(name)
        self.name, self.group_size = name, group_size
        self.calls = 0
        self.fallback_count = 0
        self.explicit_dequant_bytes = 0
        self.gemm_calls = 0
        self.activation_quantize_calls = 0
        # Row splitting remains an explicit diagnostic option, not a hidden
        # fallback or the claimed fix for the legacy kernel's race hazards.
        self.marlin_max_rows_per_call = int(os.environ.get('ONDEVICEINFER_MARLIN_MAX_ROWS','0'))
        self.marlin_use_fp32_reduce = os.environ.get('ONDEVICEINFER_MARLIN_FP32_REDUCE','0')=='1'
        if name in ('marlin', 'int8_gemm'):
            from vllm import _custom_ops as ops
            self.ops = ops
            if name=='marlin':
                self.marlin_version=importlib.metadata.version('vllm')
                if self.marlin_version=='0.11.2' and os.environ.get('ONDEVICEINFER_ALLOW_UNVALIDATED_MARLIN')!='1':
                    raise RuntimeError('vLLM 0.11.2 failed repeated SM86 real-weight stability gates; use the validated kernel environment. Set ONDEVICEINFER_ALLOW_UNVALIDATED_MARLIN=1 only for failure reproduction.')
                self.marlin_has_a_scales='a_scales' in inspect.signature(ops.gptq_marlin_gemm).parameters

    def _marlin_gemm(self, x, prepared, output=None, *, use_fp32_reduce=None):
        packed,scale,zero,perm,workspace,qtype,k,n=prepared
        self.gemm_calls += 1
        leading=(x,output,packed,None,scale)
        if getattr(self,'marlin_has_a_scales',False):leading+=(None,)
        trailing=(None,zero,perm,perm,workspace,qtype,x.shape[0],n,k)
        if use_fp32_reduce is None:use_fp32_reduce=getattr(self,'marlin_use_fp32_reduce',False)
        return self.ops.gptq_marlin_gemm(*leading,*trailing,use_atomic_add=False,use_fp32_reduce=use_fp32_reduce)

    def prepare(self, layer: dict, name: str) -> None:
        prefix = name.removesuffix('.weight')
        if self.name == 'marlin':
            from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales, marlin_zero_points, marlin_make_workspace_new
            from vllm.scalar_type import scalar_types
            n, k = (int(x) for x in layer[prefix + '.weight_shape'].tolist())
            packed = layer[prefix + '.weight_packed']
            perm = torch.empty(0, dtype=torch.int32, device=packed.device)
            if layer.get(prefix + '.weight_g_idx') is not None:
                raise ValueError('runtime G-index unsupported in this experiment; disable act-order')
            repacked = self.ops.gptq_marlin_repack(packed.t().contiguous(), perm, k, n, 4)
            scale = marlin_permute_scales(layer[prefix + '.weight_scale'].t().half().contiguous(), k, n, self.group_size)
            raw_zero = layer.get(prefix + '.weight_zero_point')
            if raw_zero is None:
                zero, qtype = perm, scalar_types.uint4b8
            else:
                unsigned_zero = unpack_unsigned(raw_zero, 4, 0, n).t().contiguous()
                zero = marlin_zero_points(unsigned_zero, k // self.group_size, n, 4)
                qtype = scalar_types.uint4
            layer[prefix + '._prepared'] = (repacked, scale, zero, perm, marlin_make_workspace_new(packed.device), qtype, k, n)
            # Fused path retains only the repacked compressed weights.
            for suffix in ('.weight_packed', '.weight_scale', '.weight_zero_point'):
                layer.pop(prefix + suffix, None)
        elif self.name == 'int8_gemm':
            q = layer[name]
            if q.dtype != torch.int8:
                raise TypeError(f'{name}: expected INT8, got {q.dtype}')
            layer[prefix + '._prepared'] = (q.t(), layer[prefix + '.weight_scale'].float().reshape(1, -1))

    def linear(self, inputs: torch.Tensor, weights: dict, name: str) -> torch.Tensor:
        self.calls += 1
        prefix = name.removesuffix('.weight')
        with nvtx_range(f'weight_quant::{self.name}::{name}'):
            if self.name == 'marlin':
                packed, scale, zero, perm, workspace, qtype, k, n = weights[prefix + '._prepared']
                x = inputs.reshape(-1, k).contiguous()
                prepared=weights[prefix+'._prepared']
                if self.marlin_max_rows_per_call<=0 or x.shape[0] <= self.marlin_max_rows_per_call:
                    out = self._marlin_gemm(x,prepared)
                else:
                    out=x.new_empty((x.shape[0],n))
                    for start in range(0,x.shape[0],self.marlin_max_rows_per_call):
                        end=min(start+self.marlin_max_rows_per_call,x.shape[0])
                        self._marlin_gemm(x[start:end],prepared,out[start:end])
                return out.reshape(*inputs.shape[:-1], n)
            if self.name == 'int8_gemm':
                qweight, scale = weights[prefix + '._prepared']
                x = inputs.reshape(-1, inputs.shape[-1]).contiguous()
                qx, xscale, _ = self.ops.scaled_int8_quant(x, symmetric=True)
                self.activation_quantize_calls += 1
                out = self.ops.cutlass_scaled_mm(qx, qweight, xscale, scale, torch.float16)
                self.gemm_calls += 1
                return out.reshape(*inputs.shape[:-1], qweight.shape[-1])
            weight = decode_weight(weights, name, group_size=self.group_size)
            self.explicit_dequant_bytes += weight.numel() * weight.element_size()
            if self.name == 'reference_w8a8':
                x = inputs.float()
                scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127
                inputs = (x.div(scale).round().clamp(-127, 127) * scale).half()
            return F.linear(inputs, weight)
