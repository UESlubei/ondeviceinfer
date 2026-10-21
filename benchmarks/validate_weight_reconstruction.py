#!/usr/bin/env python3
"""Sampled effective-weight errors, respecting exported Norm transforms.

AWQ Up/Down carry an additional equivalent row/column transform whose search
scales were not retained. They are explicitly excluded, not compared against
an incompatible original matrix. Full-model layer errors cover these paths.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ondeviceinfer.weights import SafetensorIndex
from ondeviceinfer.runtime.quantized import LINEAR_WEIGHT_NAMES
from ondeviceinfer.weight_quantization.backend import load_compressed_store, unpack_unsigned


def decode_rows(weights, name, rows):
    prefix=name.removesuffix('.weight')
    if prefix+'.weight_packed' not in weights:
        return (weights[name][rows].float()*weights[prefix+'.weight_scale'][rows].float().reshape(-1,1)).half()
    n,k=map(int,weights[prefix+'.weight_shape'].tolist())
    q=unpack_unsigned(weights[prefix+'.weight_packed'][rows],4,1,k).float()
    zero=weights.get(prefix+'.weight_zero_point')
    z=unpack_unsigned(zero,4,0,n)[rows].float() if zero is not None else 8
    scale=weights[prefix+'.weight_scale'][rows].float()
    groups=torch.arange(k)//128
    if isinstance(z,torch.Tensor):z=z[:,groups]
    return ((q-z)*scale[:,groups]).half()


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('/mnt/ondeviceinfer/models/Qwen3-14B'))
    parser.add_argument('--algorithm',required=True)
    parser.add_argument('--rows',type=int,default=128)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(8)
    store=load_compressed_store(args.model);source=SafetensorIndex(args.source)
    report={'algorithm':args.algorithm,'checkpoint':str(args.model.resolve()),'source':str(args.source.resolve()),
            'sampling':'128 evenly spaced output rows; all input columns; every layer',
            'metric':'effective weight with associated input RMSNorm gamma folded; FP16 checkpoint reconstruction then FP32 comparison',
            'excluded':[], 'cases':[]}
    for layer_index,weights in enumerate(store.layers):
        prefix=f'model.layers.{layer_index}.'
        for name in LINEAR_WEIGHT_NAMES:
            if args.algorithm=='awq' and name in ('mlp.up_proj.weight','mlp.down_proj.weight'):
                report['excluded'].append({'layer':layer_index,'name':name,'reason':'unretained equivalent Up/Down search scales; covered by full-model layer-output quality'})
                continue
            tensor=source.get_tensor(prefix+name)
            rows=torch.linspace(0,tensor.shape[0]-1,min(args.rows,tensor.shape[0])).round().long().unique()
            expected=tensor[rows].half().float()
            actual=decode_rows(weights,name,rows).float()
            norm=None
            if name in ('self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight'):
                norm='input_layernorm.weight'
            elif name in ('mlp.gate_proj.weight','mlp.up_proj.weight'):
                norm='post_attention_layernorm.weight'
            if norm:
                expected*=source.get_tensor(prefix+norm).half().float()
                actual*=weights[norm].float()
            delta=actual-expected
            report['cases'].append({'layer':layer_index,'name':name,'sample_rows':len(rows),
                'elements':actual.numel(),'mse':float(delta.square().mean()),
                'cosine':float(F.cosine_similarity(actual.flatten(),expected.flatten(),dim=0)),
                'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(expected).clamp_min(1e-20)),
                'max_abs':float(delta.abs().max()),'finite':bool(torch.isfinite(actual).all())})
        print('weight reconstruction',layer_index+1,len(store.layers),flush=True)
    report['complete']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
