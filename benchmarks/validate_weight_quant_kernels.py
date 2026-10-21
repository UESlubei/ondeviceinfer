#!/usr/bin/env python3
"""Golden tests on actual calibrated matrices, not synthetic scales alone."""
import argparse
import json
import importlib.metadata
import statistics
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from ondeviceinfer.weight_quantization.backend import load_compressed_store, CompressedLinearBackend, decode_weight
from ondeviceinfer.runtime.quantized import LINEAR_WEIGHT_NAMES


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--backend',choices=['marlin','int8_gemm'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmups',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=100)
    parser.add_argument('--correctness-repeats',type=int,default=20)
    args=parser.parse_args();torch.set_num_threads(8);torch.manual_seed(42)
    store=load_compressed_store(args.model)
    report={'checkpoint':str(args.model),'backend':args.backend,'warmups':args.warmups,'repeats':args.repeats,'timing':'CUDA events; fused includes dynamic activation quantization; FP16 Linear uses same QDQ weight and input; reference includes explicit QDQ','cases':[]}
    report.update(vllm=importlib.metadata.version('vllm'),torch=torch.__version__,cuda=torch.version.cuda)
    def measure(operation):
        for _ in range(args.warmups):operation()
        torch.cuda.synchronize()
        events=[]
        for _ in range(args.repeats):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();operation();end.record();events.append((start,end))
        torch.cuda.synchronize()
        values=sorted(start.elapsed_time(end) for start,end in events)
        return {'median_ms':statistics.median(values),'p95_ms':values[min(len(values)-1,int(0.95*len(values)))], 'samples_ms':values}
    for layer_idx in sorted({0,len(store.layers)//2,len(store.layers)-1}):
        for name in LINEAR_WEIGHT_NAMES:
            prefix=name.removesuffix('.weight')
            weights={key:value.cuda() for key,value in store.layers[layer_idx].items() if key==name or key.startswith(prefix+'.')}
            reference_weight=decode_weight(weights,name)
            reference_weights=dict(weights)
            reference_backend=CompressedLinearBackend('reference_int4' if args.backend=='marlin' else 'reference_w8a8')
            backend=CompressedLinearBackend(args.backend)
            torch.cuda.synchronize();start=time.perf_counter()
            backend.prepare(weights,name)
            torch.cuda.synchronize();repack_seconds=time.perf_counter()-start
            for m in (1,4,32,256):
                x=torch.randn((m,reference_weight.shape[1]),device='cuda',dtype=torch.float16)*0.1
                if args.backend=='int8_gemm':
                    qx,sx,_=backend.ops.scaled_int8_quant(x)
                    xr=(qx.float()*sx).half()
                else:xr=x
                expected=F.linear(xr,reference_weight)
                actual=backend.linear(x,weights,name)
                delta=actual.float()-expected.float()
                cosine=float(F.cosine_similarity(actual.float().flatten(),expected.float().flatten(),dim=0))
                relative=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
                worst_cosine=cosine;worst_relative=relative;changes=0
                finite=bool(torch.isfinite(actual).all());worst_max_abs=float(delta.abs().max());variation_abs=0.0
                for _ in range(args.correctness_repeats-1):
                    repeated=backend.linear(x,weights,name)
                    error=repeated.float()-expected.float()
                    finite=finite and bool(torch.isfinite(repeated).all())
                    worst_max_abs=max(worst_max_abs,float(error.abs().max()))
                    variation_abs=max(variation_abs,float((repeated.float()-actual.float()).abs().max()))
                    worst_cosine=min(worst_cosine,float(F.cosine_similarity(repeated.float().flatten(),expected.float().flatten(),dim=0)))
                    worst_relative=max(worst_relative,float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)))
                    changes+=int(not torch.equal(repeated,actual))
                timing={
                    'fused':measure(lambda:backend.linear(x,weights,name)),
                    'fp16_linear':measure(lambda:F.linear(xr,reference_weight)),
                    'explicit_qdq_reference':measure(lambda:reference_backend.linear(x,reference_weights,name)),
                }
                report['cases'].append({'layer':layer_idx,'name':name,'m':m,'cosine':worst_cosine,'relative_l2':worst_relative,'max_abs':worst_max_abs,'finite':finite,'passed':worst_cosine>=0.9999 and worst_relative<=0.02 and finite,'bitwise_changes':changes,'repeat_variation_max_abs':variation_abs,'correctness_repeats':args.correctness_repeats,'marlin_max_rows_per_call':backend.marlin_max_rows_per_call if args.backend=='marlin' else None,'marlin_use_fp32_reduce':backend.marlin_use_fp32_reduce if args.backend=='marlin' else None,'fallback_count':backend.fallback_count,'repack_seconds':repack_seconds,'timing':timing})
            del weights,reference_weights,reference_weight,backend,reference_backend
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    report['all_passed']=all(row['passed'] and row['finite'] for row in report['cases'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('all_passed',report['all_passed'],flush=True)
    if not report['all_passed']:raise RuntimeError('calibrated kernel golden failed')


if __name__=='__main__':main()
