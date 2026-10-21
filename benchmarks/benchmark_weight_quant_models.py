#!/usr/bin/env python3
"""End-to-end comparisons in the custom Qwen3 forward and cache runtimes."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from benchmark_kv_offload import generate_profiled, exact_length_batch_ids, DEFAULT_TEXT, tensor_ids_sha256
from ondeviceinfer.quantized_model import QuantizedQwen3ForCausalLM
from ondeviceinfer.runtime.quantized import load_quantized_cpu_weight_store, QuantizedResidentRuntime
from ondeviceinfer.kv_offload import OffloadedKVCache
from ondeviceinfer.weight_quantization.backend import load_compressed_store, CompressedResidentRuntime, CompressedLinearBackend


def gpu_snapshot():
    fields='temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current,memory.used'
    result=subprocess.run(['nvidia-smi',f'--query-gpu={fields}','--format=csv,noheader,nounits'],capture_output=True,text=True)
    return {'fields':fields.split(','),'values':result.stdout.strip().split(', '),'captured_unix':time.time(),'error':result.stderr.strip() if result.returncode else None}


def load_model(path: Path, backend_name: str):
    if backend_name=='current_int8':
        store=load_quantized_cpu_weight_store(path,pin_memory=True,progress=True)
        runtime=QuantizedResidentRuntime(store,device='cuda:0')
        return QuantizedQwen3ForCausalLM(store,runtime,device='cuda:0'),None
    store=load_compressed_store(path)
    backend=CompressedLinearBackend(backend_name)
    runtime=CompressedResidentRuntime(store,backend)
    return QuantizedQwen3ForCausalLM(store,runtime,device='cuda:0',linear_backend=backend),backend


def make_cache(model,batch,capacity,mode):
    if mode=='resident':return model.make_resident_cache(batch_size=batch,max_seq_len=capacity)
    return OffloadedKVCache(num_layers=model.config.num_hidden_layers,batch_size=batch,num_key_value_heads=model.config.num_key_value_heads,max_seq_len=capacity,head_dim=model.config.head_dim,device='cuda:0',dtype=torch.float16,mode='prefetch',pin_memory=True,storage_format='int8_hqq',kv_group_size=64)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--backend',choices=['marlin','int8_gemm','reference_int4','reference_w8a8','current_int8'],required=True)
    parser.add_argument('--contexts',default='512,4096')
    parser.add_argument('--batches',default='1,2')
    parser.add_argument('--generation',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--kv-mode',choices=['resident','int8_prefetch'],default='resident')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage-path',type=Path)
    args=parser.parse_args()
    def stage(value):
        if args.stage_path:
            args.stage_path.parent.mkdir(parents=True,exist_ok=True)
            args.stage_path.write_text(json.dumps({'stage':value})+'\n')
    stage('load')
    torch.set_num_threads(8);torch.manual_seed(42)
    start=time.perf_counter();model,backend=load_model(args.model,args.backend);torch.cuda.synchronize()
    report={'manifest':{'model':str(args.model.resolve()),'backend':args.backend,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'load_and_repack_seconds':time.perf_counter()-start,'kv_mode':args.kv_mode,'warmup':'one full identical-length run per configuration','cpu_embedding':True,'threads':8},'measurements':[]}
    report['manifest'].update(startup_decoder_h2d_bytes=model.runtime.stats.startup_h2d_bytes,
        persistent_dequant_scratch_bytes=model.dequant_scratch.numel()*model.dequant_scratch.element_size(),
        gpu_base_allocated_bytes=torch.cuda.memory_allocated(),
        resident_decoder_storage_bytes=getattr(model.runtime,'resident_storage_bytes',None))
    report['manifest'].update(marlin_max_rows_per_call=backend.marlin_max_rows_per_call if backend and backend.name=='marlin' else None)
    report['manifest'].update(marlin_use_fp32_reduce=backend.marlin_use_fp32_reduce if backend and backend.name=='marlin' else None)
    report['manifest'].update(vllm=importlib.metadata.version('vllm'),compressed_tensors=importlib.metadata.version('compressed-tensors'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tokenizer_source=model.store.manifest.get('source',model.store.manifest.get('source_model',str(args.model)))
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_source,local_files_only=True)
    for context in map(int,args.contexts.split(',')):
        for batch in map(int,args.batches.split(',')):
            ids_cpu=exact_length_batch_ids(tokenizer,seed_text=DEFAULT_TEXT,length=context,batch_size=batch,batch_mode='identical')
            ids=ids_cpu.cuda()
            stage('warmup_cache_allocation')
            cache=make_cache(model,batch,context+args.generation,args.kv_mode)
            generate_profiled(model,ids,max_new_tokens=args.generation,window_tokens=16,cache=cache,stage_callback=lambda value:stage('warmup_'+value))
            del cache;torch.cuda.synchronize();torch.cuda.empty_cache()
            for repeat in range(args.repeats):
                stage('formal_cache_allocation')
                cache=make_cache(model,batch,context+args.generation,args.kv_mode)
                torch.cuda.reset_peak_memory_stats()
                before_calls=backend.calls if backend else 0
                before_gemms=backend.gemm_calls if backend else 0
                before_activation_quant=backend.activation_quantize_calls if backend else 0
                before_current_calls=model.compute_stats.linear_calls
                before_dequant=backend.explicit_dequant_bytes if backend else model.compute_stats.dequantized_weight_bytes
                gpu_before=gpu_snapshot()
                row=generate_profiled(model,ids,max_new_tokens=args.generation,window_tokens=16,cache=cache,stage_callback=lambda value:stage('formal_'+value))
                row.update(gpu_before=gpu_before,gpu_after=gpu_snapshot())
                row.update(backend_gemm_calls=backend.gemm_calls-before_gemms if backend else None,
                    activation_quantize_calls=backend.activation_quantize_calls-before_activation_quant if backend else 0)
                row.update(context=context,batch_size=batch,repeat=repeat,input_ids_sha256=tensor_ids_sha256(ids_cpu),backend=args.backend,backend_fallback_count=backend.fallback_count if backend else 0,linear_calls=backend.calls-before_calls if backend else model.compute_stats.linear_calls-before_current_calls,explicit_dequant_bytes=(backend.explicit_dequant_bytes if backend else model.compute_stats.dequantized_weight_bytes)-before_dequant,weight_h2d_bytes=model.runtime.stats.h2d_bytes if hasattr(model.runtime.stats,'h2d_bytes') else 0)
                report['measurements'].append(row)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({k:row[k] for k in ['context','batch_size','repeat','decode_tokens_per_s','itl_ms_p95','itl_ms_p99','peak_allocated_bytes']}),flush=True)
                del cache;torch.cuda.synchronize();torch.cuda.empty_cache()
    report['complete']=True
    stage('complete')
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
