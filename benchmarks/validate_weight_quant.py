#!/usr/bin/env python3
"""Held-out source-text PPL, teacher logits and independent greedy generations."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark_weight_quant_models import load_model
from ondeviceinfer.generation import greedy_generate
from ondeviceinfer.offload_model import OffloadedQwen3ForCausalLM
from ondeviceinfer.runtime.offload import load_cpu_weight_store, PrefetchOffloadRuntime


def validate_reference(reference, manifest, *, require_greedy, languages):
    """A completed teacher baseline can be used before slow greedy evaluation.

    Never relax the numerical protocol: only the independent greedy portion
    may be incomplete, and only for experiments that explicitly skip it.
    """
    expected = reference.get('manifest', {})
    if expected.get('backend') != 'fp16':
        raise ValueError('Quality reference must use the original FP16 backend')
    keys = ('dataset_sha256', 'samples', 'generation', 'ppl_window',
            'corpus_tokens', 'compute_dtype', 'teacher_logit_positions', 'kv',
            'greedy_batching', 'torch', 'cuda', 'forward_code_sha256')
    for key in keys:
        if expected.get(key) != manifest.get(key):
            raise ValueError(f'reference protocol mismatch: {key}')
    for language in languages:
        corpus = reference.get('corpora', {}).get(language, {})
        if not corpus.get('effective_tokens', 0) or not math.isfinite(corpus.get('ppl', float('nan'))) or corpus['ppl'] <= 0:
            raise ValueError(f'Incomplete FP16 corpus reference: {language}')
    prompts = reference.get('prompts', [])
    if len(prompts) != manifest['samples']:
        raise ValueError('Incomplete FP16 teacher reference')
    for index, row in enumerate(prompts):
        if row.get('index') != index or not row.get('finite_logits'):
            raise ValueError(f'Invalid FP16 teacher reference row {index}')
        if require_greedy and len(row.get('generated_ids', [])) != manifest['generation']:
            raise ValueError(f'Incomplete FP16 greedy reference row {index}')
    if require_greedy and not reference.get('complete'):
        raise ValueError('Full quality evaluation requires a complete FP16 reference')


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--backend',required=True)
    parser.add_argument('--data',type=Path,default=Path('reports/raw/weight_quant_compare/dataset.json'))
    parser.add_argument('--reference-dir',type=Path,default=Path('/mnt/ondeviceinfer/datasets/weight_quant_compare/fp16_reference'))
    parser.add_argument('--reference-meta',type=Path,help='Explicit FP16 metadata; teacher-only evaluation may use its saved partial checkpoint')
    parser.add_argument('--samples',type=int,default=96)
    parser.add_argument('--generation',type=int,default=128)
    parser.add_argument('--corpus-tokens',type=int,default=65536)
    parser.add_argument('--ppl-window',type=int,default=1024)
    parser.add_argument('--fp16-budget',type=int,default=32)
    parser.add_argument('--skip-greedy',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(8);torch.manual_seed(42)
    data=json.loads(args.data.read_text())
    args.reference_dir.mkdir(parents=True,exist_ok=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'manifest':{'model':str(args.model.resolve()),'backend':args.backend,'torch':torch.__version__,'cuda':torch.version.cuda,'compute_dtype':'float16','fp16_weight_layer_budget':args.fp16_budget if args.backend=='fp16' else None,'dataset_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'samples':args.samples,'generation':args.generation,'ppl_window':args.ppl_window,'corpus_label':data['corpus_label'],'teacher_logit_positions':'last 16 real source-text positions per prompt','kv':'resident_fp16','greedy_batching':'equal prompt lengths only, up to 4, no padding, identical grouping for all algorithms'},'corpora':{},'prompts':[]}
    report['manifest'].update(corpus_tokens=args.corpus_tokens,skip_greedy=args.skip_greedy)
    report['manifest'].update(vllm=importlib.metadata.version('vllm'),
        marlin_use_fp32_reduce=os.environ.get('ONDEVICEINFER_MARLIN_FP32_REDUCE','0')=='1',
        marlin_max_rows_per_call=int(os.environ.get('ONDEVICEINFER_MARLIN_MAX_ROWS','0')))
    fingerprint=hashlib.sha256()
    project_root=Path(__file__).resolve().parents[1]
    for relative in ['ondeviceinfer/quantized_model.py','ondeviceinfer/offload_model.py','ondeviceinfer/weight_quantization/backend.py']:
        fingerprint.update((project_root/relative).read_bytes())
    report['manifest']['forward_code_sha256']=fingerprint.hexdigest()
    reference_meta=args.reference_meta or args.reference_dir/'quality_fp16.json'
    reference=None
    if args.backend!='fp16':
        reference=json.loads(reference_meta.read_text())
        validate_reference(reference,report['manifest'],require_greedy=not args.skip_greedy,languages=data['corpora'])
        for index in range(args.samples):
            for kind in ('logits','layers'):
                if not (args.reference_dir/f'{kind}_{index:03d}.pt').exists():
                    raise ValueError(f'Missing FP16 {kind} artifact for row {index}')
        report['manifest']['reference_meta']=str(reference_meta.resolve())
        report['reference_status_at_start']={'complete':bool(reference.get('complete')),
            'teacher_rows':len(reference['prompts']),
            'greedy_rows':sum('generated_ids' in row for row in reference['prompts'])}
    if args.output.exists():
        previous=json.loads(args.output.read_text())
        if previous.get('resource_contaminated'):
            raise RuntimeError('Cannot resume GPU-contaminated quality data; archive it and start a fresh output.')
        if previous.get('manifest')!=report['manifest']:
            raise RuntimeError('Existing quality output has a different protocol/code fingerprint; use a new output path.')
        if previous.get('complete'):
            print('SKIP complete quality',args.output,flush=True)
            return
        report=previous
        print('RESUME quality',len(report.get('prompts',[])),'teacher rows',flush=True)
    def save_report():
        temporary=args.output.with_suffix(args.output.suffix+'.tmp')
        temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(args.output)
    if args.backend=='fp16':
        store=load_cpu_weight_store(args.model,dtype=torch.float16,pin_memory=True)
        runtime=PrefetchOffloadRuntime(store,device='cuda:0',gpu_layer_budget=args.fp16_budget)
        model=OffloadedQwen3ForCausalLM(store,runtime,device='cuda:0')
    else:
        model,_=load_model(args.model,args.backend)
    for language,stream in data['corpora'].items():
        if language in report['corpora']:
            print('SKIP complete corpus',language,flush=True)
            continue
        total_nll=0.;tokens=0;start=time.perf_counter()
        stream=stream[:args.corpus_tokens]
        for offset in range(0,len(stream)-1,args.ppl_window):
            ids=torch.tensor([stream[offset:offset+args.ppl_window]],device='cuda')
            if ids.shape[1]<2:continue
            cache=model.make_resident_cache(batch_size=1,max_seq_len=ids.shape[1])
            out=model(ids,cache=cache).logits
            for position in range(0,ids.shape[1]-1,128):
                end=min(position+128,ids.shape[1]-1)
                nll=F.cross_entropy(out[:,position:end].float().reshape(-1,out.shape[-1]),ids[:,position+1:end+1].reshape(-1),reduction='sum')
                total_nll+=float(nll);tokens+=end-position
            del out,cache
        report['corpora'][language]={'total_nll':total_nll,'effective_tokens':tokens,'ppl':math.exp(total_nll/tokens),'seconds':time.perf_counter()-start}
        save_report();print('PPL',language,report['corpora'][language],flush=True)
    for index,row in enumerate(data['splits']['test'][:args.samples]):
        if index<len(report['prompts']):
            artifacts_present=args.backend!='fp16' or all((args.reference_dir/f'{kind}_{index:03d}.pt').exists() for kind in ('logits','layers'))
            if not artifacts_present:raise RuntimeError(f'Missing FP16 teacher artifacts for completed row {index}; do not mix reference states.')
            continue
        layer_samples = {}
        layer_metrics = []
        if args.backend == 'fp16':
            def capture_layer(layer_index, hidden):
                layer_samples[layer_index] = hidden[:, -16:].detach().half().cpu()
        else:
            reference_layers = torch.load(args.reference_dir/f'layers_{index:03d}.pt',weights_only=True)
            def capture_layer(layer_index, hidden):
                actual = hidden[:, -16:].detach().float().cpu()
                expected = reference_layers[layer_index].float()
                difference = actual-expected
                layer_metrics.append({'layer':layer_index,'cosine':float(F.cosine_similarity(actual.flatten(),expected.flatten(),dim=0)),'rmse':float(difference.square().mean().sqrt()),'relative_l2':float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(expected).clamp_min(1e-20)),'finite':bool(torch.isfinite(actual).all())})
        model.layer_probe_callback = capture_layer
        ids=torch.tensor([row['input_ids']],device='cuda')
        keep=min(16,ids.shape[1])
        cache=model.make_resident_cache(batch_size=1,max_seq_len=ids.shape[1])
        logits=model(ids,cache=cache,logits_to_keep=keep).logits[0].float()
        model.layer_probe_callback = None
        metrics={'index':index,'language':row['language'],'source':row['source'],'source_row':row['source_row'],'teacher_positions':keep,'finite_logits':bool(torch.isfinite(logits).all())}
        if args.backend=='fp16':
            torch.save(logits.half().cpu(),args.reference_dir/f'logits_{index:03d}.pt')
            torch.save(layer_samples,args.reference_dir/f'layers_{index:03d}.pt')
        else:
            metrics['layer_output_errors'] = layer_metrics
            expected=torch.load(args.reference_dir/f'logits_{index:03d}.pt',weights_only=True).to('cuda').float()
            delta=logits-expected
            metrics.update(logits_cosine=float(F.cosine_similarity(expected.flatten(),logits.flatten(),dim=0)),logits_rmse=float(delta.square().mean().sqrt()),logits_kl=float(F.kl_div(logits.log_softmax(-1),expected.softmax(-1),reduction='batchmean')),teacher_top1_agreement=float((expected.argmax(-1)==logits.argmax(-1)).float().mean()))
            del expected,delta
        del cache,logits
        report['prompts'].append(metrics)
        save_report()
        print('teacher quality',index+1,args.samples,metrics.get('teacher_top1_agreement'),flush=True)
    if not args.skip_greedy:
        by_length=defaultdict(list)
        for index,row in enumerate(data['splits']['test'][:args.samples]):by_length[len(row['prompt_ids'])].append(index)
        completed=0
        for length,indices in sorted(by_length.items()):
            for offset in range(0,len(indices),4):
                selected_indices=indices[offset:offset+4]
                if all('generated_ids' in report['prompts'][i] for i in selected_indices):
                    completed+=len(selected_indices)
                    continue
                prompt=torch.tensor([data['splits']['test'][i]['prompt_ids'] for i in selected_indices],device='cuda')
                cache=model.make_resident_cache(batch_size=len(selected_indices),max_seq_len=length+args.generation)
                result=greedy_generate(model,prompt,max_new_tokens=args.generation,cache=cache,eos_token_id=None)
                generated_batch=result.generated_ids.tolist()
                for index,generated in zip(selected_indices,generated_batch):
                    metrics=report['prompts'][index]
                    metrics.update(generated_ids=generated,greedy_batch_size=len(selected_indices))
                    if args.backend!='fp16':
                        target=reference['prompts'][index]['generated_ids']
                        lcp=next((i for i,(a,b) in enumerate(zip(generated,target)) if a!=b),len(generated))
                        metrics.update(greedy_token_agreement=sum(a==b for a,b in zip(generated,target))/len(target),lcp=lcp,first_divergence=None if lcp==len(target) else lcp)
                completed+=len(selected_indices)
                save_report()
                print('greedy quality',completed,args.samples,'batch',len(selected_indices),flush=True)
                del cache
    if args.backend!='fp16':
        for language in report['corpora']:
            report['corpora'][language]['relative_ppl_change']=report['corpora'][language]['ppl']/reference['corpora'][language]['ppl']-1
    report['complete']=True
    save_report()
    if args.backend=='fp16':reference_meta.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
