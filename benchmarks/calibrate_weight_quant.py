#!/usr/bin/env python3
"""Sequential library calibration of the project's five comparison formats."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import resource
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from compressed_tensors.offload import align_modules


def norm_transform_samples(mappings, active_names, exclude_up=False):
    snapshots = []
    for mapping in mappings or []:
        name = getattr(mapping, 'smooth_name', '')
        norm = mapping.smooth_layer
        if name not in active_names or norm.weight.ndim != 1:
            continue
        # AWQ also scales up_proj outputs later; gate and Q/K/V have no such
        # subsequent transform, so their effective Norm->Linear weights can
        # be compared before/after the actual library transformation.
        for index, layer in enumerate(mapping.balance_layers):
            if exclude_up and name.endswith('post_attention_layernorm') and index == 1:
                continue
            with align_modules([norm, layer]):
                indices = [0, layer.weight.shape[0]//2, layer.weight.shape[0]-1]
                effective = layer.weight[indices].float() * norm.weight.float()
                snapshots.append((name, norm, layer, indices, effective.cpu()))
    return snapshots


def verify_norm_transform(snapshots):
    result=[]
    for name,norm,layer,indices,expected in snapshots:
        with align_modules([norm,layer]):
            actual=(layer.weight[indices].float()*norm.weight.float()).cpu()
        relative=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-20))
        result.append({'mapping':name,'shape':list(layer.weight.shape),'relative_l2':relative,'finite':bool(torch.isfinite(actual).all())})
    return result


class FP32LossAWQModifier(AWQModifier):
    """Keep the library search, but prevent FP16 sum-reduction overflow."""
    transform_checks: list = []

    def _apply_smoothing(self, model):
        snapshots=norm_transform_samples(self._resolved_mappings,set(self._smooth_activation_means),exclude_up=True)
        super()._apply_smoothing(model)
        self.transform_checks.extend(verify_norm_transform(snapshots))
    def _compute_loss(self, fp16_outputs, int_w_outputs):
        return super()._compute_loss(
            [value.float() for value in fp16_outputs],
            [value.float() for value in int_w_outputs],
        )

    @torch.no_grad()
    def _run_samples(self, module):
        # AWQ replays bound keyword arguments, while compressed-tensors 0.14
        # expects Linear's input as args[0].  Normalize only this call boundary.
        outputs = []
        for batch in self._parent_args_cache[module]:
            kwargs = dict(batch)
            if isinstance(module, torch.nn.Linear) and 'input' in kwargs:
                output = module(kwargs.pop('input'), **kwargs)
            else:
                output = module(**kwargs)
            if isinstance(output, tuple):
                output = output[0]
            outputs.append(output)
        return outputs


class CheckedSmoothQuantModifier(SmoothQuantModifier):
    """Record sampled effective weights around the real smoothing operation."""
    transform_checks: list = []

    def _apply_smoothing(self, model):
        snapshots=norm_transform_samples(self.resolved_mappings_,set(self.scales_ or {}))
        super()._apply_smoothing(model)
        self.transform_checks.extend(verify_norm_transform(snapshots))


def make_recipe(algorithm, alpha, dampening):
    targets=['re:.*\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$']
    if algorithm in ('awq','gptq','rtn'):
        scheme=QuantizationScheme(targets=targets,weights=QuantizationArgs(num_bits=4,type='int',symmetric=False,strategy='group',group_size=128,observer='minmax'))
    else:
        scheme=QuantizationScheme(targets=targets,weights=QuantizationArgs(num_bits=8,type='int',symmetric=True,strategy='channel',observer='minmax'),input_activations=QuantizationArgs(num_bits=8,type='int',symmetric=True,strategy='token',dynamic=True))
    if algorithm=='gptq':
        return [GPTQModifier(config_groups={'group_0':scheme},ignore=['lm_head'],block_size=128,dampening_frac=dampening,actorder=None)]
    quantizer=QuantizationModifier(config_groups={'group_0':scheme},ignore=['lm_head'])
    if algorithm=='awq':return [FP32LossAWQModifier(),quantizer]
    if algorithm=='smoothquant':return [CheckedSmoothQuantModifier(smoothing_strength=alpha),quantizer]
    return [quantizer]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--algorithm',choices=['awq','gptq','smoothquant','rtn','unsmoothed'],required=True)
    parser.add_argument('--data',type=Path,default=Path('reports/raw/weight_quant_compare/dataset.json'))
    parser.add_argument('--samples',type=int,default=128)
    parser.add_argument('--length',type=int,default=512)
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument('--dampening',type=float,default=0.01)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if (args.output/'experiment_manifest.json').exists():
        raise RuntimeError('completed checkpoint already exists; choose a new output')
    torch.set_num_threads(8);torch.manual_seed(42)
    data=json.loads(args.data.read_text())
    # Stratified selection for smoke, full shared calibration for formal runs.
    source=data['splits']['calibration']
    # Simpler interleaving avoids assigning all smoke samples to Chinese.
    by_language={lang:[r for r in source if r['language']==lang] for lang in ('zh','en','code')}
    selected=[by_language[lang][i] for i in range(48) for lang in by_language if i<len(by_language[lang])][:args.samples]
    dataset=Dataset.from_list([{'input_ids':row['input_ids'][:args.length],'attention_mask':[1]*len(row['input_ids'][:args.length])} for row in selected])
    recipe=make_recipe(args.algorithm,args.alpha,args.dampening)
    report={'schema_version':1,'algorithm':args.algorithm,'source':str(args.model.resolve()),'dataset_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'seed':42,'samples':len(selected),'max_seq_length':args.length,'effective_tokens':sum(len(row['input_ids']) for row in dataset),'recipe':[r.model_dump(mode='json') for r in recipe],'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','llmcompressor','compressed-tensors')},'status':'running','alpha':args.alpha,'dampening':args.dampening}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,default=str)+'\n')
    start=time.perf_counter()
    try:
        model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,device_map=None,local_files_only=True,attn_implementation='sdpa')
        original_tied_embeddings = json.loads((args.model/'config.json').read_text()).get('tie_word_embeddings', False)
        model.eval();model.config.use_cache=False
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        loaded=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        oneshot(model=model,tokenizer=tokenizer,dataset=dataset,recipe=recipe,max_seq_length=args.length,num_calibration_samples=len(selected),pipeline='sequential',sequential_targets=['Qwen3DecoderLayer'],batch_size=1)
        torch.cuda.synchronize()
        calibrated=time.perf_counter()
        args.output.mkdir(parents=True,exist_ok=True)
        model.config.use_cache=True
        model.config.tie_word_embeddings = original_tied_embeddings
        if original_tied_embeddings:
            model.tie_weights()
        model.save_pretrained(args.output,save_compressed=True,max_shard_size='2GB')
        tokenizer.save_pretrained(args.output)
        files=[{'name':p.name,'bytes':p.stat().st_size} for p in sorted(args.output.glob('*.safetensors'))]
        report.update(status='complete',load_seconds=loaded-start,calibration_seconds=calibrated-loaded,save_seconds=time.perf_counter()-calibrated,total_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,tensor_files=files,checkpoint_file_bytes=sum(f['bytes'] for f in files))
        report['transform_checks']=[check for modifier in recipe for check in getattr(modifier,'transform_checks',[])]
        (args.output/'experiment_manifest.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    except Exception as error:
        report.update(status='failed',error=repr(error),total_seconds=time.perf_counter()-start)
        raise
    finally:
        args.report.write_text(json.dumps(report,indent=2,default=str)+'\n')


if __name__=='__main__':main()
