import copy
import importlib.util
import sys
from pathlib import Path

import pytest


BENCHMARKS = Path(__file__).resolve().parents[1] / 'benchmarks'
spec = importlib.util.spec_from_file_location('weight_quant_quality', BENCHMARKS / 'validate_weight_quant.py')
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(BENCHMARKS))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def protocol():
    manifest = dict(dataset_sha256='frozen', samples=2, generation=4,
                    ppl_window=1024, corpus_tokens=65536, compute_dtype='float16',
                    teacher_logit_positions='last16', kv='resident_fp16',
                    greedy_batching='equal-length', torch='fixed', cuda='12.8',
                    forward_code_sha256='unchanged')
    reference = dict(manifest=manifest | {'backend': 'fp16'},
                     corpora={'zh': {'ppl': 7.1, 'effective_tokens': 65472}},
                     prompts=[{'index': i, 'finite_logits': True} for i in range(2)])
    return manifest, reference


def test_partial_reference_only_allows_completed_teacher_stage():
    manifest, reference = protocol()
    module.validate_reference(reference, manifest, require_greedy=False, languages=['zh'])
    with pytest.raises(ValueError, match='greedy'):
        module.validate_reference(reference, manifest, require_greedy=True, languages=['zh'])
    for row in reference['prompts']:
        row['generated_ids'] = [1, 2, 3, 4]
    reference['complete'] = True
    module.validate_reference(reference, manifest, require_greedy=True, languages=['zh'])


@pytest.mark.parametrize('key', ['dataset_sha256', 'ppl_window', 'compute_dtype', 'forward_code_sha256'])
def test_partial_reference_does_not_relax_protocol(key):
    manifest, reference = protocol()
    reference = copy.deepcopy(reference)
    reference['manifest'][key] = 'changed'
    with pytest.raises(ValueError, match=key):
        module.validate_reference(reference, manifest, require_greedy=False, languages=['zh'])


def test_missing_or_nonfinite_reference_is_rejected():
    manifest, reference = protocol()
    with pytest.raises(ValueError, match='corpus'):
        module.validate_reference(reference, manifest, require_greedy=False, languages=['en'])
    reference['corpora']['zh']['ppl'] = float('nan')
    with pytest.raises(ValueError, match='corpus'):
        module.validate_reference(reference, manifest, require_greedy=False, languages=['zh'])


def test_incomplete_teacher_rows_rejected():
    manifest, reference = protocol()
    reference['prompts'].pop()
    with pytest.raises(ValueError, match='teacher'):
        module.validate_reference(reference, manifest, require_greedy=False, languages=['zh'])
