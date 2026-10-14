"""Complete saved 062 models after missing CuBLAS deterministic configuration."""
import json
import time
import torch
import experiment62 as run

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)
start = time.monotonic()
run.ex.progress('resume_saved_models_evaluation')
d = run.ex.base.load(run.ex.SRC / 'entity_diagnostic42_20261012/artifacts/data/train.npz')
A = run.ex.base.load(run.ex.SRC / 'replay_strength53_20261013/artifacts/evaluation/A.npz')
quality, evaluation = run.evaluate(d['old'], d['new'], A)
prep = json.loads((run.ROOT / 'preparation.json').read_text())
costs = {'preparation': prep, 'training': {name: json.loads((run.ROOT / 'models' / name / 'training.json').read_text()) for name in run.ARMS},
         'evaluation': evaluation, 'shared_prefix_cache': str(run.PRIOR / 'cache/prefix10.pt'),
         'evaluation_continuation_wall_seconds': time.monotonic() - start,
         'execution_note': 'Original run.log retained: all 8 models saved; first evaluation failed before any ranking due to missing CUBLAS_WORKSPACE_CONFIG. Only saved-model evaluation continued; no retraining.',
         'time_accounting': 'measured stage sum, excluding initial interpreter/import/failure overhead'}
costs['measured_stage_seconds'] = prep['seconds'] + sum(x['seconds'] for x in costs['training'].values()) + evaluation['seconds']
run.report_results(quality, costs)
run.ex.progress('complete', measured_stage_seconds=costs['measured_stage_seconds'])
