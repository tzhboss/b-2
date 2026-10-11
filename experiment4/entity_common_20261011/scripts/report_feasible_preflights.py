"""Qualify only short TRAIN controls and retain demonstrated solver failures."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
import yaml
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from common_training import sha
from feasible_train_optimization import TrainLossProblem
from shared_rank_cache_v2 import SharedRankCache
from shared_rank_solver_v2 import GuardSeparator

def load(name):
    path=root/'scripts'/name;spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
failed=load('control_feasible_train41_initial_failed.py')
try:failed.controls()
except AssertionError:first='EXPECTED_ORIGINAL_ACTIVE_CUT_CONTROL_FAILURE'
else:raise AssertionError('Original retained control no longer fails')
path=root/'src/feasible_train_optimization_v2_failed_start_repair.py'
spec=importlib.util.spec_from_file_location('failed_start_repair',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
failed.optimize_train_row=m.optimize_train_row
try:failed.controls()
except AssertionError:second='EXPECTED_FAILED_FEASIBLE_START_ATTEMPT'
else:raise AssertionError('Retained attempted repair no longer fails')
E=torch.tensor([[1.,0.],[.5,1.],[2.,0.],[1.,0.],[0.,0.]],dtype=torch.float64);R=torch.ones((1,2),dtype=torch.float64)
cache=SharedRankCache(E,R,[[0,0]],[[0,3]],k=2,score_batch=2)
problem=TrainLossProblem(E.numpy(),R.numpy(),4,[[1,0,4]],[[1,0,2]],np.zeros(2),3.,.0005,1e-8)
options=json.loads((root/'configs/feasible-train40.json').read_text())['solver'];opt=problem.solve(np.zeros(2),options);assert opt.success
separator=GuardSeparator(cache,[4],opt.x.reshape(1,-1),[]);patch=cache.propose([4],opt.x.reshape(1,-1));pairs=separator.separate(patch,64)
immutable=[pair for pair in pairs if 4 not in (pair[0],pair[2],pair[3])]
assert (0,0,3,0) in immutable
problem.add(immutable);assert problem.values(np.zeros(2))[0]<-options['primal_tolerance']
failure=dict(status='PASS_RETAINED_FAILURE_REPLAY_AND_IMMUTABLE_TIE_ROOT_CAUSE',original_failure=first,attempted_start_repair=second,
       immutable_pairs=immutable,constant_pair_residual=float(problem.values(np.zeros(2))[0]),
       source_sha256={str(p.relative_to(root)):sha(p) for p in [root/'scripts/control_feasible_train41_initial_failed.py',path]},scientific_utility_asserted=False)
target=root/'results/feasible-train-working-set-failure.json';assert not target.exists();target.write_text(json.dumps(failure,indent=2))
records=[]
for exp in ('KGE-ENTITY-COMMON-FEASIBLE-TRAIN-PREFLIGHT-040','KGE-ENTITY-COMMON-FEASIBLE-TRAIN-PREFLIGHT-040R2'):
    state=yaml.safe_load((root/'experiments'/exp/'experiment.yaml').read_text());assert state['status']=='completed_audited_train_preflight_only' and not state['active_processes']
    for name,digest in state['frozen_input_sha256'].items():assert sha(root/name)==digest,name
    summary=json.loads((root/state['result_directory']/'summary.json').read_text());assert summary['completed_audited_jobs']==6
    rows=[]
    for row in summary['rows']:
        r=row['receipt'];a=row['audit'];assert not a['failures'] and a['patches_replayed']==8
        rows.append(dict(id=row['job']['id'],accepted=r['accepted'],elapsed_s=r['elapsed_s'],independent_cpu_cells=a['score_cells_replayed']))
    records.append(dict(experiment=exp,rows=rows,total_patch_checks=48,independent_cpu_cells=sum(r['independent_cpu_cells'] for r in rows),utility_asserted=False))
result=dict(status='PASS_BOTH_SIX_JOB_PREFLIGHTS_WITH_RETAINED_CONTROL_FAILURES',rows=records,scientific_utility_asserted=False)
target=root/'results/feasible-train-preflight-disclosure.json';assert not target.exists();target.write_text(json.dumps(result,indent=2))
append='''

040 and040R2 both completed six TRAIN-only eight-coordinate preflights,
with full mining/scalar-loss/state checks and32 terminal full-pool CPU query
fixtures each (2,161,536 cells per experiment; same fixture cohort reused,
not doubled independent coverage). Original active-cut control and attempted
feasible-start repair failures are retained. Root cause is an impossible
positive-margin inequality between an immutable tied competitor and reference.
V2 omits immutable-only solver working rows; actual rank acceptance is unchanged.
Strong active-cut TRAIN descent and exhausted-budget unchanged-fallback controls
now pass. This is numerical qualification, not demonstrated useful retrieval.

041 is separately registered: six full3753-row direct TRAIN-loss sweeps,
original full/scoped guard identities, frozen old parameters, original bounds,
complete common TEST and unchanged directional/old-loss gates. Its current
execution and utility are determined only by remote041 canonical/processes
and final independent audits; registration is not a success result.
'''
for name in ('plan/review/outline-mechanism-status.md','plan/review/shared-graph-disposition.md'):
    p=root/name;text=p.read_text();assert '040 and040R2 both completed' not in text;p.write_text(text.rstrip()+append,encoding='utf8')
with (root/'.learnings/ERRORS.md').open('a',encoding='utf8') as f:
    f.write('\n\n## 2026-10-11 active rank-cut control\nOriginal direct-TRAIN active-cut control failed. A feasible-start attempt also failed. Root cause: immutable candidate tied to frozen reference was turned into impossible strict-margin inequality. Skip only fully immutable solver rows, retain actual complete rank acceptance; repaired active-cut/fallback controls and six040R2 graph preflights pass. Original controls/failures retained. No utility or novelty inference.\n')
print(json.dumps(result),flush=True)
