import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/wn_smallprotect127_20261014')
import experiment127 as worker
base,core,cx,source=worker.base,worker.core,worker.cx,worker.source
ROOT=Path(__file__).resolve().parent
COMMON=['absolute','answer_anchor']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def train_missing():
    tick=time.monotonic();src=Path('/data/lc/untzh/wn_transfer112_20261014');old=Path('/data/lc/untzh/init_onset115_20261014');out=ROOT/'WN_DistMult-low';out.mkdir(exist_ok=True)
    base.N=core.N=24567;base.TOTAL=core.TOTAL=28660
    initial=torch.load(old/'cache/low-initial.pt',map_location='cpu',weights_only=True);prefix=torch.load(old/'cache/low-prefix10.pt',map_location='cpu',weights_only=True)
    ordinary=torch.load(old/'models/lowinit_none/epoch-20.pt',map_location='cpu',weights_only=True);z=base.load(src/'cache/train.npz');truths=core.truthmap(z['old'],z['new']);tape=base.load(src/'cache/learning.npz')
    c=base.load(src/'cache/cohort.npz');data={k:torch.as_tensor(c[k],device='cuda') for k in ['q','candidates','legal']};data['selections']=c['selections']
    space=torch.load(src/'cache/query-space.pt',map_location='cpu',weights_only=True)
    for g in space['groups']:g['indices']=g['indices'].cuda();g['basis']=g['basis'].cuda()
    space['rows']=space['rows'].cuda();space['vectors']=space['vectors'].cuda()
    replay={k:torch.as_tensor(v,device='cuda') for k,v in base.load(src/'cache/full-replay.npz').items() if k!='positions'}
    E,R,opt=source.model(initial,prefix);norms=[]
    for k in range(8):
        gl=torch.autograd.grad(base.learning(E,R,z['new'],tape,110+k),E)[0];gp=torch.autograd.grad(cx.protection(E,data,data['selections'][k],False),E)[0]
        norms.append([float(gl[24567:].double().norm()),float(gp[24567:].double().norm())])
    a=np.array(norms);coefficient=.1*np.sqrt(np.mean(a[:,0]**2))/np.sqrt(np.mean(a[:,1]**2));training={}
    for arm in COMMON+['full_replay_joint']:
        status('train_missing',arm=arm);training[arm]=worker.train('DistMult',arm,initial,prefix,ordinary,z['new'],tape,data,space,replay,coefficient,out)
    costs=dict(preparation_and_training_seconds=time.monotonic()-tick,training=training);base.write(out/'costs.json',costs)
    # All three genuinely new models committed before current VALID read.
    queries=base.load(src/'evaluation/queries.npz');cache={s:base.load(src/'cache'/f'fast-{s}.npz') for s in ['old_train','old_valid']};evaluation=out/'evaluation';evaluation.mkdir(exist_ok=True);metrics={};ec={}
    for arm in COMMON+['full_replay_joint']:
        status('evaluate_missing',arm=arm);m=torch.load(out/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda();states={};cost={}
        for split in ['old_train','old_valid']:states[split],cost[split]=source.fast_ranks(E,R,queries[split],cache[split],truths)
        st=time.monotonic();states['new_valid']=base.ranks(E,R,queries['new_valid'],truths);cost['new_valid']=dict(seconds=time.monotonic()-st);np.savez(evaluation/f'{arm}.npz',**states)
        metric={s:base.metrics(states[s],q) for s,q in queries.items()}
        for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],cache[split]['teacher'])
        metrics[arm]=metric;ec[arm]=cost
    costs['evaluation']=ec;base.write(out/'summary.json',dict(metrics=metrics,costs=costs));return metrics,costs

def protected_path(parent,mode,arm):
    if isinstance(parent,int):
        realarm='anchor_null' if arm=='answer_anchor' else arm
        if mode=='low':return Path(f'/data/lc/untzh/lowinit_confirm116_20261014/seed{parent}/evaluation/{realarm}.npz')
        return Path(f'/data/lc/untzh/smallinit_protect126_20261014/seed{parent}/{"random_half" if mode=="half" else "zero"}/evaluation/{realarm}.npz')
    name='ComplEx' if parent=='WN_ComplEx' else 'DistMult'
    if mode=='low':
        return Path(f'/data/lc/untzh/complex_transfer118_20261014/evaluation/{arm}.npz') if name=='ComplEx' else ROOT/'WN_DistMult-low/evaluation'/f'{arm}.npz'
    return Path(f'/data/lc/untzh/wn_smallprotect127_20261014/{name}/{"random_half" if mode=="half" else "zero"}/evaluation/{arm}.npz')

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    base.ROOT=core.ROOT=ROOT;tick=time.monotonic();status('prepare')
    decisions=json.loads(Path('/data/lc/untzh/admission_action128_20261014/decisions.json').read_text(encoding='utf8'))
    base.write(ROOT/'reused-decisions.json',decisions);new_metrics,costs=train_missing();reports={}
    for parent in [43,17,29,'WN_DistMult','WN_ComplEx']:
        status('compare',parent=str(parent));fb=isinstance(parent,int);base.N=core.N=11258 if fb else 24567
        if fb:
            queries=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')
            teacher=base.load('/data/lc/untzh/parent_confirm110_20261014/evaluation/teacher.npz')['old_valid'] if parent==43 else base.load(f'/data/lc/untzh/full_valid102_20261014/seed{parent}/teacher.npz')['old_valid']
        else:
            src=Path('/data/lc/untzh')/('complex_transfer118_20261014' if parent=='WN_ComplEx' else 'wn_transfer112_20261014');queries=base.load(src/'evaluation/queries.npz');teacher=base.load(src/'cache/fast-old_valid.npz')['teacher']
        split=base.load(f'/data/lc/untzh/admission_action128_20261014/cache/{parent}-split.npz');held=split['evaluation'];choice=decisions[str(parent)]['historical_reference']['chosen'];assert choice in ['low','half','zero']
        fixed={};metric_cache={}
        for mode in ['low','half','zero']:
            fixed[mode]={}
            for arm in COMMON:
                states=base.load(protected_path(parent,mode,arm));metric=dict(new_valid=base.metrics(states['new_valid'],queries['new_valid']),old_eval=base.metrics(states['old_valid'][held],queries['old_valid'][held]))
                metric['old_eval']['drops']=core.drops(states['old_valid'][held],teacher[held]);fixed[mode][arm]=metric
                metric_cache[(mode,arm)]=metric
        selected={arm:dict(alias=choice,metrics=metric_cache[(choice,arm)]) for arm in COMMON}
        reports[str(parent)]=dict(choice=choice,selected=selected,fixed_controls=fixed,calibration_excluded_from_old_evaluation=True)
        print('PIPELINE129',parent,choice,{a:(s['metrics']['new_valid']['all']['filtered_train_truths']['mrr'],s['metrics']['old_eval']['drops']['filtered_train_truths']['drops']) for a,s in selected.items()},flush=True)
    base.write(ROOT/'summary.json',dict(parents=reports,new_wn_low_metrics=new_metrics,new_wn_low_costs=costs,total_seconds=time.monotonic()-tick,new_training_configs=3,
        cached_protected_evaluations=30,retrospective_pipeline_comparison=True));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
