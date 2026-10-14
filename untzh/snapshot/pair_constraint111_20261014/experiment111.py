import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/online_shadow106_20261014')
import experiment106 as online
sys.path.insert(0,'/data/lc/untzh/fast_old_eval107_20261014')
import experiment107 as fastmodule
base,core=online.base,online.core
ROOT=Path(__file__).resolve().parent
N=11258
ARMS=['pair4','pair16','query_normmatched_pair4']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

@torch.no_grad()
def pair_scores(x,p,r):return (x[p[:,0]]*r*x[p[:,1]]).sum(-1)

@torch.no_grad()
def correct(x,reference,p,r,degree,space,iterations):
    target=pair_scores(reference,p,r);before=pair_scores(x,p,r)-target
    for _ in range(iterations):
        gh=x[p[:,1]]*r;gt=x[p[:,0]]*r
        gap=pair_scores(x,p,r)-target
        weight=gap/(gh.square().sum(-1)+gt.square().sum(-1)).clamp_min(1e-30)
        gradient=torch.zeros_like(x)
        gradient.index_add_(0,p[:,0],weight[:,None]*gh)
        gradient.index_add_(0,p[:,1],weight[:,None]*gt)
        gradient=gradient/degree[:,None]
        x=x-online.prior.project(gradient,space)
    after=pair_scores(x,p,r)-target
    return x,dict(before_rms=float(before.square().mean().sqrt()),after_rms=float(after.square().mean().sqrt()),after_max=float(after.abs().max()))

def train(arm,initial,prefix,new,tape,data,space,reps,inverse,out,targets=None):
    setup=time.monotonic();anchor=online.OnlineAnchor(initial,prefix,new,tape);setup_seconds=time.monotonic()-setup
    E,R,opt=core.newmodel(initial,prefix);rng=np.random.default_rng(1010029);records=[];tick=time.monotonic();correction_seconds=0.
    pair=new[(new[:,0]>=N)&(new[:,2]>=N)]
    p=torch.as_tensor(pair[:,[0,2]]-N,device='cuda');r=R[pair[:,1]].double()
    degree=torch.bincount(p.flatten(),minlength=1876).clamp_min(1).double()
    for j in range(220,440):
        group_ids=rng.choice(len(reps),32,replace=False);chosen=reps[group_ids]
        loss=base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        penalty,_=online.action.penalty(E,data,chosen,'absolute_full');gp=torch.autograd.grad(penalty,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm());alpha=.1*bn/pn if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        reference=anchor[j-220];reference64=reference.double()
        displacement=online.prior.project((E.detach()[N:]-reference).double(),space)
        scale=1.;stats={}
        if arm in ['pair4','pair16']:
            start=time.monotonic();x,stats=correct(reference64+displacement,reference64,p,r,degree,space,4 if arm=='pair4' else 16)
            correction_seconds+=time.monotonic()-start;displacement=x-reference64
        else:
            target=targets[j-220];norm=float(displacement.norm());scale=target/norm if norm else 0.;displacement=displacement*scale
        with torch.no_grad():E[N:].copy_(reference+displacement.to(E.dtype))
        actual=(E.detach()[N:]-reference).double();gap=pair_scores(E.detach()[N:].double(),p,r)-pair_scores(reference64,p,r)
        cross=(space['vectors']*actual[space['rows']]).sum(-1)
        assert torch.isfinite(E).all() and torch.isfinite(gap).all()
        records.append(dict(step=j,norm=float(actual.norm()),scale=scale,group_ids=group_ids.tolist(),pair_rms=float(gap.square().mean().sqrt()),pair_max=float(gap.abs().max()),cross_max=float(cross.abs().max()),**stats))
    ordinary=torch.load(online.route.SOURCE/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
    assert torch.equal(anchor.E.detach().cpu().double(),ordinary['E']) and torch.equal(anchor.R.cpu().double(),ordinary['R'])
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    directory=out/'models'/arm;directory.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt');base.write(directory/'steps.json',records)
    report=dict(seconds=time.monotonic()-tick,online_setup_seconds=setup_seconds,correction_seconds_included=correction_seconds,
                pair_direction_facts=len(pair),max_pair_rms=max(x['pair_rms'] for x in records),final_pair_rms=records[-1]['pair_rms'],
                final_pair_max=records[-1]['pair_max'],max_cross_score_difference=max(x['cross_max'] for x in records),
                final_norm=records[-1]['norm'],max_scale=max(x['scale'] for x in records),amplified_steps=sum(x['scale']>1 for x in records),
                frozen_equal=True,ordinary_endpoint_equal=True,shadow_compute_included=True)
    if arm=='query_normmatched_pair4':report['max_norm_error']=max(abs(x['norm']-t) for x,t in zip(records,targets))
    base.write(directory/'training.json',report);return report,[x['norm'] for x in records]

def fast43(initial,truths):
    source=Path('/data/lc/untzh/parent_confirm110_20261014');fast=fastmodule.FastOld.__new__(fastmodule.FastOld)
    fast.seed=43;fast.truths=truths;fast.oldE=initial['E'][:N].cuda().double();fast.R=initial['R'].cuda().double()
    fast.queries=base.load(source/'evaluation/queries.npz');fast.queries.pop('new_valid')
    fast.teacher=base.load(source/'evaluation/teacher.npz');fast.cache={}
    for split in ['old_train','old_valid']:
        z=base.load(source/'cache'/f'fast-{split}.npz')
        fast.cache[split]=dict(gold=torch.as_tensor(z['gold'],device='cuda'),rows=z['rows'],cols=z['cols'])
    return fast

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();status('starting');bundles=[]
    for seed in [43,17,29]:
        source=Path('/data/lc/untzh')/({43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[seed])
        out=ROOT/f'seed{seed}';out.mkdir();online.configure(out,source);torch.manual_seed(seed);status('prepare',seed=seed);start=time.monotonic()
        initial=torch.load(source/'cache/initial.pt',map_location='cpu',weights_only=True);prefix=torch.load(source/'cache/prefix10.pt',map_location='cpu',weights_only=True)
        tape=base.load(source/'cache/learning.npz');z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')
        new=z['new'];truths=core.truthmap(z['old'],np.concatenate([new,z['extra']]))
        c=base.load(source/'cache/cohort.npz');teacher=base.load(source/'cache/teacher-train.npz')['old_train']
        data=dict(q=torch.as_tensor(c['q'],device='cuda'),candidates=torch.as_tensor(c['candidates'],device='cuda'),fixed=np.column_stack([teacher[c['ids'],1],np.zeros((len(c['ids']),7))]))
        online.action.extra_data(truths,data)
        spacepath={43:source,17:Path('/data/lc/untzh/anchor_confirm95_20261014'),29:Path('/data/lc/untzh/query_space92_20261014')}[seed]
        space=online.replay.previous.load_space(spacepath/'cache/query-space.pt')
        groups,reps,inverse=np.unique(c['queries'][:,:2],axis=0,return_index=True,return_inverse=True)
        costs=dict(preparation_seconds=time.monotonic()-start,historical_assets_not_free=True,training={});targets=None
        for arm in ARMS:
            status('train',seed=seed,arm=arm);costs['training'][arm],norms=train(arm,initial,prefix,new,tape,data,space,reps,inverse,out,targets)
            if arm=='pair4':targets=norms
            base.write(out/'costs-running.json',costs)
        bundles.append((seed,out,source,initial,truths,costs))
    # All nine models committed before new VALID evaluation.
    summaries={};full=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')
    for seed,out,source,initial,truths,costs in bundles:
        online.configure(out,source);status('evaluation_prepare',seed=seed);start=time.monotonic()
        fast=fast43(initial,truths) if seed==43 else fastmodule.FastOld(seed,truths)
        costs['fast_prepare_seconds']=time.monotonic()-start
        reference=json.loads((source/'summary.json').read_text())['metrics'] if seed==43 else json.loads(Path('/data/lc/untzh/full_valid102_20261014/summary.json').read_text())['seeds'][str(seed)]['metrics']
        query_reference=reference['query_uniform'] if seed==43 else json.loads(Path('/data/lc/untzh/query_confirm108_20261014/summary.json').read_text())['metrics']['query_uniform'] if seed==17 else reference['query_uniform']
        queries=dict(fast.queries,new_valid=full['new_valid']);evaluation=out/'evaluation';evaluation.mkdir();metrics={};ec={}
        for arm in ARMS:
            status('evaluate',seed=seed,arm=arm);m=torch.load(out/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda();states={};cost={}
            for split in ['old_train','old_valid']:states[split],cost[split]=fast.ranks(E,R,split)
            t=time.monotonic();states['new_valid']=base.ranks(E,R,full['new_valid'],truths,arm=arm,split='new_valid');cost['new_valid']=dict(seconds=time.monotonic()-t)
            np.savez(evaluation/f'{arm}.npz',**states);metric={s:base.metrics(states[s],q) for s,q in queries.items()}
            for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],fast.teacher[split])
            metric['directional_ratios']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/reference['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
            metrics[arm]=metric;ec[arm]=cost
            print('EVALUATED111',seed,arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],metric['directional_ratios'],flush=True)
        costs['evaluation']=ec;summaries[str(seed)]=dict(metrics=metrics,query_uniform_reference=query_reference,costs=costs)
        base.write(out/'summary.json',summaries[str(seed)])
    base.write(ROOT/'summary.json',dict(seeds=summaries,total_seconds=time.monotonic()-tick));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
