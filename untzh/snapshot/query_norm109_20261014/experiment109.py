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
base,core,absolute=online.base,online.core,online.absolute
ROOT=Path(__file__).resolve().parent
N=11258;ARM='answer_uniform_normmatched'

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def train(initial,prefix,new,tape,data,space,targets,out):
    setup=time.monotonic();anchor=online.OnlineAnchor(initial,prefix,new,tape);setup_seconds=time.monotonic()-setup
    E,R,opt=core.newmodel(initial,prefix);tick=time.monotonic();records=[]
    for j in range(220,440):
        chosen=data['selections'][j-220]
        loss=base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p,_=online.action.penalty(E,data,chosen,'absolute_full');gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm());alpha=.1*bn/pn if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        reference=anchor[j-220];difference=(E.detach()[N:]-reference).double()
        projected=online.prior.project(difference,space);pnorm=float(projected.norm());target=targets[j-220]
        assert pnorm>0 and np.isfinite(target)
        scale=target/pnorm
        with torch.no_grad():E[N:].copy_(reference+(projected*scale).to(E.dtype))
        actual=(E.detach()[N:]-reference).double();anorm=float(actual.norm())
        changes=(space['vectors']*actual[space['rows']]).sum(-1)
        assert torch.isfinite(E).all()
        records.append(dict(step=j,target_norm=target,actual_norm=anorm,scale=scale,max_score_difference=float(changes.abs().max())))
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    source=online.route.SOURCE;ordinary=torch.load(source/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
    assert torch.equal(anchor.E.detach().cpu().double(),ordinary['E']) and torch.equal(anchor.R.cpu().double(),ordinary['R'])
    directory=out/'models'/ARM;directory.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt');base.write(directory/'steps.json',records)
    report=dict(seconds=time.monotonic()-tick,online_setup_seconds=setup_seconds,shadow_compute_included=True,
                max_norm_error=max(abs(x['target_norm']-x['actual_norm']) for x in records),
                amplified_steps=sum(x['scale']>1 for x in records),max_scale=max(x['scale'] for x in records),
                max_cross_train_score_difference=max(x['max_score_difference'] for x in records),final_norm=records[-1]['actual_norm'],frozen_equal=True,ordinary_endpoint_equal=True)
    base.write(directory/'training.json',report);return report

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert json.loads((fastmodule.ROOT/'summary.json').read_text())['all_equal']
    tick=time.monotonic();status('starting');bundles=[]
    for seed in [29,17]:
        source=Path('/data/lc/untzh')/('risk_increment88_20261014' if seed==29 else 'increment87_20261014')
        out=ROOT/f'seed{seed}';out.mkdir();online.configure(out,source);torch.manual_seed(seed)
        status('prepare',seed=seed);start=time.monotonic()
        if seed==29:
            initial,prefix,old,new,truths,tape,data,model,coefficient,ps=online.route.prepare();online.action.extra_data(truths,data)
        else:initial,prefix,new,tape,truths,data,ps=online.source17.prepare()
        space=online.replay.previous.load_space('/data/lc/untzh/'+('query_space92_20261014' if seed==29 else 'anchor_confirm95_20261014')+'/cache/query-space.pt')
        targetroot='query_budget101_20261014' if seed==29 else 'query_confirm108_20261014'
        steps=json.loads(Path(f'/data/lc/untzh/{targetroot}/models/query_uniform/steps.json').read_text())
        targets=[x['accepted_anchor_displacement_norm'] for x in steps]
        assert len(targets)==220
        costs=dict(preparation_seconds=time.monotonic()-start,target_source=targetroot,historical_assets_not_free=True)
        status('train',seed=seed);costs['training']=train(initial,prefix,new,tape,data,space,targets,out)
        bundles.append((seed,out,truths,costs));base.write(out/'costs.json',costs)
    summaries={}
    for seed,out,truths,costs in bundles:
        online.configure(out,Path('/data/lc/untzh')/('risk_increment88_20261014' if seed==29 else 'increment87_20261014'))
        status('evaluate',seed=seed);fast=fastmodule.FastOld(seed,truths);costs['fast_preparation']=fast.preparation
        queries=dict(fast.queries,new_valid=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['new_valid'])
        m=torch.load(out/'models'/ARM/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda()
        states={};ec={}
        for split in ['old_train','old_valid']:states[split],ec[split]=fast.ranks(E,R,split)
        start=time.monotonic();states['new_valid']=base.ranks(E,R,queries['new_valid'],truths,arm=ARM,split='new_valid');ec['new_valid']=dict(seconds=time.monotonic()-start)
        evaluation=out/'evaluation';evaluation.mkdir();np.savez(evaluation/f'{ARM}.npz',**states)
        metric={s:base.metrics(states[s],q) for s,q in queries.items()}
        for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],fast.teacher[split])
        reference=json.loads(Path('/data/lc/untzh/full_valid102_20261014/summary.json').read_text())['seeds'][str(seed)]['metrics']
        metric['directional_ratios']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/reference['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
        costs['evaluation']=ec;summaries[str(seed)]=dict(metrics={ARM:metric},costs=costs)
        base.write(out/'summary.json',summaries[str(seed)]);print('EVALUATED109',seed,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],flush=True)
    base.write(ROOT/'summary.json',dict(seeds=summaries,total_seconds=time.monotonic()-tick));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
