import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import copy
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/wn_transfer112_20261014')
import experiment112 as source
base,core=source.base,source.core
ROOT=Path(__file__).resolve().parent
SRC=Path('/data/lc/untzh/wn_transfer112_20261014')
N=source.N
ARMS=['lowinit_none','lowinit_query_late','standard_query_start0']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def save(E,R,arm):
    out=ROOT/'models'/arm;out.mkdir(parents=True);torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic();base.ROOT=ROOT;base.N=N;base.TOTAL=source.TOTAL;status('prepare');(ROOT/'cache').mkdir()
    initial=torch.load(SRC/'cache/initial.pt',map_location='cpu',weights_only=True);low=copy.deepcopy(initial);low['E'][N:]*=.25;torch.save(low,ROOT/'cache/low-initial.pt')
    z=base.load(SRC/'cache/train.npz');old,new=z['old'],z['new'];truths=core.truthmap(old,new);t=base.load(SRC/'cache/learning.npz');first=int(np.flatnonzero(t['epochs']==11)[0])
    c=base.load(SRC/'cache/cohort.npz');data=dict(q=torch.as_tensor(c['q'],device='cuda'),legal=torch.as_tensor(c['legal'],device='cuda'))
    space=torch.load(SRC/'cache/query-space.pt',map_location='cpu',weights_only=True)
    for g in space['groups']:g['indices']=g['indices'].cuda();g['basis']=g['basis'].cuda()
    space['rows']=space['rows'].cuda();space['vectors']=space['vectors'].cuda()
    prefix_tick=time.monotonic();E,R,opt=source.model(low)
    for j in range(first):loss=base.learning(E,R,new,t,j);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    prefix=dict(E=E.detach().cpu(),R=R.cpu(),optimizer=copy.deepcopy(opt.state_dict()));torch.save(prefix,ROOT/'cache/low-prefix10.pt')
    costs=dict(low_prefix_seconds=time.monotonic()-prefix_tick,initialization_scale=.25,historical_assets_not_free=True,training={})
    for arm in ARMS:
        status('train',arm=arm);st=time.monotonic();start=0 if arm=='standard_query_start0' else first;init=initial if start==0 else low;pref=None if start==0 else prefix
        E,R,opt=source.model(init,pref);shadow=source.model(init,pref) if arm!='lowinit_none' else None;rng=np.random.default_rng(1010029);records=[]
        for j in range(start,len(t['epochs'])):
            loss=base.learning(E,R,new,t,j);gl=torch.autograd.grad(loss,E)[0];bn=float(gl[N:].double().norm());pn=0.;alpha=0.
            if shadow:
                chosen=c['representatives'][rng.choice(len(c['representatives']),32,replace=False)]
                p=source.protection(E,data,chosen,True);gp=torch.autograd.grad(p,E)[0];pn=float(gp[N:].double().norm());alpha=.1*bn/pn if pn else 0.;gradient=gl+alpha*gp
            else:gradient=gl
            opt.zero_grad(set_to_none=True);E.grad=gradient;opt.step();norm=None;cross=None
            if shadow:
                se,sr,so=shadow;sl=base.learning(se,sr,new,t,j);so.zero_grad(set_to_none=True);sl.backward();so.step();reference=se.detach()[N:]
                accepted=source.space_module.project((E.detach()[N:]-reference).double(),space)
                with torch.no_grad():E[N:].copy_(reference+accepted.to(E.dtype))
                actual=(E.detach()[N:]-reference).double();norm=float(actual.norm());cross=float((space['vectors']*actual[space['rows']]).sum(-1).abs().max())
            assert torch.isfinite(E).all();records.append(dict(step=j,gradient_ratio=alpha*pn/bn if bn else 0.,norm=norm,cross_max=cross))
        assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
        if shadow:
            ordinary_path=SRC/'models/none/epoch-20.pt' if start==0 else ROOT/'models/lowinit_none/epoch-20.pt'
            ordinary=torch.load(ordinary_path,map_location='cpu',weights_only=True)
            assert torch.equal(shadow[0].detach().cpu().double(),ordinary['E']) and torch.equal(shadow[1].cpu().double(),ordinary['R'])
        save(E,R,arm);base.write(ROOT/'models'/arm/'steps.json',records)
        costs['training'][arm]=dict(seconds=time.monotonic()-st,steps=len(records),shadow_compute_included=bool(shadow),final_norm=records[-1]['norm'],frozen_equal=True,ordinary_endpoint_equal=True if shadow else None);base.write(ROOT/'costs-running.json',costs)
    # All three models committed before current VALID evaluation.
    status('evaluation_prepare');queries=base.load(SRC/'evaluation/queries.npz');teachers={s:base.load(SRC/'cache'/f'fast-{s}.npz') for s in ['old_train','old_valid']};metrics={};ec={};out=ROOT/'evaluation';out.mkdir()
    lowinitial,cost0=source.fast_ranks(low['E'].cuda().double(),low['R'].cuda().double(),queries['old_valid'],teachers['old_valid'],truths)
    lowinitial_report=core.drops(lowinitial,teachers['old_valid']['teacher']);np.savez(out/'low-expanded-initial.npz',old_valid=lowinitial);base.write(out/'low-expanded-initial.json',dict(drops=lowinitial_report,cost=cost0))
    reference=source.core.json.loads((SRC/'summary.json').read_text())['metrics'];st=time.monotonic()
    for arm in ARMS:
        status('evaluate',arm=arm);m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda();states={};cost={}
        for split in ['old_train','old_valid']:states[split],cost[split]=source.fast_ranks(E,R,queries[split],teachers[split],truths)
        s=time.monotonic();states['new_valid']=base.ranks(E,R,queries['new_valid'],truths);cost['new_valid']=dict(seconds=time.monotonic()-s);np.savez(out/f'{arm}.npz',**states)
        metric={s:base.metrics(states[s],q) for s,q in queries.items()}
        for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],teachers[split]['teacher'])
        metric['directional_ratios_to_standard']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/reference['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
        metrics[arm]=metric;ec[arm]=cost;print('EVALUATED115',arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],metric['directional_ratios_to_standard'],flush=True)
    costs.update(evaluation=ec,evaluation_seconds=time.monotonic()-st,total_seconds=time.monotonic()-tick)
    base.write(ROOT/'summary.json',dict(metrics=metrics,standard_controls=reference,low_initial_drops=lowinitial_report,costs=costs));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
