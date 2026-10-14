import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/complex_transfer118_20261014')
import experiment118 as cx
source,base,core=cx.source,cx.base,cx.core
ROOT=Path(__file__).resolve().parent
N,TOTAL=24567,28660
INITS=['zero','random_half']
ARMS=['absolute','answer_anchor','full_replay_joint']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def train(name,arm,initial,prefix,ordinary,new,tape,data,space,replay,coefficient,out):
    tick=time.monotonic();E,R,opt=source.model(initial,prefix);shadow=None if arm=='absolute' else source.model(initial,prefix);records=[]
    learning=cx.learning if name=='ComplEx' else base.learning
    for j in range(110,220):
        k=j-110;gl=torch.autograd.grad(learning(E,R,new,tape,j),E)[0]
        penalty=cx.replay_penalty(E,replay,k) if arm=='full_replay_joint' else cx.protection(E,data,data['selections'][k],arm!='absolute')
        gp=torch.autograd.grad(penalty,E)[0];bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm())
        alpha=min(coefficient,.1*bn/pn) if arm=='absolute' and pn else .1*bn/pn if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step();cross=None;norm=None
        if shadow:
            se,sr,so=shadow;so.zero_grad(set_to_none=True);learning(se,sr,new,tape,j).backward();so.step();ref=se.detach()[N:]
            delta=(E.detach()[N:]-ref).double()
            if arm=='answer_anchor':
                accepted=source.space_module.project(delta,space)
                with torch.no_grad():E[N:].copy_(ref+accepted.to(E.dtype))
            actual=(E.detach()[N:]-ref).double();cross=float((space['vectors']*actual[space['rows']]).sum(-1).abs().max());norm=float(actual.norm())
        assert torch.isfinite(E).all();records.append(dict(step=j,gradient_ratio=alpha*pn/bn if bn else 0.,cross_max=cross,norm=norm))
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    if shadow:assert torch.equal(shadow[0].detach().cpu().double(),ordinary['E']) and torch.equal(shadow[1].cpu().double(),ordinary['R'])
    directory=out/'models'/arm;directory.mkdir(parents=True,exist_ok=True);torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt')
    report=dict(seconds=time.monotonic()-tick,steps=110,shadow_compute_included=bool(shadow),ordinary_endpoint_equal=True if shadow else None,
        mean_gradient_ratio=float(np.mean([v['gradient_ratio'] for v in records])),final_norm=records[-1]['norm'],
        max_cross_difference=max((v['cross_max'] for v in records if v['cross_max'] is not None),default=None),frozen_equal=True)
    base.write(directory/'steps.json',records);base.write(directory/'training.json',report);return report

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    base.ROOT=core.ROOT=ROOT;base.N=core.N=N;base.TOTAL=core.TOTAL=TOTAL
    tick=time.monotonic();status('prepare');z=base.load('/data/lc/untzh/wn_transfer112_20261014/cache/train.npz');new=z['new'];truths=core.truthmap(z['old'],new)
    tape=base.load('/data/lc/untzh/wn_transfer112_20261014/cache/learning.npz');bundles=[]
    for name,folder in [('DistMult','wn_transfer112_20261014'),('ComplEx','complex_transfer118_20261014')]:
        src=Path('/data/lc/untzh')/folder;c=base.load(src/'cache/cohort.npz')
        data={k:torch.as_tensor(c[k],device='cuda') for k in ['q','candidates','legal']};data['selections']=c['selections']
        space=torch.load(src/'cache/query-space.pt',map_location='cpu',weights_only=True)
        for g in space['groups']:g['indices']=g['indices'].cuda();g['basis']=g['basis'].cuda()
        space['rows']=space['rows'].cuda();space['vectors']=space['vectors'].cuda()
        replay={k:torch.as_tensor(v,device='cuda') for k,v in base.load(src/'cache/full-replay.npz').items() if k!='positions'}
        for init_name in INITS:
            out=ROOT/name/init_name;out.mkdir(parents=True,exist_ok=True);st=time.monotonic()
            path=Path(f'/data/lc/untzh/wn_init122_20261014/{name}/models/{init_name}')
            initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=True);prefix=torch.load(path/'prefix10.pt',map_location='cpu',weights_only=True);ordinary=torch.load(path/'epoch-20.pt',map_location='cpu',weights_only=True)
            E,R,opt=source.model(initial,prefix);norms=[];learning=cx.learning if name=='ComplEx' else base.learning
            for k in range(8):
                gl=torch.autograd.grad(learning(E,R,new,tape,110+k),E)[0];gp=torch.autograd.grad(cx.protection(E,data,data['selections'][k],False),E)[0]
                norms.append([float(gl[N:].double().norm()),float(gp[N:].double().norm())])
            a=np.array(norms);coefficient=.1*np.sqrt(np.mean(a[:,0]**2))/np.sqrt(np.mean(a[:,1]**2));assert np.isfinite(coefficient)
            base.write(out/'calibration.json',dict(norms=norms,coefficient=coefficient));costs=dict(preparation_seconds=time.monotonic()-st,historical_assets_not_free=True,training={})
            for arm in ARMS:status('train',model=name,init=init_name,arm=arm);costs['training'][arm]=train(name,arm,initial,prefix,ordinary,new,tape,data,space,replay,coefficient,out)
            bundles.append((name,init_name,src,out,costs))
    # All twelve new protected models committed before current VALID read.
    reports={}
    for name,init_name,src,out,costs in bundles:
        status('evaluation_prepare',model=name,init=init_name);queries=base.load(src/'evaluation/queries.npz');cache={s:base.load(src/'cache'/f'fast-{s}.npz') for s in ['old_train','old_valid']}
        fast=cx.fast_ranks if name=='ComplEx' else source.fast_ranks;rank=cx.ranks if name=='ComplEx' else base.ranks
        ordinary=json.loads(Path('/data/lc/untzh/wn_init122_20261014/'+name+'/summary.json').read_text(encoding='utf8'))['metrics'][init_name]
        existing=json.loads((src/'summary.json').read_text(encoding='utf8'))['metrics']
        if name=='DistMult':low=json.loads(Path('/data/lc/untzh/init_onset115_20261014/summary.json').read_text(encoding='utf8'))['metrics']
        else:low=existing
        metrics={};ec={};directory=out/'evaluation';directory.mkdir(exist_ok=True)
        for arm in ARMS:
            status('evaluate',model=name,init=init_name,arm=arm);m=torch.load(out/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda();states={};cost={}
            for split in ['old_train','old_valid']:states[split],cost[split]=fast(E,R,queries[split],cache[split],truths)
            st=time.monotonic();states['new_valid']=rank(E,R,queries['new_valid'],truths);cost['new_valid']=dict(seconds=time.monotonic()-st);np.savez(directory/f'{arm}.npz',**states)
            metric={s:base.metrics(states[s],q) for s,q in queries.items()}
            for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],cache[split]['teacher'])
            metric['directional_ratios_to_own_none']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/ordinary['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
            metrics[arm]=metric;ec[arm]=cost;base.write(directory/'metrics-running.json',metrics)
            print('EVALUATED127',name,init_name,arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],metric['directional_ratios_to_own_none'],flush=True)
        costs['evaluation']=ec;reports[name+'-'+init_name]=dict(metrics=metrics,reused_own_none=ordinary,reused_low_controls=low,reused_standard_controls=existing,costs=costs)
        base.write(out/'summary.json',reports[name+'-'+init_name])
    base.write(ROOT/'summary.json',dict(cases=reports,total_seconds=time.monotonic()-tick));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
