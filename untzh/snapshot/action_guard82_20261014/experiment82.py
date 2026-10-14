"""Direct candidate-action cost guard using actual current Adam proposals."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import sys
import time
from pathlib import Path
import collections
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/blend_confirm80_20261014')
import experiment80 as core
base=core.base;ex=core.ex;ROOT=Path(__file__).resolve().parent
core.ROOT=ROOT;base.ROOT=ROOT
ARMS=['absolute_guard','blend_guard','absolute_meanmatched','blend_meanmatched']

def prepare(initial,new,t):
    tick=time.monotonic();z=base.load('/data/lc/untzh/risk58_20261013/cache/cohort.npz')
    scale=base.load('/data/lc/untzh/normalized77_20261014/scale-cache.npz')
    d={k:torch.as_tensor(z[k],device='cuda') for k in ['q','candidates','gold','tau']}
    d['logZ']=torch.as_tensor(scale['cohort_logZ'],device='cuda')
    d['legal_newcount']=torch.as_tensor(scale['legal_newcount'][scale['cohort_positions']],device='cuda')
    rng=np.random.default_rng(17);d['selections']=np.stack([rng.choice(core.SIZE,32,replace=False) for _ in range(370)])
    # Constant TRAIN probes with first-epoch legal negatives from the existing tape.
    inverse=np.empty(len(new),np.int64);inverse[t['pos'][:len(new)]]=np.arange(len(new))
    rng=np.random.default_rng(820017);probes=[];payload={}
    masks=[(new[:,0]<core.N)&(new[:,2]>=core.N),(new[:,0]>=core.N)&(new[:,2]<core.N),(new[:,0]>=core.N)&(new[:,2]>=core.N)]
    for direction,mask in zip(core.DIRECTIONS,masks):
        ids=rng.choice(np.flatnonzero(mask),256,replace=False);pos=inverse[ids]
        arrays=dict(p=new[ids],nh=t['nh'][pos],nt=t['nt'][pos]);payload.update({direction+'_'+k:v for k,v in arrays.items()})
        probes.append({k:torch.as_tensor(v,device='cuda',dtype=torch.long) for k,v in arrays.items()})
    np.savez(ROOT/'train-probes.npz',**payload)
    return d,probes,dict(seconds=time.monotonic()-tick,cohort='058',scale='077',probe_answers_per_direction=256)

@torch.no_grad()
def probe_losses(E,R,probes):
    values=[]
    for z in probes:
        h,r,t=z['p'].unbind(1)
        ps=(E[h]*R[r]*E[t]).sum(-1);ns=(E[z['nh']]*R[r[:,None]]*E[z['nt']]).sum(-1)
        values.append(F.softplus(-ps).mean()+F.softplus(ns).mean()+.0005*(E[h].square().mean()+R[r].square().mean()))
    return torch.stack(values).double()

@torch.no_grad()
def old_proxy(E,d,chosen):
    s=(d['q'][chosen,None]*E[d['candidates'][chosen]].double()).sum(-1)
    return float(F.softplus((s-d['gold'][chosen,None])/d['tau'][chosen,None]).mean())

def train(name,kind,guard,multiplier,initial,prefix,new,t,d,probes,cal):
    tick=time.monotonic();E,R,opt=core.newmodel(initial,prefix);records=[];selection_seconds=0.;proposals=0
    for j in range(370,740):
        gl=torch.autograd.grad(base.learning(E,R,new,t,j),E)[0]
        bn=float(gl[core.N:].double().norm());chosen=d['selections'][j-370];gp=torch.zeros_like(gl)
        for s,fraction in [('absolute',1.)] if kind=='absolute' else [('absolute',.5),('mass',.5)]:
            p=core.penalty(E,d,chosen,s);g=torch.autograd.grad(p,E)[0];pn=float(g[core.N:].double().norm())
            alpha=min(cal[s],.1*bn/pn) if pn else 0.;gp+=fraction*alpha*g
        selected=multiplier;ratios=None;old_improvement=None;attempts=[]
        if guard:
            torch.cuda.synchronize();st=time.monotonic()
            host=E.detach()+ex.proposal(E,opt,gl);proposals+=1
            hostloss=probe_losses(host,R,probes);hostold=old_proxy(host,d,chosen)
            selected=0.
            for fraction in [1.,.5,.25,.125]:
                trial=E.detach()+ex.proposal(E,opt,gl+fraction*gp);proposals+=1
                v=probe_losses(trial,R,probes);rat=(v/hostloss).cpu().numpy();op=old_proxy(trial,d,chosen)
                attempts.append(dict(multiplier=fraction,ratios=rat.tolist(),old_proxy_gain=hostold-op))
                assert np.isfinite(rat).all() and np.isfinite(op)
                if np.all(rat<=1.001) and op<hostold-1e-12:
                    selected=fraction;ratios=rat.tolist();old_improvement=hostold-op;break
            torch.cuda.synchronize();selection_seconds+=time.monotonic()-st
        opt.zero_grad(set_to_none=True);E.grad=gl+selected*gp;opt.step()
        assert torch.isfinite(E).all()
        ratio=float((selected*gp)[core.N:].double().norm())/bn if bn else 0.
        assert ratio<=.1*(1+1e-5)
        records.append(dict(step=j,multiplier=selected,probe_loss_ratios=ratios,old_proxy_gain=old_improvement,attempts=attempts,gradient_ratio=ratio))
        if j==739 or t['epochs'][j+1]!=t['epochs'][j]:core.progress('train',arm=name,epoch=int(t['epochs'][j]),seconds=time.monotonic()-tick)
    assert torch.equal(E[:core.N].cpu(),initial['E'][:core.N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt');base.write(out/'steps.json',records)
    histogram=collections.Counter(x['multiplier'] for x in records)
    result=dict(seconds=time.monotonic()-tick,selection_seconds_included=selection_seconds,proposals=proposals,
        mean_multiplier=float(np.mean([x['multiplier'] for x in records])),histogram=dict(histogram),frozen_equal=True)
    base.write(out/'training.json',result);return result

def evaluate(old,new):
    tick=time.monotonic();tr=core.truthmap(old,new)
    valid=base.load(core.SRC/'entity_refresh43_20261012/artifacts/data/validation.npz')
    A=base.load(core.SRC/'replay_strength53_20261013/artifacts/evaluation/A.npz')
    p=json.loads(Path('/data/lc/untzh/risk58_20261013/evaluation/metrics.json').read_text())
    b=json.loads(Path('/data/lc/untzh/normalized77_20261014/evaluation/metrics.json').read_text())
    report=dict(none=p['new_only'],absolute_fixed=p['uniform'],blend_fixed=b['cohort_blend0.5'])
    out=ROOT/'evaluation';out.mkdir()
    for arm in ARMS:
        m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda()
        qs=dict(old_train=old,old_valid=valid['old'],new_valid=valid['new'])
        states={k:base.ranks(E,R,q,tr,arm=arm,split=k) for k,q in qs.items()};np.savez(out/f'{arm}.npz',**states)
        report[arm]={k:base.metrics(states[k],q) for k,q in qs.items()}
        for k in ['old_train','old_valid']:report[arm][k]['drops']=core.drops(states[k],A[k])
        core.progress('evaluated',arm=arm,new_mrr=report[arm]['new_valid']['all']['filtered_train_truths']['mrr'],train_drops=report[arm]['old_train']['drops']['filtered_train_truths']['drops'])
    for arm,item in report.items():item['directional_ratios']=[item['new_valid'][x]['filtered_train_truths']['mrr']/report['none']['new_valid'][x]['filtered_train_truths']['mrr'] for x in core.DIRECTIONS]
    core.dump('evaluation/metrics.json',report);return report,time.monotonic()-tick

def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial=torch.load(core.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load('/data/lc/untzh/risk58_20261013/cache/prefix10.pt',map_location='cpu',weights_only=True)
    z=base.load(core.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=z['old'],z['new']
    t=base.load(core.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz');d,probes,pc=prepare(initial,new,t)
    cal,cs=core.calibrate(initial,prefix,new,t,d)
    prior=json.loads(Path('/data/lc/untzh/normalized77_20261014/calibration.json').read_text())['cohort_blend0.5']
    differences={s:abs(cal[s]-prior[s]['lambda']) for s in cal};assert max(differences.values())<1e-10
    core.dump('calibration-reuse-check.json',dict(differences=differences,scope='same cache/gradient coefficient only; not reranking controls'))
    costs=dict(preparation=pc,calibration_seconds=cs,training={})
    for kind in ['absolute','blend']:
        guard=kind+'_guard';matched=kind+'_meanmatched'
        costs['training'][guard]=train(guard,kind,True,1.,initial,prefix,new,t,d,probes,cal)
        costs['training'][matched]=train(matched,kind,False,costs['training'][guard]['mean_multiplier'],initial,prefix,new,t,d,probes,cal)
        core.dump('costs-running.json',costs)
    report,es=evaluate(old,new);costs['evaluation_seconds']=es;costs['total_seconds']=time.monotonic()-tick
    core.dump('summary.json',dict(metrics=report,costs=costs));core.progress('complete',seconds=costs['total_seconds'])

if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
