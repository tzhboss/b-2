"""Relation-stratified protection with matched strength/learning controls."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,'/data/lc/untzh/coverage74_20261014')
import experiment74 as prev
core=prev.core;ex=prev.ex
ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={
 **{f'relation_w{weight}_rho{rho}':(1.,weight,rho)
     for weight in [1.,1.1,1.2] for rho in [.1,.15,.2]},
 **{f'power{power}_rho{rho}':(power,1.,rho) for power in [.5,.75] for rho in [.1,.15,.2]},
 **{f'uniform_w{weight}_rho{rho}':(0.,weight,rho) for weight in [1.,1.1,1.2] for rho in [.15,.2]},
}


def learning(E,R,new,tape,j,weight):
    if weight==1.:return ex.base.learning(E,R,new,tape,j)
    lo,hi=tape['bounds'][j:j+2]
    p,nh,nt=[torch.as_tensor(a,device='cuda') for a in (new[tape['pos'][lo:hi]],tape['nh'][lo:hi],tape['nt'][lo:hi])]
    h,r,t=p.unbind(1);ps=(E[h]*R[r]*E[t]).sum(-1);ns=(E[nh]*R[r[:,None]]*E[nt]).sum(-1)
    w=torch.where((h<ex.N)&(t>=ex.N),weight,1.)
    return ((F.softplus(-ps)+F.softplus(ns).mean(-1))*w).mean()+.0005*(E[h].square().mean()+R[r].square().mean())


def sampling(data,power,rng):
    if power==0.:return rng.choice(len(data['eligible']),32,replace=False)
    p=data['relation_prob']**power;p/=p.sum()
    return rng.choice(len(p),32,replace=False,p=p)


def calibration(initial,prefix,new,tape,data,spec):
    power,weight,rho=spec;E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);rows=[]
    for j in range(370,378):
        selected=sampling(data,power,rng)
        gl=torch.autograd.grad(learning(E,R,new,tape,j,weight),E)[0]
        gp=torch.autograd.grad(prev.penalty(E,data,selected),E)[0]
        rows.append([float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())])
    a=np.array(rows)
    return {'lambda':rho*np.sqrt(np.mean(a[:,0]**2))/np.sqrt(np.mean(a[:,1]**2)),'norms':rows}


def train(name,spec,initial,prefix,new,tape,data,cal):
    start=time.monotonic();power,weight,rho=spec;E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);records=[]
    for j in range(370,740):
        selected=sampling(data,power,rng)
        loss=learning(E,R,new,tape,j,weight);gl=torch.autograd.grad(loss,E)[0]
        p=prev.penalty(E,data,selected);gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
        alpha=min(cal['lambda'],rho*bn/pn) if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(E).all():raise FloatingPointError(name)
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalty':float(p),'alpha':alpha})
        if j==739 or tape['epochs'][j+1]!=tape['epochs'][j]:ex.progress('train',arm=name,epoch=int(tape['epochs'][j]),seconds=time.monotonic()-start)
    assert torch.equal(E[:ex.N].detach().cpu(),initial['E'][:ex.N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save({'E':E.detach().cpu().double(),'R':R.cpu().double()},out/'epoch-20.pt')
    ex.base.write(out/'steps.json',records)
    result={'seconds':time.monotonic()-start,'spec':spec,'frozen_rows_equal':True}
    ex.base.write(out/'training.json',result);return result


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(core.PRIOR/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    data,prep=prev.prepare(initial,old,new)
    # Save the large shared preparation once for future fair comparisons.
    np.savez(ROOT/'shared-cache.npz',q=data['q'].cpu().numpy(),candidates=data['candidates'].cpu().numpy(),
             eligible=data['eligible'],queries=data['queries'],relation_prob=data['relation_prob'])
    cal={name:calibration(initial,prefix,new,tape,data,spec) for name,spec in ARMS.items()}
    ex.dump('calibration.json',cal);costs={'preparation':prep,'training':{}}
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,cal[name]);ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
