"""Matched learning pressure and probability protection comparisons."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,'/data/lc/untzh/probability67_20261014')
import experiment67 as prev
core=prev.core;ex=prev.ex
ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;prev.prev.ROOT=ROOT;core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={
 **{f'mix{int(mu*100)}_{signal}':dict(host=f'mix{mu}',signal=signal,rho=rho)
     for mu,rho in [(.1,.1),(.25,.2),(.5,.3)] for signal in ['none','absolute','mass_sample']},
 **{f'contrastive_{signal}_{int(rho*10)}':dict(host='contrastive',signal=signal,rho=rho)
     for signal in ['absolute','mass_old'] for rho in [.5,1.,2.]},
}
original_learning=prev.learning


def learning(E,R,new,tape,j,host,data):
    if host.startswith('mix'):
        mu=float(host[3:])
        a=original_learning(E,R,new,tape,j,'original',data)
        b=original_learning(E,R,new,tape,j,'contrastive',data)
        return (1-mu)*a+mu*b
    return original_learning(E,R,new,tape,j,host,data)


def train(name,spec,initial,prefix,new,tape,data,cal):
    started=time.monotonic();E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);records=[]
    rho=spec['rho']
    for j in range(370,740):
        selected=rng.choice(ex.SIZE,32,replace=False)
        loss=learning(E,R,new,tape,j,spec['host'],data);gl=torch.autograd.grad(loss,E)[0]
        p=E.new_zeros(());alpha=0.;gp=torch.zeros_like(E)
        if spec['signal']!='none':
            p=prev.penalty(E,data,selected,spec['signal']);gp=torch.autograd.grad(p,E)[0]
            bn,pn=float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
            alpha=min(cal[name]['lambda']*rho/.1,rho*bn/pn) if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(E).all():raise FloatingPointError(name)
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalty':float(p),'alpha':alpha})
        if j==739 or tape['epochs'][j+1]!=tape['epochs'][j]:ex.progress('train',arm=name,epoch=int(tape['epochs'][j]),seconds=time.monotonic()-started)
    assert torch.equal(E[:ex.N].detach().cpu(),initial['E'][:ex.N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save({'E':E.detach().cpu().double(),'R':R.cpu().double()},out/'epoch-20.pt')
    ex.base.write(out/'steps.json',records)
    result={'seconds':time.monotonic()-started,'spec':spec,'frozen_rows_equal':True}
    ex.base.write(out/'training.json',result);return result


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(core.PRIOR/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    tick=time.monotonic();data=prev.prepare(initial,old,new,tape)
    prev.ARMS=ARMS;prev.learning=learning;cal=prev.calibrate(initial,prefix,new,tape,data)
    costs={'prepare_seconds':time.monotonic()-tick,'training':{}}
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,cal);ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
