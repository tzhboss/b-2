"""Selective head/tail role separation, capacity and protection controls."""
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,'/data/lc/untzh/probability67_20261014')
import experiment67 as probability
core=probability.core;ex=probability.ex
ROOT=Path(__file__).resolve().parent
core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={
 'shared_none':('shared','original','none',.1),
 'shared_absolute':('shared','original','absolute',.1),
 **{f'split_{host}_{signal}_{int(rho*10)}':('split',host,signal,rho)
     for host in ['original','mixed'] for signal,rho in [('none',.1),('absolute',.1),('absolute',.3),('mass_sample',.1)]},
 **{f'split_contrastive_{signal}_{int(rho*10)}':('split','contrastive',signal,rho)
     for signal,rho in [('none',.1),('absolute',.3),('absolute',1.),('mass_sample',.3)]},
}
ACTIVE=None
HEAD=None


def model(initial,architecture,prefix=None):
    old=initial['E'][:ex.N].cuda().clone();R=initial['R'].cuda().clone()
    H=torch.nn.Parameter((initial['E'][ex.N:] if prefix is None else prefix['H'][ex.N:]).cuda().clone())
    T=torch.nn.Parameter((initial['E'][ex.N:] if prefix is None else prefix['E'][ex.N:]).cuda().clone()) if architecture=='split' else H
    params=[H,T] if architecture=='split' else [H]
    opt=torch.optim.Adam(params,lr=.007)
    if prefix is not None:opt.load_state_dict(copy.deepcopy(prefix['optimizer']))
    def entities():return torch.cat([old,H]),torch.cat([old,T])
    return H,T,params,opt,R,entities


def learning(H,T,R,new,tape,j,host):
    lo,hi=tape['bounds'][j:j+2]
    p,nh,nt=[torch.as_tensor(a,device='cuda') for a in (new[tape['pos'][lo:hi]],tape['nh'][lo:hi],tape['nt'][lo:hi])]
    h,r,t=p.unbind(1)
    ps=(H[h]*R[r]*T[t]).sum(-1);ns=(H[nh]*R[r[:,None]]*T[nt]).sum(-1)
    ordinary=F.softplus(-ps).mean()+F.softplus(ns).mean()
    contrastive=F.cross_entropy(torch.cat([ps[:,None],ns],dim=1),torch.zeros(len(p),dtype=torch.long,device='cuda'))
    loss=ordinary if host=='original' else contrastive if host=='contrastive' else .5*(ordinary+contrastive)
    # Symmetric regularization applied identically to shared and split controls.
    return loss+.00025*(H[h].square().mean()+T[t].square().mean())+.0005*R[r].square().mean()


def warmup(initial,architecture,new,tape):
    start=time.monotonic();H,T,params,opt,R,entities=model(initial,architecture)
    for j in range(370):
        eh,et=entities();loss=learning(eh,et,R,new,tape,j,'original')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if j==369 or tape['epochs'][j+1]!=tape['epochs'][j]:ex.progress('warmup',architecture=architecture,epoch=int(tape['epochs'][j]))
    eh,et=entities()
    prefix={'H':eh.detach().cpu(),'E':et.detach().cpu(),'R':R.cpu(),'optimizer':opt.state_dict()}
    torch.save(prefix,ROOT/f'prefix10-{architecture}.pt')
    return prefix,time.monotonic()-start


def calibrate(initial,prefix,spec,new,tape,data):
    architecture,host,signal,rho=spec
    if signal=='none':return {'lambda':0.}
    H,T,params,opt,R,entities=model(initial,architecture,prefix)
    rng=np.random.default_rng(17);rows=[]
    for j in range(370,378):
        selected=rng.choice(ex.SIZE,32,replace=False)
        eh,et=entities();loss=learning(eh,et,R,new,tape,j,host)
        grads=torch.autograd.grad(loss,params)
        pen=probability.penalty(et,data,selected,signal)
        gp=torch.autograd.grad(pen,T)[0]
        rows.append([float(grads[-1].double().norm()),float(gp.double().norm())])
    a=np.array(rows);denom=np.sqrt(np.mean(a[:,1]**2))
    return {'lambda':rho*np.sqrt(np.mean(a[:,0]**2))/denom if denom else 0.,'norms':rows,'scope':'candidate-tail learning gradient norm'}


def train(name,spec,initial,prefix,new,tape,data,cal):
    started=time.monotonic();architecture,host,signal,rho=spec
    H,T,params,opt,R,entities=model(initial,architecture,prefix)
    rng=np.random.default_rng(17);records=[]
    for j in range(370,740):
        selected=rng.choice(ex.SIZE,32,replace=False)
        eh,et=entities();loss=learning(eh,et,R,new,tape,j,host)
        grads=list(torch.autograd.grad(loss,params));pen=eh.new_zeros(());alpha=0.
        if signal!='none':
            pen=probability.penalty(et,data,selected,signal)
            gp=torch.autograd.grad(pen,T)[0]
            bn,pn=float(grads[-1].double().norm()),float(gp.double().norm())
            alpha=min(cal['lambda'],rho*bn/pn) if pn else 0.
            grads[-1]=grads[-1]+alpha*gp
        opt.zero_grad(set_to_none=True)
        for p,g in zip(params,grads):p.grad=g
        opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(H).all() or not torch.isfinite(T).all():raise FloatingPointError(name)
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalty':float(pen),'alpha':alpha})
        if j==739 or tape['epochs'][j+1]!=tape['epochs'][j]:ex.progress('train',arm=name,epoch=int(tape['epochs'][j]),seconds=time.monotonic()-started)
    eh,et=entities()
    assert torch.equal(eh[:ex.N].detach().cpu(),initial['E'][:ex.N]) and torch.equal(et[:ex.N].detach().cpu(),initial['E'][:ex.N])
    assert torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save({'H':eh.detach().cpu().double(),'E':et.detach().cpu().double(),'R':R.cpu().double()},out/'epoch-20.pt')
    ex.base.write(out/'steps.json',records)
    result={'seconds':time.monotonic()-started,'architecture':architecture,'host':host,'signal':signal,'rho':rho,
            'trainable_parameters':sum(p.numel() for p in params),'old_head_tail_R_frozen':True}
    ex.base.write(out/'training.json',result);return result


@torch.no_grad()
def ranks(E,R,queries,truths,arm=None,split=None):
    global ACTIVE,HEAD
    if ACTIVE!=arm:
        HEAD=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)['H'].cuda();ACTIVE=arm
    result=[]
    for lo in range(0,len(queries),64):
        a=queries[lo:lo+64];q=torch.as_tensor(a,device='cuda')
        scores=(HEAD[q[:,0]]*R[q[:,1]])@E.T
        gold=scores[torch.arange(len(q),device='cuda'),q[:,2]].clone()
        raw=1+(scores>gold[:,None]).sum(-1)
        rows=[];cols=[]
        for i,(h,r,t) in enumerate(a):
            for c in truths[int(h),int(r)]:
                if c!=int(t):rows.append(i);cols.append(c)
        if rows:scores[rows,cols]=-torch.inf
        filtered=1+(scores>gold[:,None]).sum(-1)
        result.append(torch.stack([raw,filtered],dim=1).cpu().numpy())
        if lo%4096==0:ex.dump('status.json',{'stage':'evaluate','arm':arm,'split':split,'completed':lo+len(a),'total':len(queries)})
    return np.concatenate(result)


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    tick=time.monotonic();data=probability.prepare(initial,old,new,tape)
    costs={'prepare_seconds':time.monotonic()-tick,'warmup':{},'training':{}}
    prefixes={}
    for architecture in ['shared','split']:
        prefixes[architecture],costs['warmup'][architecture]=warmup(initial,architecture,new,tape)
    calibration={name:calibrate(initial,prefixes[spec[0]],spec,new,tape,data) for name,spec in ARMS.items()}
    ex.dump('calibration.json',calibration)
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefixes[spec[0]],new,tape,data,calibration[name]);ex.dump('costs-running.json',costs)
    ex.base.ranks=ranks
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
