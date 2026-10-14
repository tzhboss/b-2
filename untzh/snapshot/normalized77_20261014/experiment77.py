"""Score-scale normalization and mass/absolute gradient combination controls."""
import collections
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,'/data/lc/untzh/stratified75_20261014')
import experiment75 as prev
core=prev.core;ex=prev.ex
ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={
 **{f'{pool}_{signal}':(pool,signal,0.,1.) for pool in ['cohort','global','relation'] for signal in ['qnorm','std']},
 **{f'cohort_blend{mix}':('cohort','blend',mix,1.) for mix in [.25,.5,.75]},
 **{f'relation_{signal}_weighted':('relation',signal,0.,1.1) for signal in ['qnorm','std']},
 'cohort_linear':('cohort','linear',0.,1.),
}


def prepare(initial,old,new):
    tick=time.monotonic();cached=ex.base.load('/data/lc/untzh/stratified75_20261014/shared-cache.npz')
    data={'q':torch.as_tensor(cached['q'],device='cuda'),'candidates':torch.as_tensor(cached['candidates'],device='cuda'),
          'eligible':cached['eligible'],'queries':cached['queries'],'relation_prob':cached['relation_prob']}
    cohort=ex.base.load(core.PRIOR/'cache/cohort.npz');positions=np.searchsorted(cached['eligible'],cohort['ids'])
    data['cohort']=positions
    data['qnorm']=data['q'].norm(dim=-1).clamp_min(1e-8)
    # Score std from teacher covariance; no per-query old-pool scan required.
    e=initial['E'][:ex.N].cuda().double();centered=e-e.mean(0)
    covariance=centered.T@centered/(len(e)-1)
    data['std']=((data['q']@covariance)*data['q']).sum(-1).clamp_min(1e-16).sqrt()
    truth=collections.defaultdict(set)
    for h,r,t in np.concatenate([old,new]):truth[int(h),int(r)].add(int(t))
    logzs=[]
    with torch.no_grad():
        for lo in range(0,len(positions),128):
            picked=positions[lo:lo+128];s=data['q'][picked]@e.T;rows=[];cols=[]
            for i,(h,r,t) in enumerate(data['queries'][picked]):
                for c in truth[int(h),int(r)]:
                    if c<ex.N and c!=int(t):rows.append(i);cols.append(c)
            s[rows,cols]=-torch.inf;logzs.append(torch.logsumexp(s,dim=-1))
    data['logZ']=torch.zeros(len(cached['eligible']),device='cuda',dtype=torch.float64)
    data['logZ'][positions]=torch.cat(logzs)
    data['legal_newcount']=torch.as_tensor([ex.TOTAL-ex.N-sum(c>=ex.N for c in truth[int(h),int(r)]) for h,r,t in data['queries']],device='cuda')
    np.savez(ROOT/'scale-cache.npz',qnorm=data['qnorm'].cpu().numpy(),std=data['std'].cpu().numpy(),cohort_positions=positions,
             cohort_logZ=data['logZ'][positions].cpu().numpy(),legal_newcount=data['legal_newcount'].cpu().numpy())
    report={'seconds':time.monotonic()-tick,'cache_reused':'075 shared-cache','std_rule':'teacher entity covariance',
        'qnorm_quantiles':np.quantile(data['qnorm'].cpu().numpy(),[0,.5,1]).tolist(),
        'std_quantiles':np.quantile(data['std'].cpu().numpy(),[0,.5,1]).tolist()}
    ex.dump('preparation.json',report);return data,report


def select(data,pool,rng):
    if pool=='cohort':return rng.choice(data['cohort'],32,replace=False)
    return prev.sampling(data,1. if pool=='relation' else 0.,rng)


def penalties(E,data,selected,signal):
    q=data['q'][selected];ids=data['candidates'][selected]
    scores=(q[:,None]*E[ids].double()).sum(-1)
    if signal=='qnorm':return F.softplus(scores/data['qnorm'][selected,None]).mean()
    if signal=='std':return F.softplus(scores/data['std'][selected,None]).mean()
    if signal=='linear':return scores.mean()
    if signal=='absolute':return F.softplus(scores).mean()
    if signal=='mass':
        lse=torch.logsumexp(scores,dim=-1)+data['legal_newcount'][selected].double().log()-np.log(128)
        return F.softplus(lse-data['logZ'][selected]).mean()
    raise ValueError(signal)


def calibration(initial,prefix,new,tape,data,spec):
    pool,signal,mix,weight=spec;E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);norms=collections.defaultdict(list)
    for j in range(370,378):
        chosen=select(data,pool,rng)
        gl=torch.autograd.grad(prev.learning(E,R,new,tape,j,weight),E)[0]
        for s in ['absolute','mass'] if signal=='blend' else [signal]:
            gp=torch.autograd.grad(penalties(E,data,chosen,s),E)[0]
            norms[s].append([float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())])
    result={}
    for s,rows in norms.items():
        a=np.array(rows);result[s]={'lambda':.1*np.sqrt(np.mean(a[:,0]**2))/np.sqrt(np.mean(a[:,1]**2)),'norms':rows}
    return result


def train(name,spec,initial,prefix,new,tape,data,cal):
    start=time.monotonic();pool,signal,mix,weight=spec;E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);records=[]
    for j in range(370,740):
        chosen=select(data,pool,rng)
        loss=prev.learning(E,R,new,tape,j,weight);gl=torch.autograd.grad(loss,E)[0]
        combined=gl.clone();pvalues=[]
        for s,fraction in [('absolute',1-mix),('mass',mix)] if signal=='blend' else [(signal,1.)]:
            p=penalties(E,data,chosen,s);gp=torch.autograd.grad(p,E)[0]
            bn,pn=float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
            alpha=min(cal[s]['lambda'],.1*bn/pn) if pn else 0.
            combined+=fraction*alpha*gp;pvalues.append(float(p))
        opt.zero_grad(set_to_none=True);E.grad=combined;opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(E).all():raise FloatingPointError(name)
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalties':pvalues})
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
    data,prep=prepare(initial,old,new)
    cal={name:calibration(initial,prefix,new,tape,data,spec) for name,spec in ARMS.items()};ex.dump('calibration.json',cal)
    costs={'preparation':prep,'training':{}}
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,cal[name]);ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
