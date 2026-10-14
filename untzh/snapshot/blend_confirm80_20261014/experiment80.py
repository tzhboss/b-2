"""Fixed blend confirmation on pre-existing seeds29/43; no tuning or TEST."""
import collections
import copy
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, '/data/lc/untzh/risk58_20261013')
import experiment58 as ex
base = ex.base
ROOT = Path(__file__).resolve().parent
SRC = ex.SRC
N, TOTAL, SIZE = ex.N, ex.TOTAL, ex.SIZE
ARMS = ['none', 'absolute', 'mass', 'blend', 'absolute_normmatched', 'absolute075']
DIRECTIONS = ['head_old_gold_new', 'head_new_gold_old', 'both_new']

def dump(name, value):
    base.write(ROOT / name, value)

def progress(stage, **values):
    dump('status.json', dict(stage=stage, pid=os.getpid(), **values))
    print(stage.upper(), values, flush=True)

def truthmap(old, new):
    tr = collections.defaultdict(set)
    for h, r, t in np.concatenate([old, new]):
        tr[int(h), int(r)].add(int(t))
    return tr

def tape(seed, new, tr, nr):
    start = time.monotonic()
    gen = torch.Generator().manual_seed(seed + 71)
    known = np.sort(np.array([(h * nr + r) * TOTAL + t for (h,r), ts in tr.items() for t in ts], dtype=np.int64))
    pieces = collections.defaultdict(list)
    bounds, epochs, rejected = [0], [], 0
    for ep in range(1, 21):
        for ids in torch.randperm(len(new), generator=gen).split(1024):
            p = torch.tensor(new[ids.numpy()]); h,r,t = p.unbind(1)
            draw = torch.randint(TOTAL, (2,len(p),10), generator=gen)
            tail = torch.rand((len(p),10), generator=gen) > .5
            nh = torch.where(tail,h[:,None],draw[0]).numpy().copy()
            nt = torch.where(tail,draw[1],t[:,None]).numpy().copy()
            rr = r.numpy()[:,None]
            while True:
                keys = (nh * nr + rr) * TOTAL + nt
                idx = np.searchsorted(known, keys)
                bad = (idx < len(known)) & (known[np.minimum(idx,len(known)-1)] == keys)
                if not bad.any(): break
                # Keep rejection order deterministic within this new tape.
                for i,j in np.argwhere(bad):
                    value = int(torch.randint(TOTAL,(1,),generator=gen))
                    if tail[i,j]: nt[i,j]=value
                    else: nh[i,j]=value
                rejected += int(bad.sum())
            pieces['pos'].append(ids.numpy().astype(np.int32))
            pieces['nh'].append(nh.astype(np.int32));pieces['nt'].append(nt.astype(np.int32))
            bounds.append(bounds[-1]+len(p));epochs.append(ep)
        progress('prepare_tape', seed=seed, epoch=ep)
    result = {k:np.concatenate(v) for k,v in pieces.items()}
    result.update(bounds=np.array(bounds),epochs=np.array(epochs))
    np.savez(ROOT/'cache'/'learning.npz',**result)
    return result, dict(seconds=time.monotonic()-start,rejections=rejected,seed=seed+71)

def newmodel(initial,prefix=None):
    E,R,opt = ex.init_model(initial)
    if prefix:
        E.data.copy_(prefix['E'].cuda());opt.load_state_dict(copy.deepcopy(prefix['optimizer']))
    return E,R,opt

def prefix10(initial,new,t):
    tick=time.monotonic();E,R,opt=newmodel(initial)
    for j in range(370):
        loss=base.learning(E,R,new,t,j)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if j==369 or t['epochs'][j+1]!=t['epochs'][j]:progress('prefix',epoch=int(t['epochs'][j]))
    assert torch.isfinite(E).all() and torch.equal(E[:N].cpu(),initial['E'][:N])
    state=dict(E=E.detach().cpu(),R=R.cpu(),optimizer=opt.state_dict())
    torch.save(state,ROOT/'cache'/'prefix10.pt')
    return state,time.monotonic()-tick

@torch.no_grad()
def teacher(initial,old,tr):
    # This is the preparation needed to select TRAIN-qualified answers.
    tick=time.monotonic()
    a=base.ranks(initial['E'][:N].cuda().double(),initial['R'].cuda().double(),old,tr,arm='teacher',split='old_train')
    np.savez(ROOT/'cache'/'teacher-train.npz',old_train=a)
    return a,time.monotonic()-tick

@torch.no_grad()
def cohort(initial,old,tr,A,seed):
    tick=time.monotonic();rng=np.random.default_rng(seed)
    eligible=np.flatnonzero(A[:,1]<=10)
    assert len(eligible)>=SIZE
    ids=rng.choice(eligible,SIZE,replace=False);queries=old[ids]
    candidates=rng.integers(N,TOTAL,size=(SIZE,128),dtype=np.int64)
    for i,(h,r,t) in enumerate(queries):
        known=list(tr[int(h),int(r)])
        bad=np.isin(candidates[i],known)
        while bad.any():
            candidates[i,bad]=rng.integers(N,TOTAL,size=int(bad.sum()))
            bad=np.isin(candidates[i],known)
    e=initial['E'][:N].cuda().double();r=initial['R'].cuda().double()
    q=e[queries[:,0]]*r[queries[:,1]];logs=[]
    for lo in range(0,SIZE,128):
        s=q[lo:lo+128]@e.T;rows=[];cols=[]
        for i,(h,rr,t) in enumerate(queries[lo:lo+128]):
            for c in tr[int(h),int(rr)]:
                if c<N and c!=int(t):rows.append(i);cols.append(c)
        if rows:s[rows,cols]=-torch.inf
        logs.append(torch.logsumexp(s,dim=-1))
    logZ=torch.cat(logs)
    counts=torch.as_tensor([TOTAL-N-sum(c>=N for c in tr[int(h),int(r)]) for h,r,t in queries],device='cuda')
    assert (counts>0).all() and torch.isfinite(logZ).all()
    np.savez(ROOT/'cache'/'cohort.npz',ids=ids,queries=queries,candidates=candidates,q=q.cpu().numpy(),logZ=logZ.cpu().numpy(),legal_newcount=counts.cpu().numpy())
    # All arms share a fixed 370x32 sequence, independent of outcomes.
    selections=np.stack([np.random.default_rng(seed+1000+j).choice(SIZE,32,replace=False) for j in range(370)])
    np.savez(ROOT/'cache'/'protection.npz',selections=selections)
    return dict(q=q,candidates=torch.as_tensor(candidates,device='cuda'),logZ=logZ,legal_newcount=counts,selections=selections),dict(seconds=time.monotonic()-tick,eligible=len(eligible),sampled=SIZE)

def penalty(E,d,chosen,kind):
    s=(d['q'][chosen,None]*E[d['candidates'][chosen]].double()).sum(-1)
    if kind=='absolute':return F.softplus(s).mean()
    lse=torch.logsumexp(s,dim=-1)+d['legal_newcount'][chosen].double().log()-np.log(128)
    return F.softplus(lse-d['logZ'][chosen]).mean()

def calibrate(initial,prefix,new,t,d):
    tick=time.monotonic();E,R,opt=newmodel(initial,prefix);rows=collections.defaultdict(list)
    for k,j in enumerate(range(370,378)):
        gl=torch.autograd.grad(base.learning(E,R,new,t,j),E)[0]
        for kind in ['absolute','mass']:
            gp=torch.autograd.grad(penalty(E,d,d['selections'][k],kind),E)[0]
            rows[kind].append([float(gl[N:].double().norm()),float(gp[N:].double().norm())])
    c={}
    for kind,v in rows.items():
        a=np.array(v);den=np.sqrt(np.mean(a[:,1]**2));assert den>0 and np.isfinite(a).all()
        c[kind]=.1*np.sqrt(np.mean(a[:,0]**2))/den
    dump('calibration.json',dict(coefficients=c,norms=dict(rows),seconds=time.monotonic()-tick))
    return c,time.monotonic()-tick

def train(arm,initial,prefix,new,t,d,cal):
    tick=time.monotonic();E,R,opt=newmodel(initial,prefix);records=[]
    for j in range(370,740):
        loss=base.learning(E,R,new,t,j);gl=torch.autograd.grad(loss,E)[0]
        bn=float(gl[N:].double().norm());components={};chosen=d['selections'][j-370]
        needed=[] if arm=='none' else ['absolute','mass'] if arm in ['blend','absolute_normmatched'] else ['absolute' if arm=='absolute075' else arm]
        rho=.075 if arm=='absolute075' else .1
        for kind in needed:
            p=penalty(E,d,chosen,kind);gp=torch.autograd.grad(p,E)[0];pn=float(gp[N:].double().norm())
            alpha=min(cal[kind]*rho/.1,rho*bn/pn) if pn else 0.
            components[kind]=alpha*gp
        if arm=='none':g=torch.zeros_like(gl)
        elif arm=='blend':g=.5*(components['absolute']+components['mass'])
        elif arm=='absolute_normmatched':
            mixed=.5*(components['absolute']+components['mass']);g=components['absolute']
            gn=float(g[N:].double().norm());g=g*(float(mixed[N:].double().norm())/gn) if gn else g
        else:g=components['absolute' if arm=='absolute075' else arm]
        ratio=float(g[N:].double().norm())/bn if bn else 0.
        assert ratio<=rho*(1+1e-5)
        opt.zero_grad(set_to_none=True);E.grad=gl+g;opt.step()
        assert torch.isfinite(loss) and torch.isfinite(E).all()
        records.append(dict(step=j,epoch=int(t['epochs'][j]),loss=float(loss),gradient_ratio=ratio))
        if j==739 or t['epochs'][j+1]!=t['epochs'][j]:progress('train',arm=arm,epoch=int(t['epochs'][j]))
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/arm;out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    base.write(out/'steps.json',records)
    result=dict(seconds=time.monotonic()-tick,max_gradient_ratio=max(x['gradient_ratio'] for x in records),frozen_equal=True)
    base.write(out/'training.json',result);return result

def drops(a,A):
    return {name:dict(eligible=int((A[:,j]<=10).sum()),drops=int(((A[:,j]<=10)&(a[:,j]>10)).sum())) for j,name in enumerate(['raw','filtered_train_truths'])}

def evaluate(initial,old,new,tr,A):
    tick=time.monotonic();valid=base.load(SRC/'entity_refresh43_20261012/artifacts/data/validation.npz')
    av=base.ranks(initial['E'][:N].cuda().double(),initial['R'].cuda().double(),valid['old'],tr,arm='teacher',split='old_valid')
    out=ROOT/'evaluation';out.mkdir();np.savez(out/'teacher.npz',old_train=A,old_valid=av)
    report={}
    for arm in ARMS:
        m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
        E,R=m['E'].cuda(),m['R'].cuda()
        qs=dict(old_train=old,old_valid=valid['old'],new_valid=valid['new'])
        states={key:base.ranks(E,R,q,tr,arm=arm,split=key) for key,q in qs.items()}
        np.savez(out/f'{arm}.npz',**states)
        report[arm]={key:base.metrics(states[key],q) for key,q in qs.items()}
        for key,teacher_a in [('old_train',A),('old_valid',av)]:report[arm][key]['drops']=drops(states[key],teacher_a)
        progress('evaluated',arm=arm,new_mrr=report[arm]['new_valid']['all']['filtered_train_truths']['mrr'],train_drops=report[arm]['old_train']['drops']['filtered_train_truths']['drops'])
    for arm in ARMS:
        report[arm]['directional_ratios']=[report[arm]['new_valid'][d]['filtered_train_truths']['mrr']/report['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in DIRECTIONS]
    dump('evaluation/metrics.json',report)
    return report,time.monotonic()-tick

def run(seed):
    global ROOT
    parent=ROOT;ROOT=parent/f'seed{seed}';ROOT.mkdir(exist_ok=True);(ROOT/'cache').mkdir(exist_ok=True);base.ROOT=ROOT
    tick=time.monotonic();torch.manual_seed(seed)
    initial=torch.load(SRC/f'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s{seed}/initial.pt',map_location='cpu',weights_only=True)
    d=base.load(SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new'];tr=truthmap(old,new)
    if (ROOT/'cache'/'learning.npz').exists():
        reuse_tick=time.monotonic();t=base.load(ROOT/'cache'/'learning.npz')
        assert len(t['epochs'])==740 and len(t['pos'])==len(new)*20
        tape_cost=dict(reuse_seconds=time.monotonic()-reuse_tick,prior_seconds=None,note='reused preparation retained from CuBLAS configuration failure before training')
    else:t,tape_cost=tape(seed,new,tr,len(initial['R']))
    A,teacher_seconds=teacher(initial,old,tr)
    data,cohort_cost=cohort(initial,old,tr,A,seed)
    prefix,prefix_seconds=prefix10(initial,new,t)
    c,cal_seconds=calibrate(initial,prefix,new,t,data)
    costs=dict(tape=tape_cost,teacher_seconds=teacher_seconds,cohort=cohort_cost,prefix_seconds=prefix_seconds,calibration_seconds=cal_seconds,training={})
    for arm in ARMS:
        costs['training'][arm]=train(arm,initial,prefix,new,t,data,c);dump('costs-running.json',costs)
    metrics,es=evaluate(initial,old,new,tr,A);costs['evaluation_seconds']=es;costs['total_seconds']=time.monotonic()-tick
    dump('costs.json',costs);progress('complete',seed=seed,seconds=costs['total_seconds']);ROOT=parent
    return metrics,costs

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base.ROOT=ROOT;tick=time.monotonic();progress('starting')
    dump('environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),seeds=[29,43]))
    reports={};costs={}
    for seed in [29,43]:reports[str(seed)],costs[str(seed)]=run(seed);dump('summary-running.json',dict(metrics=reports,costs=costs))
    dump('summary.json',dict(metrics=reports,costs=costs,total_seconds=time.monotonic()-tick))
    progress('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:progress('failed',error=repr(exc));raise
