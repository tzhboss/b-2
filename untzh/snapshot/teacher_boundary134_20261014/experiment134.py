import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/reference_train131_20261014')
import experiment131 as worker
base,core,cx=worker.base,worker.core,worker.cx
ROOT=Path(__file__).resolve().parent
ARMS=['absolute_uniform','boundary_softplus_uniform','boundary_hinge_uniform',
      'boundary_hinge_pressure','boundary_hinge_randompool','boundary_softplus_pressure']

def status(stage,**kw):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def paths(parent):
    out=Path('/data/lc/untzh/train_init121_20261014/seed17') if parent==17 else Path('/data/lc/untzh/wn_init122_20261014/ComplEx')
    protected=Path('/data/lc/untzh/smallinit_protect126_20261014/seed17/zero/evaluation/anchor_null.npz') if parent==17 else Path('/data/lc/untzh/wn_smallprotect127_20261014/ComplEx/zero/evaluation/answer_anchor.npz')
    return out/'models/zero/initial.pt',out/'models/zero/prefix10.pt',out/'models/zero/epoch-20.pt',out/'evaluation/zero.npz',protected

@torch.no_grad()
def pressure(rows,data):
    tick=time.monotonic()
    values=[]
    for lo in range(0,len(data['q']),256):
        ss=data['q'][lo:lo+256]@rows.double().T
        gap=(ss-data['boundary'][lo:lo+256,None])/data['qnorm'][lo:lo+256,None]
        legal=data['legal'][lo:lo+256]
        loss=F.relu(gap).square().masked_fill(~legal,0).sum(1)/legal.sum(1)
        values.append(loss.cpu().numpy())
    return np.concatenate(values),time.monotonic()-tick

@torch.no_grad()
def teacher_boundary(initial,keys,q,n,truths):
    tick=time.monotonic()
    old=initial['E'][:n].cuda().double()
    scores=[]
    candidates=[]
    for lo in range(0,len(keys),128):
        b=keys[lo:lo+128]
        ss=q[lo:lo+len(b)]@old.T
        for i,(h,r) in enumerate(b):
            ids=[v for v in truths[int(h),int(r)] if v<n]
            if ids:
                ss[i,ids]=-torch.inf
        top=ss.topk(10,dim=1)
        assert bool(torch.isfinite(top.values).all())
        scores.append(top.values.cpu().numpy())
        candidates.append(top.indices.cpu().numpy())
    return np.concatenate(scores),np.concatenate(candidates),time.monotonic()-tick

def references(parent,initial,prefix,n,total,src,truths,steps,out):
    tick=time.monotonic()
    c=base.load(src/'cache/cohort.npz')
    keys,indices=np.unique(c['queries'][:,:2],axis=0,return_index=True)
    q=torch.as_tensor(c['q'][indices],device='cuda')
    qnorm=q.norm(dim=1).clamp_min(1e-30)
    top_scores,top_ids,teacher_seconds=teacher_boundary(initial,keys,q,n,truths)
    legal=np.ones((len(keys),total-n),bool)
    for i,(h,r) in enumerate(keys):
        legal[i,[v-n for v in truths[int(h),int(r)] if v>=n]]=False
    rng=np.random.default_rng(1340017)
    selections=np.stack([rng.choice(len(keys),32,replace=False) for _ in range(steps)])
    data=dict(q=q,qnorm=qnorm,boundary=torch.as_tensor(top_scores[:,-1],device='cuda'),
              legal=torch.as_tensor(legal,device='cuda'),selections=selections)
    first_risk,risk_seconds=pressure(prefix['E'][n:].cuda(),data)
    data['first_risk']=first_risk
    out.mkdir(parents=True,exist_ok=True)
    z=dict(keys=keys,cohort_positions=indices,q=q.cpu().numpy(),qnorm=qnorm.cpu().numpy(),
           teacher_top10_scores=top_scores,teacher_top10_ids=top_ids,boundary=top_scores[:,-1],legal=legal,
           selections=selections,first_risk=first_risk)
    np.savez(out/'reference.npz',**z)
    report=dict(seconds=time.monotonic()-tick,teacher_boundary_seconds_included=teacher_seconds,first_risk_seconds_included=risk_seconds,
                query_count=len(keys),answer_count=len(c['queries']),negative_boundaries=int((top_scores[:,-1]<0).sum()),
                mean_boundary=float(top_scores[:,-1].mean()),mean_normalized_boundary=float((top_scores[:,-1]/z['qnorm']).mean()),
                bytes=sum(v.nbytes for v in z.values()),train_queries_only=True,all_train_truths_excluded_from_old_prediction_pool=True)
    base.write(out/'reference.json',report)
    return data,report

def train(parent,arm,initial,prefix,ordinary,n,new,tape,data,space,out):
    tick=time.monotonic()
    E,R,opt=worker.model(initial,n,prefix)
    se,sr,so=worker.model(initial,n,prefix)
    learning=cx.learning if parent=='WN_ComplEx' else base.learning
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    rng=np.random.default_rng(1340017)
    rows=[]
    decisions=[]
    detection_seconds=0.
    pool=None
    for j in range(start,len(tape['epochs'])):
        k=j-start
        epoch=int(tape['epochs'][j])
        if arm.endswith('uniform'):
            chosen=data['selections'][k]
        else:
            if j==start or tape['epochs'][j]!=tape['epochs'][j-1]:
                count=int(np.ceil(.05*len(data['q'])))
                if arm.endswith('pressure'):
                    if j==start:
                        risk=data['first_risk']
                        seconds=0.
                    else:
                        risk,seconds=pressure(E.detach()[n:],data)
                    pool=np.argsort(-risk,kind='stable')[:count]
                else:
                    pool=np.random.default_rng(1341700+epoch).choice(len(data['q']),count,replace=False)
                    seconds=0.
                detection_seconds+=seconds
                decisions.append(dict(epoch=epoch,pool=pool.tolist(),seconds=seconds,first_risk_reused=(j==start and arm.endswith('pressure'))))
            first=rng.choice(pool,26,replace=False)
            allowed=np.ones(len(data['q']),bool)
            allowed[first]=False
            chosen=np.concatenate([first,rng.choice(np.flatnonzero(allowed),6,replace=False)])
        assert len(np.unique(chosen))==32
        gl=torch.autograd.grad(learning(E,R,new,tape,j),E)[0]
        ss=data['q'][chosen]@E[n:].double().T
        legal=data['legal'][chosen]
        if arm.startswith('absolute'):
            values=F.softplus(ss)
        else:
            gap=(ss-data['boundary'][chosen,None])/data['qnorm'][chosen,None]
            values=F.relu(gap).square() if 'hinge' in arm else F.softplus(gap)
        p=(values.masked_fill(~legal,0).sum(1)/legal.sum(1)).mean()
        gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[n:].double().norm()),float(gp[n:].double().norm())
        alpha=.1*bn/pn if pn else 0.
        opt.zero_grad(set_to_none=True)
        E.grad=gl+alpha*gp
        opt.step()
        so.zero_grad(set_to_none=True)
        learning(se,sr,new,tape,j).backward()
        so.step()
        ref=se.detach()[n:]
        delta=(E.detach()[n:]-ref).double()
        accepted=cx.source.space_module.project(delta,space)
        with torch.no_grad():
            E[n:].copy_(ref+accepted.to(E.dtype))
        actual=(E.detach()[n:]-ref).double()
        changes=(space['vectors']*actual[space['rows']]).sum(1)
        assert bool(torch.isfinite(E).all())
        rows.append(dict(step=j,ratio=alpha*pn/bn if bn else 0.,protection_norm=pn,
                         selected=chosen.tolist(),cross_max=float(changes.abs().max()),norm=float(actual.norm())))
    assert torch.equal(E[:n].cpu(),initial['E'][:n]) and torch.equal(R.cpu(),initial['R'])
    assert torch.equal(se.detach().cpu().double(),ordinary['E']) and torch.equal(sr.cpu().double(),ordinary['R'])
    directory=out/'models'/arm
    directory.mkdir(parents=True,exist_ok=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt')
    base.write(directory/'steps.json',rows)
    base.write(directory/'decisions.json',decisions)
    report=dict(seconds=time.monotonic()-tick,steps=len(rows),detection_seconds_included=detection_seconds,
                mean_ratio=float(np.mean([r['ratio'] for r in rows])),nonzero_steps=sum(r['protection_norm']>0 for r in rows),
                max_cross_score_difference=max(r['cross_max'] for r in rows),shadow_compute_included=True,shadow_equal=True,
                final_norm=rows[-1]['norm'],frozen_equal=True)
    base.write(directory/'training.json',report)
    return report

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic()
    bundles=[]
    for parent in [17,'WN_ComplEx']:
        status('prepare',parent=str(parent))
        n,total,src,new,tape,truths,_,space=worker.training_assets(parent)
        ip,pp,op,_,_=paths(parent)
        initial=torch.load(ip,map_location='cpu',weights_only=True)
        prefix=torch.load(pp,map_location='cpu',weights_only=True)
        ordinary=torch.load(op,map_location='cpu',weights_only=True)
        out=ROOT/str(parent)
        out.mkdir(parents=True,exist_ok=True)
        data,prep=references(parent,initial,prefix,n,total,src,truths,len(tape['epochs'])//2,out/'cache')
        training={}
        for arm in ARMS:
            status('train',parent=str(parent),arm=arm)
            training[arm]=train(parent,arm,initial,prefix,ordinary,n,new,tape,data,space,out)
            print('TRAIN134',parent,arm,training[arm],flush=True)
        bundles.append((parent,n,total,src,truths,out,prep,training))
    base.write(ROOT/'committed-models.json',dict(new_training_configs=12,all_endpoints_before_current_valid=True,train_only_query_selection=True))
    reports={}
    for parent,n,total,src,truths,out,prep,training in bundles:
        base.N=core.N=n
        base.TOTAL=core.TOTAL=total
        split=base.load(f'/data/lc/untzh/admission_action128_20261014/cache/{parent}-split.npz')
        if parent==17:
            worker.fastmodule.ROOT=Path('/data/lc/untzh/fast_old_eval107_20261014')
            fast=worker.fastmodule.FastOld(17,truths)
            queries=dict(fast.queries,new_valid=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['new_valid'])
            teacher=fast.teacher
            oldrank=lambda E,R,s:fast.ranks(E,R,s)
            newrank=lambda E,R:base.ranks(E,R,queries['new_valid'],truths)
        else:
            queries=base.load(src/'evaluation/queries.npz')
            cache={s:base.load(src/'cache'/f'fast-{s}.npz') for s in ['old_train','old_valid']}
            teacher={s:cache[s]['teacher'] for s in cache}
            oldrank=lambda E,R,s:cx.fast_ranks(E,R,queries[s],cache[s],truths)
            newrank=lambda E,R:cx.ranks(E,R,queries['new_valid'],truths)
        _,_,_,ordinary_rank,anchor_rank=paths(parent)
        controls=dict(none=worker.metric(base.load(ordinary_rank),queries,teacher,split),
                      answer_anchor=worker.metric(base.load(anchor_rank),queries,teacher,split))
        evaluation=out/'evaluation'
        evaluation.mkdir(exist_ok=True)
        metrics={}
        ec={}
        for arm in ARMS:
            status('evaluate',parent=str(parent),arm=arm)
            m=torch.load(out/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
            E,R=m['E'].cuda(),m['R'].cuda()
            ranks={}
            costs={}
            for s in ['old_train','old_valid']:
                ranks[s],costs[s]=oldrank(E,R,s)
            st=time.monotonic()
            ranks['new_valid']=newrank(E,R)
            costs['new_valid']=dict(seconds=time.monotonic()-st)
            np.savez(evaluation/f'{arm}.npz',**ranks)
            metric=worker.metric(ranks,queries,teacher,split)
            metric['directional_ratios_to_own_none']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/controls['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
            metrics[arm]=metric
            ec[arm]=costs
            print('EVALUATED134',parent,arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_eval']['drops']['filtered_train_truths']['drops'],metric['directional_ratios_to_own_none'],flush=True)
        report=dict(metrics=metrics,reused_controls=controls,costs=dict(preparation=prep,training=training,evaluation=ec))
        reports[str(parent)]=report
        base.write(out/'summary.json',report)
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,new_training_configs=12,train_only_protection=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error=repr(exc))
        raise
