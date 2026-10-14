import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/teacher_boundary134_20261014')
import experiment134 as previous
worker,base,core,cx=previous.worker,previous.base,previous.core,previous.cx
ROOT=Path(__file__).resolve().parent
ARMS=['all_query','qualified_query','query_mix']

def status(stage,**kw):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def references(parent,initial,n,total,src,truths,steps,out):
    tick=time.monotonic()
    trainpath=src if parent==17 else Path('/data/lc/untzh/wn_transfer112_20261014')
    old=base.load(trainpath/'cache/train.npz')['old']
    keys,inverse,counts=np.unique(old[:,:2],axis=0,return_inverse=True,return_counts=True)
    teacher=base.load(src/'evaluation/teacher.npz')['old_train'] if parent==17 else base.load(src/'cache/fast-old_train.npz')['teacher']
    assert len(old)==len(teacher)
    qualifying=np.zeros(len(keys),bool)
    np.logical_or.at(qualifying,inverse,teacher[:,1]<=10)
    cohort=base.load('/data/lc/untzh/teacher_boundary134_20261014/'+str(parent)+'/cache/reference.npz')['keys']
    nr=len(initial['R'])
    encoded=keys[:,0]*nr+keys[:,1]
    index=np.searchsorted(encoded,cohort[:,0]*nr+cohort[:,1])
    assert np.array_equal(keys[index],cohort)
    E,R=initial['E'].cuda().double(),initial['R'].cuda().double()
    q_parts=[]
    for lo in range(0,len(keys),1024):
        b=torch.as_tensor(keys[lo:lo+1024],device='cuda')
        q=cx.tail_query(E[b[:,0]],R[b[:,1]]) if parent=='WN_ComplEx' else E[b[:,0]]*R[b[:,1]]
        q_parts.append(q)
    q=torch.cat(q_parts)
    forbidden=[]
    offsets=[0]
    for h,r in keys:
        values=[v-n for v in truths[int(h),int(r)] if v>=n]
        forbidden.extend(values)
        offsets.append(len(forbidden))
    forbidden=np.array(forbidden,dtype=np.int64)
    offsets=np.array(offsets,dtype=np.int64)
    selections={}
    all_ids=np.arange(len(keys))
    for arm in ARMS:
        rng=np.random.default_rng(1350017)
        rows=[]
        for k in range(steps):
            if arm=='query_mix':
                first=rng.choice(index,16,replace=False)
                remaining=np.setdiff1d(all_ids,first,assume_unique=True)
                chosen=np.concatenate([first,rng.choice(remaining,16,replace=False)])
            else:
                pool=all_ids if arm=='all_query' else np.flatnonzero(qualifying)
                chosen=rng.choice(pool,32,replace=False)
            assert len(np.unique(chosen))==32
            rows.append(chosen)
        selections[arm]=np.stack(rows)
    out.mkdir(parents=True,exist_ok=True)
    z=dict(keys=keys,q=q.cpu().numpy(),qualified=qualifying,cohort_pool=index,counts=counts,offsets=offsets,forbidden=forbidden,
           **{'selection_'+k:v for k,v in selections.items()})
    np.savez(out/'reference.npz',**z)
    report=dict(seconds=time.monotonic()-tick,old_train_facts=len(old),queries=len(keys),qualified_queries=int(qualifying.sum()),
                cohort_queries=len(index),query_frequency_mean=float(counts.mean()),bytes=sum(v.nbytes for v in z.values()),
                dense_legal_mask_bytes_avoided=len(keys)*(total-n),sparse_forbidden_count=len(forbidden),
                sampled_unique={k:len(np.unique(v)) for k,v in selections.items()},train_only=True)
    base.write(out/'reference.json',report)
    data=dict(q=q,keys=keys,offsets=offsets,forbidden=forbidden,selections=selections,
              nr=nr,qualified=qualifying,cohort_pool=index)
    return data,report

def legal_mask(data,chosen,m):
    legal=torch.ones((len(chosen),m),device='cuda',dtype=torch.bool)
    row_ids=[]
    cols=[]
    for i,index in enumerate(chosen):
        a,b=data['offsets'][index:index+2]
        if b>a:
            row_ids.extend([i]*(b-a))
            cols.extend(data['forbidden'][a:b])
    if cols:
        legal[row_ids,cols]=False
    return legal

def train(parent,arm,initial,prefix,ordinary,n,new,tape,data,space,out):
    tick=time.monotonic()
    E,R,opt=worker.model(initial,n,prefix)
    se,sr,so=worker.model(initial,n,prefix)
    learning=cx.learning if parent=='WN_ComplEx' else base.learning
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    records=[]
    for j in range(start,len(tape['epochs'])):
        chosen=data['selections'][arm][j-start]
        gl=torch.autograd.grad(learning(E,R,new,tape,j),E)[0]
        ss=data['q'][chosen]@E[n:].double().T
        legal=legal_mask(data,chosen,len(E)-n)
        p=(F.softplus(ss).masked_fill(~legal,0).sum(1)/legal.sum(1)).mean()
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
        projected=cx.source.space_module.project((E.detach()[n:]-ref).double(),space)
        with torch.no_grad():
            E[n:].copy_(ref+projected.to(E.dtype))
        actual=(E.detach()[n:]-ref).double()
        changes=(space['vectors']*actual[space['rows']]).sum(1)
        assert bool(torch.isfinite(E).all())
        records.append(dict(step=j,ratio=alpha*pn/bn if bn else 0.,cross_max=float(changes.abs().max()),norm=float(actual.norm())))
    assert torch.equal(E[:n].cpu(),initial['E'][:n]) and torch.equal(R.cpu(),initial['R'])
    assert torch.equal(se.detach().cpu().double(),ordinary['E']) and torch.equal(sr.cpu().double(),ordinary['R'])
    directory=out/'models'/arm
    directory.mkdir(parents=True,exist_ok=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt')
    base.write(directory/'steps.json',records)
    report=dict(seconds=time.monotonic()-tick,steps=len(records),mean_ratio=float(np.mean([r['ratio'] for r in records])),
                max_cross_score_difference=max(r['cross_max'] for r in records),shadow_compute_included=True,shadow_equal=True,
                final_norm=records[-1]['norm'],frozen_equal=True)
    base.write(directory/'training.json',report)
    return report

def coverage(keys,nr,queries,teacher):
    reference=keys[:,0]*nr+keys[:,1]
    encoded=queries[:,0]*nr+queries[:,1]
    mask=np.isin(encoded,reference)
    eligible=teacher[:,1]<=10
    return dict(answers=len(queries),covered_answers=int(mask.sum()),eligible=int(eligible.sum()),
                covered_eligible=int((eligible&mask).sum()),fraction=float(mask.mean()),
                eligible_fraction=float(mask[eligible].mean()) if eligible.any() else None)

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic()
    bundles=[]
    for parent in [17,'WN_ComplEx']:
        status('prepare',parent=str(parent))
        n,total,src,new,tape,truths,_,space=worker.training_assets(parent)
        ip,pp,op,_,_=previous.paths(parent)
        initial=torch.load(ip,map_location='cpu',weights_only=True)
        prefix=torch.load(pp,map_location='cpu',weights_only=True)
        ordinary=torch.load(op,map_location='cpu',weights_only=True)
        out=ROOT/str(parent)
        out.mkdir(parents=True,exist_ok=True)
        data,prep=references(parent,initial,n,total,src,truths,len(tape['epochs'])//2,out/'cache')
        training={}
        for arm in ARMS:
            status('train',parent=str(parent),arm=arm)
            training[arm]=train(parent,arm,initial,prefix,ordinary,n,new,tape,data,space,out)
            print('TRAIN135',parent,arm,training[arm],flush=True)
        bundles.append((parent,n,total,src,truths,out,data,prep,training))
    base.write(ROOT/'committed-models.json',dict(new_training_configs=6,all_endpoints_before_current_valid=True,train_only_reference=True))
    reports={}
    for parent,n,total,src,truths,out,data,prep,training in bundles:
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
        _,_,_,ordinary_rank,anchor_rank=previous.paths(parent)
        controls=dict(none=worker.metric(base.load(ordinary_rank),queries,teacher,split),
                      answer_anchor=worker.metric(base.load(anchor_rank),queries,teacher,split),
                      cohort_query_absolute=worker.metric(base.load(Path('/data/lc/untzh/teacher_boundary134_20261014')/str(parent)/'evaluation/absolute_uniform.npz'),queries,teacher,split))
        diagnostic=dict(full_pool=coverage(data['keys'],data['nr'],queries['old_valid'],teacher['old_valid']),
                        qualified_pool=coverage(data['keys'][data['qualified']],data['nr'],queries['old_valid'],teacher['old_valid']),
                        cohort_pool=coverage(data['keys'][data['cohort_pool']],data['nr'],queries['old_valid'],teacher['old_valid']),sampled={})
        evaluation=out/'evaluation'
        evaluation.mkdir(exist_ok=True)
        metrics={}
        ec={}
        for arm in ARMS:
            sampled=np.unique(data['selections'][arm])
            diagnostic['sampled'][arm]=coverage(data['keys'][sampled],data['nr'],queries['old_valid'],teacher['old_valid'])
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
            print('EVALUATED135',parent,arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_eval']['drops']['filtered_train_truths']['drops'],metric['directional_ratios_to_own_none'],flush=True)
        report=dict(metrics=metrics,reused_controls=controls,coverage=diagnostic,costs=dict(preparation=prep,training=training,evaluation=ec))
        reports[str(parent)]=report
        base.write(out/'summary.json',report)
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,new_training_configs=6,train_only_protection=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error=repr(exc))
        raise
