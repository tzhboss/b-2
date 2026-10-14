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
ARMS=['reference_softplus_anchor','reference_softplus_normmatched']

def status(stage,**kw):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def paths(parent,init):
    if init=='constrained':
        out=Path('/data/lc/untzh/reference_train131_20261014')/str(parent)/'historical_reference/constrained'
        initial=Path('/data/lc/untzh/reference_init130_20261014')/str(parent)/'historical_reference/constrained-initial.pt'
        return initial,out/'prefix10.pt',out/'models/none/epoch-20.pt',out/'evaluation/none.npz',out/'evaluation/answer_anchor.npz'
    out=Path('/data/lc/untzh/train_init121_20261014/seed17') if parent==17 else Path('/data/lc/untzh/wn_init122_20261014/ComplEx')
    protected=Path('/data/lc/untzh/smallinit_protect126_20261014/seed17/zero/evaluation/anchor_null.npz') if parent==17 else Path('/data/lc/untzh/wn_smallprotect127_20261014/ComplEx/zero/evaluation/answer_anchor.npz')
    return out/'models/zero/initial.pt',out/'models/zero/prefix10.pt',out/'models/zero/epoch-20.pt',out/'evaluation/zero.npz',protected

def references(parent,initial,n,total,truths,steps,out):
    tick=time.monotonic()
    src=Path('/data/lc/untzh/increment87_20261014') if parent==17 else Path('/data/lc/untzh/complex_transfer118_20261014')
    queries=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['old_valid'] if parent==17 else base.load(src/'evaluation/queries.npz')['old_valid']
    p=base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{parent}-old_valid-half-prediction.npz')
    split=base.load(f'/data/lc/untzh/admission_action128_20261014/cache/{parent}-split.npz')
    mask=split['calibration'][p['ids']]
    ids=p['ids'][mask]
    selected=queries[ids]
    q=worker.initialization.query(initial,selected,parent=='WN_ComplEx')
    legal=np.ones((len(ids),total-n),bool)
    for i,(h,r,g) in enumerate(selected):
        legal[i,[v-n for v in truths[int(h),int(r)] if v>=n]]=False
    rng=np.random.default_rng(1320017)
    selections=np.stack([rng.choice(len(ids),32,replace=False) for _ in range(steps)])
    z=dict(ids=ids,queries=selected,q=q.cpu().numpy(),gold=p['gold'][mask],oldrank=p['oldrank'][mask],legal=legal,selections=selections)
    out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'reference.npz',**z)
    report=dict(seconds=time.monotonic()-tick,reference_queries=len(ids),steps=steps,old_valid_calibration_supervision=True,bytes=sum(v.nbytes for v in z.values()))
    base.write(out/'reference.json',report)
    data={k:torch.as_tensor(z[k],device='cuda') for k in ['q','gold','oldrank','legal']}
    data['qnorm']=data['q'].norm(dim=1).clamp_min(1e-30)
    data['selections']=selections
    return data,report

def train(parent,arm,initial,prefix,n,new,tape,data,out):
    tick=time.monotonic()
    E,R,opt=worker.model(initial,n,prefix)
    learning=cx.learning if parent=='WN_ComplEx' else base.learning
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    rows=[]
    for j in range(start,len(tape['epochs'])):
        chosen=data['selections'][j-start]
        gl=torch.autograd.grad(learning(E,R,new,tape,j),E)[0]
        ss=data['q'][chosen]@E[n:].double().T
        legal=data['legal'][chosen]
        if arm=='reference_softplus':
            values=F.softplus(ss).masked_fill(~legal,0)
            p=(values.sum(1)/legal.sum(1)).mean()
            active=int(legal.sum())
        else:
            gap=(ss-data['gold'][chosen,None])/data['qnorm'][chosen,None]
            active_mask=(gap.detach()>0)&legal
            capacity=(10-data['oldrank'][chosen]).long()
            top=gap.detach().masked_fill(~legal,-torch.inf).topk(9,dim=1).indices
            allowance=torch.arange(9,device='cuda')[None,:]<capacity[:,None]
            keep=torch.zeros_like(active_mask)
            keep.scatter_(1,top,allowance)
            active_mask=active_mask&~keep
            p=(F.relu(gap).square().masked_fill(~active_mask,0).sum(1)/legal.sum(1)).mean()
            active=int(active_mask.sum())
        gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[n:].double().norm()),float(gp[n:].double().norm())
        alpha=.1*bn/pn if pn else 0.
        opt.zero_grad(set_to_none=True)
        E.grad=gl+alpha*gp
        opt.step()
        assert bool(torch.isfinite(E).all())
        rows.append(dict(step=j,ratio=alpha*pn/bn if bn else 0.,active_pairs=active,protection_norm=pn))
    assert torch.equal(E[:n].cpu(),initial['E'][:n]) and torch.equal(R.cpu(),initial['R'])
    directory=out/'models'/arm
    directory.mkdir(parents=True,exist_ok=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt')
    base.write(directory/'steps.json',rows)
    report=dict(seconds=time.monotonic()-tick,steps=len(rows),nonzero_steps=sum(r['protection_norm']>0 for r in rows),
                mean_ratio=float(np.mean([r['ratio'] for r in rows])),shadow_compute_included=False,frozen_equal=True)
    base.write(directory/'training.json',report)
    return report

from protection133 import train, references

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic()
    bundles=[]
    for parent in [17,'WN_ComplEx']:
        n,total,src,new,tape,truths,_,_=worker.training_assets(parent)
        first=torch.load(paths(parent,'zero')[0],map_location='cpu',weights_only=True)
        data,prep=references(parent,first,n,total,truths,len(tape['epochs'])//2,ROOT/str(parent)/'cache')
        for init in ['zero','constrained']:
            ip,pp,_,_,_=paths(parent,init)
            initial=torch.load(ip,map_location='cpu',weights_only=True)
            prefix=torch.load(pp,map_location='cpu',weights_only=True)
            out=ROOT/str(parent)/init
            out.mkdir(parents=True,exist_ok=True)
            costs={}
            for arm in ARMS:
                status('train',parent=str(parent),init=init,arm=arm)
                costs[arm]=train(parent,arm,initial,prefix,n,new,tape,data,out)
                print('TRAIN133',parent,init,arm,costs[arm],flush=True)
            bundles.append((parent,n,total,src,truths,init,out,prep,costs))
    base.write(ROOT/'committed-models.json',dict(new_training_configs=8,all_endpoints_before_current_valid=True))
    reports={}
    for parent in [17,'WN_ComplEx']:
        n,total,src,_,_,truths,_,_=worker.training_assets(parent)
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
        parent_report={}
        for b in [b for b in bundles if b[0]==parent]:
            _,_,_,_,_,init,out,prep,training=b
            _,_,_,ordinary_rank_path,anchor_rank_path=paths(parent,init)
            controls=dict(none=worker.metric(base.load(ordinary_rank_path),queries,teacher,split),answer_anchor=worker.metric(base.load(anchor_rank_path),queries,teacher,split))
            evaluation=out/'evaluation'
            evaluation.mkdir(exist_ok=True)
            metrics={}
            ec={}
            for arm in ARMS:
                status('evaluate',parent=str(parent),init=init,arm=arm)
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
                print('EVALUATED133',parent,init,arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_eval']['drops']['filtered_train_truths']['drops'],metric['directional_ratios_to_own_none'],flush=True)
            parent_report[init]=dict(metrics=metrics,reused_controls=controls,costs=dict(training=training,preparation=prep,evaluation=ec))
            base.write(out/'summary.json',parent_report[init])
        reports[str(parent)]=parent_report
        base.write(ROOT/str(parent)/'summary.json',parent_report)
        base.write(ROOT/'summary-running.json',dict(parents=reports))
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,new_training_configs=8,historical_reference_supervision=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error=repr(exc))
        raise
