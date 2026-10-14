import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
sys.path.insert(0,'/data/lc/untzh/teacher_boundary134_20261014')
import experiment134 as previous
worker,base,core,cx=previous.worker,previous.base,previous.core,previous.cx
ROOT=Path(__file__).resolve().parent

def status(stage,**kw):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def query(E,R,a,complex_model):
    b=torch.as_tensor(a,device='cuda')
    return cx.tail_query(E[b[:,0]],R[b[:,1]]) if complex_model else E[b[:,0]]*R[b[:,1]]

def performance(y,score):
    order=np.argsort(-score,kind='stable')
    total=int(y.sum())
    result=dict(n=len(y),events=total,ap=float(average_precision_score(y,score)) if total else None,budgets={})
    for fraction in [.01,.05,.1]:
        k=int(np.ceil(len(y)*fraction))
        hits=int(y[order[:k]].sum())
        result['budgets'][str(fraction)]=dict(selected=k,hits=hits,precision=hits/k if k else None,recall=hits/total if total else None)
    return result

def degree(new,n,total):
    canonical=new[new[:,1]%2==0]
    d=np.zeros(total-n,np.int64)
    for h,r,t in canonical:
        if h>=n:d[h-n]+=1
        if t>=n and t!=h:d[t-n]+=1
    cross=np.zeros(total-n,np.int64)
    for h,r,t in canonical:
        if h<n<=t:cross[t-n]+=1
        elif t<n<=h:cross[h-n]+=1
    return d,cross

@torch.no_grad()
def features(initial,prefix,facts,pred,truths,n,total,complex_model,selected_ids,out,rows=False):
    tick=time.monotonic()
    positions=np.searchsorted(pred['ids'],selected_ids)
    assert np.array_equal(pred['ids'][positions],selected_ids)
    a=facts[selected_ids]
    gold=pred['gold'][positions]
    rank=pred['oldrank'][positions]
    E,R=initial['E'].cuda().double(),initial['R'].cuda().double()
    newrows=prefix['E'][n:].cuda().double()
    chosen=np.random.default_rng(1360017).choice(total-n,128,replace=False)
    scores=[]
    prefixrank=[]
    row_count=np.zeros(total-n,np.int64)
    row_max=np.full(total-n,-np.inf)
    for lo in range(0,len(a),128):
        b=a[lo:lo+128]
        q=query(E,R,b,complex_model)
        ss=q@newrows.T
        gs=torch.as_tensor(gold[lo:lo+len(b)],device='cuda')
        legal=torch.ones_like(ss,dtype=torch.bool)
        for j,(h,r,g) in enumerate(b):
            forbidden=[v-n for v in truths[int(h),int(r)] if v>=n]
            if forbidden:legal[j,forbidden]=False
        gap=ss-gs[:,None]
        gap=gap.masked_fill(~legal,-torch.inf)
        outrank=(gap>0)
        prefixrank.append(rank[lo:lo+len(b)]+outrank.sum(1).cpu().numpy())
        qnorm=q.norm(dim=1).clamp_min(1e-30)
        sample=gap[:,chosen]/qnorm[:,None]
        s32,s128=sample[:,:32],sample
        c32,c128=(s32>0).sum(1),(s128>0).sum(1)
        legal32=torch.isfinite(s32).sum(1).clamp_min(1)
        legal128=torch.isfinite(s128).sum(1).clamp_min(1)
        count=legal.sum(1)
        excess=count*c128.double()/legal128-(10-torch.as_tensor(rank[lo:lo+len(b)],device='cuda'))
        feature=torch.stack([qnorm,torch.as_tensor(rank[lo:lo+len(b)],device='cuda'),s32.max(1).values,s128.max(1).values,
            F.softplus(s32).masked_fill(~torch.isfinite(s32),0).sum(1)/legal32,
            F.softplus(s128).masked_fill(~torch.isfinite(s128),0).sum(1)/legal128,excess,c128.double()],1)
        scores.append(feature.cpu().numpy())
        if rows:
            row_count+=outrank.sum(0).cpu().numpy()
            row_max=np.maximum(row_max,(gap/qnorm[:,None]).max(0).values.cpu().numpy())
    X=np.concatenate(scores)
    pr=np.concatenate(prefixrank)
    payload=dict(ids=selected_ids,positions=positions,X=X,prefix_rank=pr,oldrank=rank,gold=gold,sampled_candidates=chosen)
    if rows:payload.update(row_count=row_count,row_max=row_max)
    np.savez(out,**payload)
    return payload,dict(seconds=time.monotonic()-tick,queries=len(a),candidates=total-n,sampled=128,full_prefix_rank_diagnostic=True)

@torch.no_grad()
def row_transitions(initial,prefix,endpoint,facts,pred,truths,n,complex_model,selected_ids):
    tick=time.monotonic()
    pos=np.searchsorted(pred['ids'],selected_ids)
    a=facts[selected_ids]
    E,R=initial['E'].cuda().double(),initial['R'].cuda().double()
    p=prefix['E'][n:].cuda().double()
    end=endpoint['E'][n:].cuda().double()
    newcross=np.zeros(len(p),np.int64)
    rescued=np.zeros(len(p),np.int64)
    for lo in range(0,len(a),128):
        b=a[lo:lo+128]
        q=query(E,R,b,complex_model)
        gold=torch.as_tensor(pred['gold'][pos[lo:lo+len(b)]],device='cuda')
        before=q@p.T>gold[:,None]
        after=q@end.T>gold[:,None]
        for i,(h,r,g) in enumerate(b):
            ix=[v-n for v in truths[int(h),int(r)] if v>=n]
            if ix:before[i,ix]=False;after[i,ix]=False
        newcross+=(after&~before).sum(0).cpu().numpy()
        rescued+=(before&~after).sum(0).cpu().numpy()
    return newcross,rescued,dict(seconds=time.monotonic()-tick)

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic()
    bundles=[]
    for parent in [17,'WN_ComplEx']:
        status('prepare',parent=str(parent))
        n,total,src,new,tape,truths,_,space=worker.training_assets(parent)
        ip,pp,op,rp,ap=previous.paths(parent)
        initial=torch.load(ip,map_location='cpu',weights_only=True)
        prefix=torch.load(pp,map_location='cpu',weights_only=True)
        queries=base.load(src/'evaluation/queries.npz')
        if parent==17:queries['old_valid']=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['old_valid']
        out=ROOT/str(parent)
        (out/'cache').mkdir(parents=True,exist_ok=True)
        c=base.load(src/'cache/cohort.npz')
        preds={s:base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{parent}-{s}-half-prediction.npz') for s in ['old_train','old_valid']}
        feat={}
        costs={}
        train_ids=np.sort(c['ids'])
        for split in preds:
            status('features',parent=str(parent),split=split)
            ids=train_ids if split=='old_train' else preds[split]['ids']
            feat[split],costs[split]=features(initial,prefix,queries[split],preds[split],truths,n,total,parent=='WN_ComplEx',ids,out/'cache'/f'{split}-features.npz',split=='old_train')
        deg,cross=degree(new,n,total)
        risk=feat['old_train']['row_count']>0
        support=np.asarray(space['ranks'])>0
        action=np.zeros(total-n,np.int8)
        action[risk&support&(deg<=3)]=1
        action[risk&support&(deg>3)]=2
        action[~support]=2
        np.savez(out/'cache/row-gate.npz',action=action,risk=risk,degree=deg,cross_degree=cross,query_support=support,
                 pressure_count=feat['old_train']['row_count'],pressure_max=feat['old_train']['row_max'])
        base.write(out/'frozen-gate.json',dict(stage='before_endpoint_read',actions={k:int((action==v).sum()) for k,v in [('none',0),('anchor',1),('replay',2)]},
            unsupported=int((~support).sum()),risk_rows=int(risk.sum()),single_rows=int((deg==1).sum()),few_rows=int(((deg>=1)&(deg<=3)).sum()),
            rule='TRAIN prefix any gold exceed: supported degree<=3 anchor, degree>3 replay; no exceed none; unsupported replay unknown'))
        bundles.append((parent,n,initial,prefix,op,rp,ap,queries,preds,feat,costs,truths,deg,cross,support,action,out))
    base.write(ROOT/'features-frozen.json',dict(no_training=True,future_endpoints_not_used_for_gate=True))
    reports={}
    for parent,n,initial,prefix,op,rp,ap,queries,preds,feat,costs,truths,deg,cross,support,action,out in bundles:
        status('label',parent=str(parent))
        ordinary=base.load(rp)
        anchor=base.load(ap)
        endpoint=torch.load(op,map_location='cpu',weights_only=True)
        cases={}
        for split,p in preds.items():
            f=feat[split]
            ids=f['ids']
            after=ordinary[split][ids,1]
            protected=anchor[split][ids,1]
            safe=f['prefix_rank']<=10
            y=after>10
            late=y&safe
            X=f['X']
            names=dict(boundary=X[:,1],max_gap32=X[:,2],max_gap128=X[:,3],softplus32=X[:,4],softplus128=X[:,5],excess128=X[:,6],
                       random=np.random.default_rng(1360017).random(len(ids)))
            prediction={k:performance(late[safe],v[safe]) for k,v in names.items()}
            order=np.argsort(-X[:,3],kind='stable')
            bands={}
            for name,ix in [('top5',order[:int(np.ceil(.05*len(ids)))]),('rest',order[int(np.ceil(.05*len(ids))):])]:
                bands[name]=dict(n=len(ix),ordinary_drops=int(y[ix].sum()),anchor_drops=int((protected[ix]>10).sum()),
                    rescued=int((y[ix]&(protected[ix]<=10)).sum()),new_harmed=int((~y[ix]&(protected[ix]>10)).sum()),
                    mean_rr_gain=float((1/protected[ix]-1/after[ix]).mean()))
            cases[split]=dict(n=len(ids),prefix_drops=int((~safe).sum()),new_late=int(late.sum()),rescued_by_training=int((~safe&~y).sum()),
                             ordinary_drops=int(y.sum()),anchor_drops=int((protected>10).sum()),prediction=prediction,anchor_association=bands,
                             costs=costs[split],note='association under shared trajectory, not proof protection necessary')
        rownew,rowres,rowcost=row_transitions(initial,prefix,endpoint,queries['old_train'],preds['old_train'],truths,n,parent=='WN_ComplEx',feat['old_train']['ids'])
        gate=base.load(out/'cache/row-gate.npz')
        row_labels=rownew>0
        strata={}
        for name,mask in [('zero',deg==0),('one',deg==1),('two_three',(deg>=2)&(deg<=3)),('four_plus',deg>=4)]:
            strata[name]=dict(rows=int(mask.sum()),unsupported=int((mask&~support).sum()),predicted_conflict=int((mask&(gate['risk'])).sum()),
                late_crossing_rows=int((mask&row_labels).sum()),new_crossing_pairs=int(rownew[mask].sum()),rescued_pairs=int(rowres[mask].sum()),
                action_counts={k:int((mask&(action==v)).sum()) for k,v in [('none',0),('anchor',1),('replay',2)]})
        rowdiag=dict(strata=strata,prediction=performance(row_labels,gate['pressure_count']),
            gate_precision=float(row_labels[gate['risk']].mean()) if gate['risk'].any() else None,
            missed_late_rows=int((row_labels&~gate['risk']).sum()),cost=rowcost,pair_crossing_is_not_query_drop=True)
        np.savez(out/'cache/row-labels.npz',late_pairs=rownew,rescued_pairs=rowres)
        drift={}
        for split,p in preds.items():
            ids=p['ids']
            for arm,state in [('none',ordinary),('anchor',anchor)]:
                before=p['oldrank'];after=state[split][ids,1]
                drift[split+'-'+arm]=dict(eligible=len(ids),worsened=int((after>before).sum()),
                    mean_rank_delta=float((after-before).mean()),mean_log_rank_ratio=float(np.log(after/before).mean()),
                    mean_rr_delta=float((1/after-1/before).mean()))
        report=dict(cases=cases,row_diagnostic=rowdiag,drift=drift,gate=__import__('json').loads((out/'frozen-gate.json').read_text()))
        reports[str(parent)]=report
        base.write(out/'summary.json',report)
        print('RISK136',parent,{s:(v['new_late'],v['prediction']['max_gap128']['ap']) for s,v in cases.items()},report['gate'],flush=True)
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,new_training_configs=0,
        decision='GO one fixed entity-action falsification, not method promotion',no_test=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
