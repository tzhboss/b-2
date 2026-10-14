import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import collections
import copy
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
sys.path.insert(0,'/data/lc/untzh/increment87_20261014')
import experiment87 as prev
core=prev.core;ex=prev.ex;base=prev.base;ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;core.ROOT=ROOT;base.ROOT=ROOT;N,TOTAL=prev.N,prev.TOTAL
SEED=29

def prepare():
    tick=time.monotonic();(ROOT/'cache').mkdir()
    z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz');old,new,extra=z['old'],z['new'],z['extra']
    parentpath=Path('/data/lc/untzh/blend_confirm80_20261014/seed29/models/none/epoch-20.pt')
    parent=torch.load(parentpath,map_location='cpu',weights_only=True)
    E=torch.empty((TOTAL,200),device='cuda',dtype=torch.float32)
    torch.nn.init.normal_(E,std=.12,generator=torch.Generator(device='cuda').manual_seed(1099));E[:N]=parent['E'].cuda().float()
    assert torch.equal(E[:N].cpu().double(),parent['E'])
    initial=dict(E=E.cpu(),R=parent['R'].float());torch.save(initial,ROOT/'cache/initial.pt')
    tr=core.truthmap(old,np.concatenate([new,extra]))
    return initial,old,new,tr,dict(seconds=time.monotonic()-tick,train_source='087 cache',parent=str(parentpath),seed=29,expansion_seed=1099)

def risk_data(initial,old,new,tr,A,d):
    tick=time.monotonic();z=base.load(ROOT/'cache/cohort.npz');q=z['queries'];ids=z['ids'];rng=np.random.default_rng(880029)
    oldsample=rng.integers(0,N,size=(core.SIZE,64));e=initial['E'].cuda().double()
    tau=[]
    for lo in range(0,core.SIZE,256):
        s=(d['q'][lo:lo+256,None]*e[oldsample[lo:lo+256]]).sum(-1)
        tau.append(s.std(-1).clamp_min(1e-8))
    tau=torch.cat(tau);gold=(d['q']*e[q[:,2]]).sum(-1)
    source=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')
    alltrain=np.concatenate([old,new,source['extra']]);hc=np.bincount(alltrain[:,0],minlength=TOTAL);gc=np.bincount(alltrain[:,2],minlength=TOTAL);rc=np.bincount(alltrain[:,1],minlength=len(initial['R']))
    known=[sum(c>=N for c in tr[int(h),int(r)]) for h,r,g in q];oldrank=A[ids,1]
    fixed=np.column_stack([oldrank,11-oldrank,gold.cpu().numpy()/tau.cpu().numpy(),d['q'].norm(dim=-1).cpu().numpy(),
        np.log1p(hc[q[:,0]]),np.log1p(rc[q[:,1]]),np.log1p(gc[q[:,2]]),known]).astype(np.float32)
    groups=sorted(set(map(tuple,q[:,:2])));order=rng.permutation(len(groups));mapping={groups[int(j)]:(0 if i<.6*len(groups) else 1 if i<.8*len(groups) else 2) for i,j in enumerate(order)}
    split=np.array([mapping[tuple(row[:2])] for row in q],np.int8);rows=[];cols=[]
    for i,(h,r,g) in enumerate(q):
        for c in tr[int(h),int(r)]:
            if c>=N:rows.append(i);cols.append(c-N)
    np.savez(ROOT/'cache/risk-data.npz',fixed=fixed,split=split,gold=gold.cpu().numpy(),tau=tau.cpu().numpy(),oldrank=oldrank,
        known_rows=np.array(rows,dtype=np.int64),known_cols=np.array(cols,dtype=np.int64))
    d.update(gold=gold,tau=tau,fixed=fixed,split=split,oldrank=oldrank,known_rows=np.array(rows,dtype=np.int64),known_cols=np.array(cols,dtype=np.int64))
    return dict(seconds=time.monotonic()-tick,query_groups=len(groups),split_answers=np.bincount(split,minlength=3).tolist())

@torch.no_grad()
def frame(E,delta,d,ep):
    torch.cuda.synchronize();tick=time.monotonic();ranks=[];features=[]
    for lo in range(0,core.SIZE,256):
        hi=min(lo+256,core.SIZE);q=d['q'][lo:hi];gold=d['gold'][lo:hi];tau=d['tau'][lo:hi]
        s=q@E[N:].double().T;above=s>gold[:,None]
        mask=(d['known_rows']>=lo)&(d['known_rows']<hi)
        above[d['known_rows'][mask]-lo,d['known_cols'][mask]]=False
        ranks.append(d['oldrank'][lo:hi]+above.sum(-1).cpu().numpy())
        if ep<20:
            ids=d['candidates'][lo:hi]
            z=((q[:,None]*E[ids].double()).sum(-1)-gold[:,None])/tau[:,None]
            dz=(q[:,None]*delta[ids].double()).sum(-1)/tau[:,None];f=z+22*dz
            v=torch.stack([z[:,:32].max(-1).values,z.max(-1).values,z.topk(4,dim=-1).values.mean(-1),z.mean(-1),z.std(-1),
                (z>0).double().mean(-1),F.softplus(z).mean(-1),(z+dz).max(-1).values,f.max(-1).values,(f>0).double().mean(-1),dz.max(-1).values,dz.abs().mean(-1),torch.full_like(gold,ep/20)],1)
            features.append(v.float().cpu().numpy())
    payload=dict(ranks=np.concatenate(ranks))
    if ep<20:
        payload['X']=np.column_stack([d['fixed'],np.concatenate(features)])
        assert payload['X'].shape==(core.SIZE,21) and np.isfinite(payload['X']).all()
    torch.cuda.synchronize();score_seconds=time.monotonic()-tick;io=time.monotonic()
    np.savez(ROOT/'cache'/f'frame-{ep:02d}.npz',**payload)
    torch.save(dict(E=E.detach().cpu(),R=None),ROOT/'cache'/f'state-{ep:02d}.pt')
    return dict(score_seconds=score_seconds,io_seconds=time.monotonic()-io,dropped=int((payload['ranks']>10).sum()))

def trajectory(initial,new,t,d):
    tick=time.monotonic();E,R,opt=core.newmodel(initial);frames=[];train_seconds=0.;proposal_seconds=0.;prefix=None
    for ep in range(21):
        delta=None
        if ep<20:
            j=ep*22;st=time.monotonic();gl=torch.autograd.grad(base.learning(E,R,new,t,j),E)[0]
            delta=ex.proposal(E,opt,gl);proposal_seconds+=time.monotonic()-st
        frames.append(frame(E,delta,d,ep))
        if ep==10:
            prefix=dict(E=E.detach().cpu(),R=R.cpu(),optimizer=copy.deepcopy(opt.state_dict()));torch.save(prefix,ROOT/'cache/prefix10.pt')
        if ep<20:
            st=time.monotonic()
            for j in range(ep*22,(ep+1)*22):
                loss=base.learning(E,R,new,t,j);opt.zero_grad(set_to_none=True);loss.backward();opt.step();assert torch.isfinite(E).all()
            torch.cuda.synchronize();train_seconds+=time.monotonic()-st
        core.progress('trajectory',epoch=ep,dropped=frames[-1]['dropped'])
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models/none';out.mkdir(parents=True);torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    cost=dict(seconds=time.monotonic()-tick,training_seconds=train_seconds,proposal_seconds=proposal_seconds,frames=frames,trained_once=True)
    core.dump('trajectory.json',cost);return prefix,cost

def predictions(d):
    tick=time.monotonic();frames=[base.load(ROOT/'cache'/f'frame-{i:02d}.npz') for i in range(21)]
    mask=d['split']==0;X=np.concatenate([frames[i]['X'][mask] for i in range(8)]);y=np.concatenate([frames[i+1]['ranks'][mask]>10 for i in range(8)]).astype(int)
    cols=list(range(15))+[20];fitstart=time.monotonic();models={}
    models['transfer_current_tree']=joblib.load('/data/lc/untzh/risk58_20261013/cache/tree_current.joblib')
    if 0<y.sum()<len(y):
        weights=np.where(y,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
        model=make_pipeline(ColumnTransformer([('columns','passthrough',cols)]),HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,max_depth=3,learning_rate=.08,l2_regularization=1.,early_stopping=False,random_state=SEED))
        model.fit(X,y,histgradientboostingclassifier__sample_weight=weights);models['local_current_tree']=model
        joblib.dump(model,ROOT/'cache/local_current_tree.joblib')
    fitseconds=time.monotonic()-fitstart;report=dict(train_samples=len(y),train_positive_events=int(y.sum()),fit_seconds=fitseconds,columns=[ex.FEATURE_NAMES[i] for i in cols],groups={})
    rng=np.random.default_rng(880029)
    for group,epochs,split in [('development',range(8,10),1),('late_query_time_holdout',range(10,20),2)]:
        m=d['split']==split;per={};counts=collections.defaultdict(lambda:collections.defaultdict(lambda:dict(selected=0,hits=0,events=0)))
        ap=collections.defaultdict(list);prediction_seconds=collections.defaultdict(float)
        for ep in epochs:
            Xi=frames[ep]['X'][m];safe=frames[ep]['ranks'][m]<=10;target=frames[ep+1]['ranks'][m]>10;new=safe&target
            scores=ex.rule_scores(Xi);scores['random']=rng.random(len(Xi))
            for name,model in models.items():
                st=time.monotonic();scores[name]=model.predict_proba(Xi)[:,1];prediction_seconds[name]+=time.monotonic()-st
            per[str(ep)]={}
            for name,s in scores.items():
                safe_metrics=ex.selection_metrics(target,s,safe);all_metrics=ex.selection_metrics(target,s)
                per[str(ep)][name]=dict(safe=safe_metrics,all=all_metrics)
                if safe_metrics['average_precision'] is not None:ap[name].append(safe_metrics['average_precision'])
                for fraction in [.05,.1,.2]:
                    k=int(np.ceil(len(s)*fraction));selected=np.argsort(-s,kind='stable')[:k];c=counts[name][str(fraction)]
                    c['selected']+=k;c['hits']+=int(new[selected].sum());c['events']+=int(new.sum())
        operational={}
        for name,v in counts.items():
            operational[name]=dict(macro_safe_ap=float(np.mean(ap[name])) if ap[name] else None,prediction_seconds=prediction_seconds[name],budgets={})
            for fraction,c in v.items():operational[name]['budgets'][fraction]=dict(**c,recall=c['hits']/c['events'] if c['events'] else None,precision=c['hits']/c['selected'])
        report['groups'][group]=dict(per_epoch=per,operational=operational)
    report['total_seconds']=time.monotonic()-tick;core.dump('risk-prediction.json',report);return report

def main():
    torch.set_num_threads(2);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting');initial,old,new,tr,pc=prepare()
    t,tc=core.tape(SEED,new,tr,len(initial['R']));assert len(t['epochs'])==440
    A,ats=core.teacher(initial,old,tr);d,cc=core.cohort(initial,old,tr,A,SEED);rc=risk_data(initial,old,new,tr,A,d)
    prefix,trajectory_cost=trajectory(initial,new,t,d);risk=predictions(d)
    cal,cs=prev.calibrate(initial,prefix,new,t,d,220);costs=dict(preparation=pc,tape=tc,teacher_seconds=ats,cohort=cc,risk_data=rc,trajectory=trajectory_cost,risk_prediction_seconds=risk['total_seconds'],calibration_seconds=cs,training={})
    for arm in core.ARMS:
        if arm=='none':continue
        costs['training'][arm]=prev.train(arm,initial,prefix,new,t,d,cal,220);core.dump('costs-running.json',costs)
    report,es=prev.evaluate(initial,old,new,tr,A);costs['evaluation_seconds']=es;costs['total_seconds']=time.monotonic()-tick
    core.dump('summary.json',dict(metrics=report,costs=costs,risk=risk));core.progress('complete',seconds=costs['total_seconds'])

if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
