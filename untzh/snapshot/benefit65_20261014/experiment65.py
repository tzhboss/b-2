"""True short-branch TRAIN intervention labels and benefit-routing learners."""
import collections
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE=Path('/data/lc/untzh/search63_20261013')
sys.path.insert(0,str(SOURCE))
import experiment63 as core
ex=core.ex
ROOT=Path(__file__).resolve().parent
core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={
 'benefit_ridge_static':dict(signal='absolute',route='ridge_static'),
 'benefit_ridge_dynamic':dict(signal='absolute',route='ridge_dynamic'),
 'benefit_tree_static':dict(signal='absolute',route='tree_static'),
 'benefit_tree_dynamic':dict(signal='absolute',route='tree_dynamic'),
}


@torch.no_grad()
def cohort_ranks(E,data,ids):
    ranks=[]
    for lo in range(0,len(ids),256):
        chosen=ids[lo:lo+256]
        score=data['q_t'][chosen]@E[ex.N:].double().T
        above=(score>data['gold_t'][chosen,None])&data['legal_t'][chosen]
        ranks.append(above.sum(-1).cpu().numpy()+data['oldrank'][chosen])
    return np.concatenate(ranks)


@torch.no_grad()
def new_ranks(E,R,queries,truth):
    return ex.base.ranks(E.detach().double(),R.double(),queries,truth,arm='TRAIN_intervention_label',split='new_train_probe')[:,1]


def short_branch(initial,prefix,new,tape,data,action=None):
    E,R,opt=core.model(initial,prefix)
    coefficient=json.loads((core.PRIOR/'coefficient.json').read_text())['lambda']
    for j in range(370,407):
        loss=ex.base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        if action is not None:
            p=core.penalty(E,data,np.array([action]),'absolute')
            gp=torch.autograd.grad(p,E)[0]
            bn,pn=float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
            alpha=min(coefficient,.1*bn/pn) if pn else 0.
            gl=gl+alpha*gp
        opt.zero_grad(set_to_none=True);E.grad=gl;opt.step()
    return E.detach(),R


def features(E,data):
    X,_=ex.features(E,torch.zeros_like(E),data,10)
    return X[:,:15]


def labels(initial,prefix,new,tape,data,truth):
    start=time.monotonic();rng=np.random.default_rng(17)
    X=ex.base.load(core.PRIOR/'cache/frame-10.npz')['X'][:,:15]
    risk=joblib.load(core.PRIOR/'cache/tree_current.joblib').predict_proba(
        ex.base.load(core.PRIOR/'cache/frame-10.npz')['X'])[:,1]
    actions=[];splits=[]
    for split,count in [(0,128),(1,32)]:
        pool=np.flatnonzero(data['split']==split)
        ranked=pool[np.argsort(-risk[pool],kind='stable')]
        high=rng.choice(ranked[:max(count,len(pool)//5)],count//2,replace=False)
        remaining=np.setdiff1d(pool,high)
        random=rng.choice(remaining,count-count//2,replace=False)
        actions.extend(np.concatenate([high,random]));splits.extend([split]*count)
    actions=np.asarray(actions);splits=np.asarray(splits)
    masks=[(new[:,0]<ex.N)&(new[:,2]>=ex.N),(new[:,0]>=ex.N)&(new[:,2]<ex.N),
           (new[:,0]>=ex.N)&(new[:,2]>=ex.N)]
    probes=[new[rng.choice(np.flatnonzero(mask),256,replace=False)] for mask in masks]
    probe=np.concatenate(probes)
    # Cost labels use TRAIN probe facts only; no VALID or TEST.
    baseline,R=short_branch(initial,prefix,new,tape,data)
    brnew=new_ranks(baseline,R,probe,truth)
    base_new_mrr=np.array([np.mean(1/brnew[lo:lo+256]) for lo in [0,256,512]])
    populations={split:np.flatnonzero(data['split']==split) for split in [0,1]}
    base_old={split:cohort_ranks(baseline,data,pop) for split,pop in populations.items()}
    result=[]
    for i,(action,split) in enumerate(zip(actions,splits)):
        E,R=short_branch(initial,prefix,new,tape,data,int(action))
        rank=cohort_ranks(E,data,populations[int(split)])
        old=base_old[int(split)]
        gain=float(np.mean(1/rank)-np.mean(1/old))
        dropgain=float(np.mean(old>10)-np.mean(rank>10))
        rrnew=new_ranks(E,R,probe,truth)
        mrnew=np.array([np.mean(1/rrnew[lo:lo+256]) for lo in [0,256,512]])
        cost=float(np.maximum(0,1-mrnew/base_new_mrr).mean())
        result.append([gain+.1*dropgain,cost,gain,dropgain,*mrnew])
        if (i+1)%8==0:ex.progress('intervention_labels',completed=i+1,total=len(actions),seconds=time.monotonic()-start)
    np.savez(ROOT/'intervention-labels.npz',actions=actions,split=splits,X=X[actions],outcomes=np.asarray(result),
             probe=probe,baseline_new_mrr=base_new_mrr)
    ex.dump('label-costs.json',{'seconds':time.monotonic()-start,'branches':len(actions),'steps_per_branch':37,
        'fit_actions':128,'development_actions':32,'source':'TRAIN only; each branch shares prefix10 and steps370:407',
        'intervention':'one answer repeated for 37 steps, 128 legal candidates, rho .1; interactions when combining32 actions remain unmodelled'})
    return X,np.asarray(result),actions,splits


def fit(X,outcomes,actions,splits):
    started=time.monotonic();fitmask=splits==0;devmask=splits==1
    scales=np.std(outcomes[fitmask,:2],axis=0)
    if np.any(scales==0):raise RuntimeError('No variation in true intervention gain/cost; utility learner not identified')
    target=outcomes[:,0]/scales[0]-outcomes[:,1]/scales[1]
    learners={
       'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10.)),
       'tree':HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=3,max_depth=2,min_samples_leaf=10,
               learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=17)}
    report={'scales':scales.tolist(),'positive_gain_actions':int((outcomes[:,0]>0).sum()),
            'positive_cost_actions':int((outcomes[:,1]>0).sum()),'models':{}}
    for name,m in learners.items():
        m.fit(X[actions[fitmask]],target[fitmask]);joblib.dump(m,ROOT/f'{name}.joblib')
        pred=m.predict(X[actions[devmask]])
        corr=float(spearmanr(pred,target[devmask]).statistic)
        top=np.argsort(-pred)[:8]
        report['models'][name]={'dev_spearman':corr if np.isfinite(corr) else None,
            'dev_top8_real_utility':float(target[devmask][top].mean()),'dev_mean_real_utility':float(target[devmask].mean())}
    report['seconds']=time.monotonic()-started;ex.dump('benefit-fit.json',report)
    return learners,report


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(core.PRIOR/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    data=core.prepare_data(initial,old,new)
    truth=collections.defaultdict(set)
    for h,r,t in np.concatenate([old,new]):truth[int(h),int(r)].add(int(t))
    X,outcomes,actions,splits=labels(initial,prefix,new,tape,data,truth)
    learners,fit_report=fit(X,outcomes,actions,splits)
    def route(E,R,opt,gl,data,tree,policy):
        name,dynamic=policy.split('_')
        Xi=features(E,data) if dynamic=='dynamic' else X
        score=learners[name].predict(Xi)
        return np.argsort(-score,kind='stable')[:int(np.ceil(.2*ex.SIZE))]
    core.route=route
    calibration=json.loads(Path('/data/lc/untzh/search63_20261013/calibration.json').read_text())
    costs={'labels':json.loads((ROOT/'label-costs.json').read_text()),'fit':fit_report,'training':{}}
    for name,spec in ARMS.items():
        costs['training'][name]=core.train(name,spec,initial,prefix,new,tape,data,None,calibration)
        ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs)
    ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
