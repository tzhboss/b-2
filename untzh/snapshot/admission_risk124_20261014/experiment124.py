import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import binom
from sklearn.metrics import average_precision_score, brier_score_loss
sys.path.insert(0,'/data/lc/untzh/complex_transfer118_20261014')
import experiment118 as cx
base,core=cx.base,cx.core
ROOT=Path(__file__).resolve().parent

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def evaluate(y,score,probability=False):
    order=np.argsort(-score,kind='stable');budgets={}
    for fraction in [.01,.05,.1,.2]:
        k=int(np.ceil(len(y)*fraction));hits=int(y[order[:k]].sum())
        budgets[str(fraction)]=dict(selected=k,hits=hits,precision=hits/k if k else None,recall=hits/int(y.sum()) if y.any() else None)
    r=dict(ap=float(average_precision_score(y,score)) if y.any() else None,budgets=budgets)
    if probability:
        r.update(brier=float(brier_score_loss(y,score)),expected_drops=float(score.sum()))
        bins=np.array([0,.001,.01,.1,.5,.9,.99,.999,1.0000000001]);rows=[]
        for a,b in zip(bins,bins[1:]):
            mask=(score>=a)&(score<b)
            rows.append(dict(lower=float(a),upper=float(b),n=int(mask.sum()),mean_predicted=float(score[mask].mean()) if mask.any() else None,observed_rate=float(y[mask].mean()) if mask.any() else None))
        r['reliability_bins']=rows
    return r

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic();status('prepare');(ROOT/'cache').mkdir(exist_ok=True);reports={}
    for parent in [43,17,29,'WN_DistMult','WN_ComplEx']:
        fb=isinstance(parent,int);complex_model=parent=='WN_ComplEx';n,total=(11258,13134) if fb else (24567,28660)
        if fb:
            src=Path('/data/lc/untzh')/{43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[parent]
            initial=torch.load(src/'cache/initial.pt',map_location='cpu',weights_only=True)
            z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz');truths=core.truthmap(z['old'],np.concatenate([z['new'],z['extra']]))
            queries=base.load(src/'evaluation/queries.npz')
            if parent!=43:queries['old_valid']=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['old_valid']
            phasepaths=dict(standard=Path(f'/data/lc/untzh/fb_phase117_20261014/seed{parent}/standard'),low=Path(f'/data/lc/untzh/fb_phase117_20261014/seed{parent}/low'))
        else:
            src=Path('/data/lc/untzh')/('complex_transfer118_20261014' if complex_model else 'wn_transfer112_20261014')
            initial=torch.load(src/'cache/initial.pt',map_location='cpu',weights_only=True)
            z=base.load('/data/lc/untzh/wn_transfer112_20261014/cache/train.npz');truths=core.truthmap(z['old'],z['new'])
            queries=base.load(src/'evaluation/queries.npz');phasepaths=None
        E,R=initial['E'].cuda().double(),initial['R'].cuda().double()
        for split in ['old_train','old_valid']:
            status('features',parent=str(parent),split=split);st=time.monotonic();a=queries[split]
            if fb and parent!=43:
                gold=base.load(f'/data/lc/untzh/fast_old_eval107_20261014/seed{parent}/{split}.npz')['gold']
                teacher=base.load(src/'evaluation/teacher.npz')['old_train'] if split=='old_train' else base.load(f'/data/lc/untzh/full_valid102_20261014/seed{parent}/teacher.npz')['old_valid']
            else:
                cache=base.load(src/'cache'/f'fast-{split}.npz');gold=cache['gold'];teacher=cache['teacher']
            ids=np.flatnonzero(teacher[:,1]<=10);a=a[ids];gold=gold[ids];oldrank=teacher[ids,1];capacity=10-oldrank
            norm=[]
            for lo in range(0,len(a),256):
                b=a[lo:lo+256];q=cx.tail_query(E[b[:,0]],R[b[:,1]]) if complex_model else E[b[:,0]]*R[b[:,1]]
                norm.append(q.norm(dim=-1).cpu().numpy())
            norm=np.concatenate(norm);legal=np.array([total-n-sum(c>=n for c in truths[int(h),int(r)]) for h,r,g in a]);cost=time.monotonic()-st
            # Predictions depend only on parent reference facts and new TRAIN truth count.
            predictions={}
            for mode,sigma in [('standard',.12),('low',.03),('half',.015),('zero',0.)]:
                scale=sigma*norm;p=(gold<0).astype(float)
                ix=scale>0;p[ix]=ndtr(-gold[ix]/scale[ix])
                probability=binom.sf(capacity,legal,p)
                assert np.isfinite(probability).all()
                scores=dict(gaussian_capacity=probability,normal_tail=p,expected_excess=legal*p-capacity,
                            negative_gold=-gold,boundary=oldrank.astype(float),random=np.random.default_rng(1240017).random(len(ids)))
                predictions[mode]=scores
                np.savez(ROOT/'cache'/f'{parent}-{split}-{mode}-prediction.npz',ids=ids,probability=probability,p=p,gold=gold,querynorm=norm,oldrank=oldrank,legal=legal)
            modes={}
            # Existing expansion ranks are read only after every prior-based prediction is saved.
            for mode,sigma in [('standard',.12),('low',.03),('half',.015),('zero',0.)]:
                if fb:
                    path=phasepaths[mode]/f'{split}-phases.npz' if mode in ['standard','low'] else Path(f'/data/lc/untzh/train_init121_20261014/seed{parent}/evaluation/{"random_half" if mode=="half" else "zero"}-{split}-phases.npz')
                    with np.load(path) as f:observed=f['expanded_initial'][ids,1]
                elif complex_model:
                    path=src/'evaluation'/f'{mode}-{split}-phases.npz' if mode in ['standard','low'] else Path(f'/data/lc/untzh/wn_init122_20261014/ComplEx/evaluation/{"random_half" if mode=="half" else "zero"}-{split}-phases.npz')
                    with np.load(path) as f:observed=f['expanded_initial'][ids,1]
                else:
                    if mode=='standard':
                        with np.load(f'/data/lc/untzh/phase_diagnostic114_20261014/{split}-phases.npz') as f:observed=f['expanded_initial'][ids,1]
                    elif mode=='low':
                        # 115 saved only VALID low expansion; compute analytic zero check on TRAIN via122 zero instead, no reranking.
                        if split=='old_train':continue
                        observed=base.load('/data/lc/untzh/init_onset115_20261014/evaluation/low-expanded-initial.npz')['old_valid'][ids,1]
                    else:
                        with np.load(f'/data/lc/untzh/wn_init122_20261014/DistMult/evaluation/{"random_half" if mode=="half" else "zero"}-{split}-phases.npz') as f:observed=f['expanded_initial'][ids,1]
                y=observed>10;scores=predictions[mode];methods={k:evaluate(y,v,k=='gaussian_capacity') for k,v in scores.items()}
                zero_check=None
                if mode=='zero':
                    expected_rank=oldrank+legal*(gold<0)
                    zero_check=dict(rank_equal=bool(np.array_equal(expected_rank,observed)),mismatch=int((expected_rank!=observed).sum()),negative_gold_queries=int((gold<0).sum()))
                modes[mode]=dict(sigma=sigma,n=len(ids),drops=int(y.sum()),methods=methods,zero_rank_formula=zero_check)
                print('ADMISSION124',str(parent),split,mode,'observed',int(y.sum()),'expected',methods['gaussian_capacity']['expected_drops'],'ap',methods['gaussian_capacity']['ap'],flush=True)
            reports[f'{parent}-{split}']=dict(features_seconds=cost,modes=modes);base.write(ROOT/'summary-running.json',dict(cases=reports))
    base.write(ROOT/'summary.json',dict(cases=reports,total_seconds=time.monotonic()-tick,no_training=True,
        no_new_embedding_score_scan=True,gaussian_prior_assumption=True,reference_gold_and_oldrank_known=True));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
