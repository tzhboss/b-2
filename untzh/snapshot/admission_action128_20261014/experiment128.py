import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/data/lc/untzh/admission_risk124_20261014')
import experiment124 as previous
base,core=previous.base,previous.core
ROOT=Path(__file__).resolve().parent
MODES=['standard','low','half','zero']
BUDGET=.01

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def select(predictions,mask):
    n=int(mask.sum());floor=float(predictions['zero']['probability'][mask].sum())
    rows={k:dict(expected_drops=float(p['probability'][mask].sum()),extra_over_zero=float(p['probability'][mask].sum())-floor) for k,p in predictions.items()}
    chosen=next(k for k in MODES if rows[k]['extra_over_zero']<=BUDGET*n+1e-9)
    return dict(chosen=chosen,n=n,budget_fraction=BUDGET,zero_floor=floor,allowed_extra=BUDGET*n,expectations=rows)

def paths(parent,mode):
    if isinstance(parent,int):
        src=Path('/data/lc/untzh')/{43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[parent]
        if mode=='standard':
            rankpath=src/'evaluation/none.npz' if parent==43 else Path(f'/data/lc/untzh/full_valid102_20261014/seed{parent}/none.npz')
            phase=Path(f'/data/lc/untzh/fb_phase117_20261014/seed{parent}/standard/old_valid-phases.npz')
        elif mode=='low':
            rankpath=Path(f'/data/lc/untzh/lowinit_confirm116_20261014/seed{parent}/evaluation/none.npz')
            phase=Path(f'/data/lc/untzh/fb_phase117_20261014/seed{parent}/low/old_valid-phases.npz')
        else:
            arm='random_half' if mode=='half' else 'zero';rankpath=Path(f'/data/lc/untzh/train_init121_20261014/seed{parent}/evaluation/{arm}.npz');phase=rankpath.parent/f'{arm}-old_valid-phases.npz'
    else:
        name='ComplEx' if parent=='WN_ComplEx' else 'DistMult';src=Path('/data/lc/untzh')/('complex_transfer118_20261014' if name=='ComplEx' else 'wn_transfer112_20261014')
        if mode=='standard':
            rankpath=src/'evaluation'/('standard_none.npz' if name=='ComplEx' else 'none.npz')
            phase=src/'evaluation/standard-old_valid-phases.npz' if name=='ComplEx' else Path('/data/lc/untzh/phase_diagnostic114_20261014/old_valid-phases.npz')
        elif mode=='low':
            rankpath=src/'evaluation/none.npz' if name=='ComplEx' else Path('/data/lc/untzh/init_onset115_20261014/evaluation/lowinit_none.npz')
            phase=src/'evaluation/low-old_valid-phases.npz' if name=='ComplEx' else Path('/data/lc/untzh/init_onset115_20261014/evaluation/low-expanded-initial.npz')
        else:
            arm='random_half' if mode=='half' else 'zero';rankpath=Path(f'/data/lc/untzh/wn_init122_20261014/{name}/evaluation/{arm}.npz');phase=rankpath.parent/f'{arm}-old_valid-phases.npz'
    return src,rankpath,phase

def main():
    tick=time.monotonic();status('decide');(ROOT/'cache').mkdir(exist_ok=True);bundles=[];decisions={}
    for parent in [43,17,29,'WN_DistMult','WN_ComplEx']:
        src,_,_=paths(parent,'standard');fb=isinstance(parent,int)
        queries=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz') if fb else base.load(src/'evaluation/queries.npz')
        q=queries['old_valid'];groups,inverse=np.unique(q[:,:2],axis=0,return_inverse=True)
        permutation=np.random.default_rng(1280017).permutation(len(groups));cal_group=np.zeros(len(groups),bool);cal_group[permutation[:len(groups)//2]]=True
        cal=cal_group[inverse];held=~cal
        prediction={k:base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{parent}-old_valid-{k}-prediction.npz') for k in MODES}
        trprediction={k:base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{parent}-old_train-{k}-prediction.npz') for k in MODES}
        ids=prediction['zero']['ids'];assert all(np.array_equal(p['ids'],ids) for p in prediction.values())
        train_policy=select(trprediction,np.ones(len(trprediction['zero']['ids']),bool));reference_policy=select(prediction,cal[ids])
        np.savez(ROOT/'cache'/f'{parent}-split.npz',calibration=cal,evaluation=held,groups=groups,inverse=inverse,cal_group=cal_group)
        decisions[str(parent)]=dict(train_only=train_policy,historical_reference=reference_policy,
            cal_answers=int(cal.sum()),eval_answers=int(held.sum()),cal_groups=int(cal_group.sum()),eval_groups=int((~cal_group).sum()),
            history_reference_is_old_valid_calibration=True,current_endpoint_labels_not_used=True)
        bundles.append((parent,queries,cal,held,prediction));print('DECISION128',parent,train_policy['chosen'],reference_policy['chosen'],flush=True)
    # Freeze all choices before reading any current expansion or trained endpoint rank.
    base.write(ROOT/'decisions.json',decisions);reports={}
    for parent,queries,cal,held,prediction in bundles:
        status('evaluate',parent=str(parent));fb=isinstance(parent,int);base.N=core.N=11258 if fb else 24567
        src,_,_=paths(parent,'standard')
        if fb:teacher=base.load(src/'evaluation/teacher.npz')['old_valid'] if parent==43 else base.load(f'/data/lc/untzh/full_valid102_20261014/seed{parent}/teacher.npz')['old_valid']
        else:teacher=base.load(src/'cache/fast-old_valid.npz')['teacher']
        metrics={};initial={};nq=queries['new_valid'];oldq=queries['old_valid']
        for mode in MODES:
            _,rp,pp=paths(parent,mode);rank=base.load(rp)
            if parent=='WN_DistMult' and mode=='low':expanded=base.load(pp)['old_valid']
            else:
                with np.load(pp) as p:expanded=p['expanded_initial']
            metric=dict(new_valid=base.metrics(rank['new_valid'],nq),old_eval=base.metrics(rank['old_valid'][held],oldq[held]),old_cal=base.metrics(rank['old_valid'][cal],oldq[cal]))
            metric['old_eval']['drops']=core.drops(rank['old_valid'][held],teacher[held]);metric['old_cal']['drops']=core.drops(rank['old_valid'][cal],teacher[cal])
            ids=prediction[mode]['ids'];y=expanded[ids,1]>10;mask=held[ids]
            init=dict(eval_n=int(mask.sum()),eval_drops=int(y[mask].sum()),expected_eval_drops=float(prediction[mode]['probability'][mask].sum()),
                      cal_n=int((~mask).sum()),cal_drops=int(y[~mask].sum()),expected_cal_drops=float(prediction[mode]['probability'][~mask].sum()))
            metrics[mode]=metric;initial[mode]=init
        selected={}
        for policy in ['train_only','historical_reference']:
            chosen=decisions[str(parent)][policy]['chosen'];selected[policy]=dict(alias=chosen,initial=initial[chosen],metrics=metrics[chosen])
        reports[str(parent)]=dict(decision=decisions[str(parent)],fixed_controls=metrics,initial=initial,selected=selected)
        base.write(ROOT/'summary-running.json',dict(parents=reports))
        print('EVALUATED128',parent,{k:dict(alias=v['alias'],newMRR=v['metrics']['new_valid']['all']['filtered_train_truths']['mrr'],old_eval_drops=v['metrics']['old_eval']['drops']['filtered_train_truths']['drops']) for k,v in selected.items()},flush=True)
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,no_training=True,
        retrospective_action_simulation=True,historical_reference_subset_is_calibration_not_evaluation=True,budget=BUDGET))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
