import json
import os
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REMOTE = Path('/data/lc/untzh')

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}

def status(stage, **kw):
    write(ROOT/'status.json', dict(stage=stage, pid=os.getpid(), **kw))

def description(x):
    return dict(n=len(x), negative=int((x<0).sum()), negative_fraction=float((x<0).mean()) if len(x) else None,
                quantiles={str(p):float(np.quantile(x,p)) for p in [.01,.05,.1,.25,.5,.75,.9,.95,.99]} if len(x) else {})

def group(mask, normgold, oldrank, endpoints, prefix=None):
    v=dict(n=int(mask.sum()), normalized_gold=description(normgold[mask]),
           parent_mrr=float((1/oldrank[mask]).mean()) if mask.any() else None, arms={})
    for arm, rank in endpoints.items():
        after=rank[mask]
        detail=dict(drops=int((after>10).sum()), drop_fraction=float((after>10).mean()) if len(after) else None,
                    eligible_mrr=float((1/after).mean()) if len(after) else None)
        if prefix is not None:
            before=prefix[mask]
            detail.update(prefix_drops=int((before>10).sum()), late_new=int(((before<=10)&(after>10)).sum()),
                          rescued=int(((before>10)&(after<=10)).sum()))
        v['arms'][arm]=detail
    return v

def parent_report(parent):
    tick=time.monotonic()
    src=REMOTE/('increment87_20261014' if parent=='17' else 'complex_transfer118_20261014')
    queries=load(src/'evaluation/queries.npz')
    if parent=='17':
        queries['old_valid']=load(REMOTE/'full_valid102_20261014/queries.npz')['old_valid']
    pred={s:load(REMOTE/f'admission_risk124_20261014/cache/{parent}-{s}-half-prediction.npz') for s in ['old_train','old_valid']}
    features={s:load(REMOTE/f'risk_link136_20261014/{parent}/cache/{s}-features.npz') for s in pred}
    endpoints={arm:load(REMOTE/f'entity_action137_20261014/{parent}/evaluation/{arm}.npz') for arm in ['none','fixed_anchor','fixed_replay']}
    assert all(np.all(np.diff(p['ids'])>0) for p in pred.values())
    facts={s:queries[s][p['ids']] for s,p in pred.items()}
    normgold={s:p['gold']/np.maximum(p['querynorm'],1e-30) for s,p in pred.items()}
    ranks={s:{arm:z[s][p['ids'],1] for arm,z in endpoints.items()} for s,p in pred.items()}
    oldrank={s:p['oldrank'] for s,p in pred.items()}
    assert all(np.all((x>=1)&(x<=10)) for x in oldrank.values())
    relation_count=474 if parent=='17' else 22
    def keys(a): return a[:,0].astype(np.int64)*relation_count+a[:,1]
    trainkey=keys(queries['old_train'])
    validkey=keys(facts['old_valid'])
    eligiblekey=keys(facts['old_train'])
    cohortkey=keys(queries['old_train'][features['old_train']['ids']])
    in_train=np.isin(validkey,np.unique(trainkey))
    in_eligible=np.isin(validkey,np.unique(eligiblekey))
    in_cohort=np.isin(validkey,np.unique(cohortkey))
    trainfacts={tuple(int(v) for v in row) for row in queries['old_train']}
    exact=np.array([tuple(int(v) for v in row) in trainfacts for row in facts['old_valid']],bool)
    prefix=features['old_valid']['prefix_rank']
    assert np.array_equal(features['old_valid']['ids'],pred['old_valid']['ids'])
    loading=time.monotonic()-tick
    populations={s:group(np.ones(len(facts[s]),bool),normgold[s],oldrank[s],ranks[s],prefix if s=='old_valid' else None) for s in facts}
    strata={name:group(mask,normgold['old_valid'],oldrank['old_valid'],ranks['old_valid'],prefix) for name,mask in [
        ('train_query_seen',in_train),('train_query_unseen',~in_train),('eligible_train_query_seen',in_eligible),
        ('cohort_query_seen',in_cohort),('cohort_query_unseen',~in_cohort),('exact_train_fact',exact),
        ('seen_query_unseen_answer',in_train&~exact),('negative_gold',normgold['old_valid']<0),('nonnegative_gold',normgold['old_valid']>=0)]}
    # Relation/rank groups and their membership fixed by cached parent quantities;
    # endpoint outcomes do not determine group boundaries or weights.
    trainstratum=facts['old_train'][:,1]*11+oldrank['old_train'].astype(np.int64)
    validstratum=facts['old_valid'][:,1]*11+oldrank['old_valid'].astype(np.int64)
    cells=[]
    accum={name:0. for name in ['train_mean_gold','valid_mean_gold','train_negative','valid_negative','train_drops','valid_drops']}
    matched=0
    for key in np.unique(validstratum):
        tm=trainstratum==key
        vm=validstratum==key
        if not tm.any(): continue
        tn,vn=int(tm.sum()),int(vm.sum())
        values=dict(train_mean_gold=float(normgold['old_train'][tm].mean()),valid_mean_gold=float(normgold['old_valid'][vm].mean()),
                    train_negative=float((normgold['old_train'][tm]<0).mean()),valid_negative=float((normgold['old_valid'][vm]<0).mean()),
                    train_drops=float((ranks['old_train']['none'][tm]>10).mean()),valid_drops=float((ranks['old_valid']['none'][vm]>10).mean()))
        for name,val in values.items(): accum[name]+=vn*val
        matched+=vn
        cells.append(dict(relation=int(key//11),oldrank=int(key%11),train_n=tn,valid_n=vn,**values))
    adjusted=dict(matched_valid=matched,unmatched_valid=len(validstratum)-matched,
                  valid_weighted_common_cells={k:v/matched if matched else None for k,v in accum.items()},
                  no_causal_or_probability_calibration_claim=True)
    cohort=features['old_train']
    pos=np.searchsorted(pred['old_train']['ids'],cohort['ids'])
    assert np.array_equal(pred['old_train']['ids'][pos],cohort['ids'])
    cohort_report=group(np.ones(len(pos),bool),normgold['old_train'][pos],oldrank['old_train'][pos],
                        {arm:rank[pos] for arm,rank in ranks['old_train'].items()},cohort['prefix_rank'])
    np.savez(ROOT/parent/'reference-membership.npz',ids=pred['old_valid']['ids'],train_query_seen=in_train,
             eligible_query_seen=in_eligible,cohort_query_seen=in_cohort,exact_train_fact=exact)
    write(ROOT/parent/'matched-cells.json',cells)
    return dict(populations=populations,valid_strata=strata,relation_rank_adjusted=adjusted,
                train_cohort=cohort_report,cost=dict(load_and_membership_seconds=loading,total_seconds=time.monotonic()-tick),
                diagnosis_not_action_input=True)

def main():
    tick=time.monotonic()
    reports={}
    for parent in ['17','WN_ComplEx']:
        status('diagnose',parent=parent)
        (ROOT/parent).mkdir(parents=True,exist_ok=True)
        result=parent_report(parent)
        reports[parent]=result
        write(ROOT/parent/'summary.json',result)
        print('GAP139',parent,{s:(v['normalized_gold']['negative_fraction'],v['arms']['none']['drops']) for s,v in result['populations'].items()},
              {s:(result['valid_strata'][s]['n'],result['valid_strata'][s]['arms']['none']['drops']) for s in ['train_query_seen','train_query_unseen','cohort_query_seen','cohort_query_unseen']},
              result['relation_rank_adjusted'],flush=True)
    write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,new_training_configs=0,
                                 no_test=True,no_model_scoring=True,interpretation_pending=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        status('failed',error=repr(exc))
        raise
