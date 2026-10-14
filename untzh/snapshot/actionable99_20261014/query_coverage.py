import json
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/risk_increment88_20261014')
ROUTE=Path('/data/lc/untzh/anchor_route97_20261014')


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def main():
    tick=time.monotonic();c=load(SOURCE/'cache/cohort.npz');qs=load(SOURCE/'evaluation/queries.npz')
    teacher=load(SOURCE/'evaluation/teacher.npz');ordinary=load(SOURCE/'evaluation/none.npz')
    keys=[tuple(q[:2]) for q in c['queries']];groups=sorted(set(keys));mapping={key:i for i,key in enumerate(groups)}
    index=np.array([mapping[key] for key in keys]);result=dict(cohort_answers=len(keys),cohort_queries=len(groups),arms={})
    for arm in ['anchor_tree','anchor_pressure128','anchor_random_pool','uniform']:
        steps=load(SOURCE/'cache/protection.npz')['selections'][:220] if arm=='uniform' else [s['selected'] for s in json.loads((ROUTE/'models'/arm/'steps.json').read_text())]
        assert len(steps)==220
        counts=np.zeros(len(groups),np.int64);distinct=[];selected=set()
        for step in steps:
            ids=index[step];np.add.at(counts,ids,1);distinct.append(len(np.unique(ids)));selected.update(keys[i] for i in step)
        p=counts/counts.sum();entry=dict(mean_distinct_queries_per_step=float(np.mean(distinct)),min_distinct_queries=int(min(distinct)),
            total_query_draws=int(counts.sum()),unique_queries=int((counts>0).sum()),effective_query_exposure=float(1/(p*p).sum()),max_query_exposure=int(counts.max()),coverage={})
        for split in ['old_train','old_valid']:
            covered=np.array([tuple(q[:2]) in selected for q in qs[split]]);eligible=teacher[split][:,1]<=10;dropped=eligible&(ordinary[split][:,1]>10)
            entry['coverage'][split]=dict(all_answers=len(covered),covered_answers=int(covered.sum()),eligible=int(eligible.sum()),covered_eligible=int((covered&eligible).sum()),
                                          ordinary_dropped=int(dropped.sum()),covered_ordinary_dropped=int((covered&dropped).sum()))
        result['arms'][arm]=entry
    result['seconds']=time.monotonic()-tick
    (ROOT/'query-coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
