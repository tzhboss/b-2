import json
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/risk_increment88_20261014')

def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}

def main():
    start=time.monotonic()
    teacher=load(SOURCE/'evaluation/teacher.npz')['old_train'][:,1]
    baseline=load(SOURCE/'evaluation/none.npz')['old_train'][:,1]
    ids=load(SOURCE/'cache/cohort.npz')['ids']
    source=load('/data/lc/tzh/experiment4/entity_diagnostic42_20261012/artifacts/data/train.npz')
    boundary=len(source['old'])
    recent=np.arange(len(teacher))>=boundary
    assert boundary==223284 and len(teacher)==260498
    in_cohort=np.zeros(len(teacher),bool);in_cohort[ids]=True
    groups=dict(all=np.ones(len(teacher),bool),snapshot0_facts=~recent,snapshot1_facts=recent,cohort=in_cohort,outside_cohort=~in_cohort)
    report=dict(source='saved TRAIN-only filtered ranks; no scoring or retraining',groups={},selections={})
    for arm in ['absolute','absolute075','transfer_tree','pressure128','pressure32','random_pool']:
        path=(SOURCE if arm.startswith('absolute') else ROOT)/'evaluation'/f'{arm}.npz'
        ranks=load(path)['old_train'][:,1];report['groups'][arm]={}
        for name,mask in groups.items():
            eligible=mask&(teacher<=10)
            rescue=eligible&(baseline>10)&(ranks<=10);newharm=eligible&(baseline<=10)&(ranks>10)
            report['groups'][arm][name]=dict(answers=int(mask.sum()),eligible=int(eligible.sum()),
                ordinary_drops=int((eligible&(baseline>10)).sum()),treated_drops=int((eligible&(ranks>10)).sum()),
                rescued=int(rescue.sum()),new_drops=int(newharm.sum()),mrr_delta=float(np.mean(1/ranks[mask]-1/baseline[mask])))
        if not arm.startswith('absolute'):
            exposures=np.zeros(len(ids),int)
            steps=json.loads((ROOT/'models'/arm/'steps.json').read_text())
            for step in steps:np.add.at(exposures,step['selected'],1)
            p=exposures/exposures.sum()
            report['selections'][arm]=dict(draws=int(exposures.sum()),unique_answers=int((exposures>0).sum()),
                effective_exposure_answers=float(1/(p*p).sum()),max_exposures=int(exposures.max()),
                snapshot1_fraction=float(exposures[ids>=boundary].sum()/exposures.sum()))
    report['seconds']=time.monotonic()-start
    (ROOT/'paired-ranks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
