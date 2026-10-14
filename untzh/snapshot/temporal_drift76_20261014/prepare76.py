"""Prepare the published ICEWS05-15 split for chronological diagnostics."""
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    start = time.perf_counter()
    raw = {}
    for split in ('train', 'valid', 'test'):
        rows = []
        for line in (ROOT/'data/src_data/ICEWS05-15'/split).read_text().splitlines():
            fields = line.split('\t')
            assert len(fields) == 4 and len(fields[3]) == 10
            rows.append(tuple(fields))
        raw[split] = rows
    allrows = sum(raw.values(), [])
    first = {}
    rfirst = {}
    for h, r, t, date in raw['train']:
        year = int(date[:4])
        for e in (h, t):
            first[e] = min(first.get(e, 9999), year)
        rfirst[r] = min(rfirst.get(r, 9999), year)
    entities = sorted({e for h,r,t,date in allrows for e in (h,t)}, key=lambda e:(first.get(e,9999),e))
    relations = sorted({r for h,r,t,date in allrows}, key=lambda r:(rfirst.get(r,9999),r))
    dates = sorted({date for h,r,t,date in allrows})
    ei = {e:i for i,e in enumerate(entities)}
    ri = {r:i for i,r in enumerate(relations)}
    ti = {t:i for i,t in enumerate(dates)}
    nr = len(relations)
    eyear = np.array([first.get(e,9999) for e in entities])
    ryear = np.array([rfirst.get(r,9999) for r in relations])
    dyear = np.array([int(date[:4]) for date in dates])
    train_set = set(raw['train'])
    valid_set = set(raw['valid'])
    directed = {}
    overlaps = {}
    for split, rows in raw.items():
        x = np.array([[ei[h],ri[r],ei[t],ti[date]] for h,r,t,date in rows], dtype=np.int64)
        inv = x[:,[2,1,0,3]].copy()
        inv[:,1] += nr
        directed[split] = np.concatenate([x, inv])
        if split == 'test':
            unique = np.array([row not in train_set and row not in valid_set for row in rows])
            overlaps['test_exact_seen_in_train'] = sum(row in train_set for row in rows)
            overlaps['test_exact_seen_in_valid'] = sum(row in valid_set for row in rows)
            directed['test_unseen_fact'] = np.concatenate([unique, unique])
    truth = defaultdict(set)
    train_truth = defaultdict(set)
    for split in ('train','valid','test'):
        for h,r,t,date in directed[split]:
            truth[(int(h),int(r),int(date))].add(int(t))
            if split == 'train':
                train_truth[(int(h),int(r),int(date))].add(int(t))
    # Sparse filter coordinates align with each test row; unchanged for its fixed date.
    filter_rows, filter_cols = [], []
    for i, (h,r,t,date) in enumerate(directed['test']):
        for c in sorted(truth[(int(h),int(r),int(date))]):
            filter_rows.append(i);filter_cols.append(c)
    tdates_train = set(directed['train'][:,3].tolist())
    time_seen = np.array([i in tdates_train for i in range(len(dates))])
    test = directed['test']
    test_year = dyear[test[:,3]]
    eligible = ((eyear[test[:,0]] <= test_year) & (eyear[test[:,2]] <= test_year)
                & (ryear[test[:,1] % nr] <= test_year) & time_seen[test[:,3]]
                & directed['test_unseen_fact'])
    annual = []
    years = sorted(set(dyear.tolist()))
    for year in years:
        ids = np.flatnonzero((test_year==year) & eligible)
        oldn = int((eyear < year).sum())
        annual.append({'year':year,'train_facts':sum(int(row[3][:4])==year for row in raw['train']),
                       'valid_facts':sum(int(row[3][:4])==year for row in raw['valid']),
                       'test_facts':sum(int(row[3][:4])==year for row in raw['test']),
                       'seen_train_entities':int((eyear<=year).sum()),
                       'new_train_entities':int((eyear==year).sum()),
                       'eligible_test_directional_queries':len(ids),
                       'excluded_test_directional_queries':int((test_year==year).sum())-len(ids),
                       'eligible_new_entity_queries':int(((test[ids,0]>=oldn)|(test[ids,2]>=oldn)).sum())})
    (ROOT/'prepared').mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'prepared/data.npz', **directed, eyear=eyear,ryear=ryear,dyear=dyear,
                        test_year=test_year,eligible=eligible,filter_rows=np.array(filter_rows),
                        filter_cols=np.array(filter_cols))
    write_json(ROOT/'prepared/mapping.json', {'entities':entities,'relations':relations,'dates':dates})
    # Store TRAIN truth as JSON solely for optional TRAIN-only CE filtering; no heldout labels in optimization.
    write_json(ROOT/'prepared/train-truth.json', {'|'.join(map(str,k)):sorted(v) for k,v in train_truth.items()})
    stats={'dataset':'ICEWS05-15','entities':len(entities),'relations':nr,'dates':len(dates),
           'years':years,'split_facts':{k:len(v) for k,v in raw.items()},'annual':annual,
           'overlaps':overlaps,'primary_eligible_directional_queries':int(eligible.sum()),
           'filter':'same (query_entity, directed_relation, exact_date), published train+valid+test truths',
           'future_heldout_only_entities':int((eyear==9999).sum()),'prepare_seconds':time.perf_counter()-start}
    write_json(ROOT/'data-statistics.json',stats)
    print(json.dumps(stats, ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
