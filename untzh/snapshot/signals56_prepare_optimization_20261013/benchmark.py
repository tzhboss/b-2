"""Compare ordered cohorts on real 056 TRAIN queries, then reproduce preparation."""
import cProfile
import io
import json
import pstats
import time
from pathlib import Path
import numpy as np
import torch
from reference_adapter import PrefixStructure as Original
from fast_adapter import PrefixStructure as Optimized
import optimized_experiment as experiment

ROOT = Path(__file__).resolve().parent
PRIOR = Path('/data/lc/untzh/signals56_20261013')


def main():
    torch.set_num_threads(2)
    d = experiment.load(experiment.SRC / 'entity_diagnostic42_20261012/artifacts/data/train.npz')
    old, new = d['old'], d['new']
    facts = {(int(h), int(r)//2, int(t)) if r%2 == 0 else (int(t), int(r)//2, int(h))
             for h, r, t in np.concatenate([old, new])}
    original = Original(facts, range(7505), range(7505,11258))
    optimized = Optimized(facts, range(7505), range(7505,11258))
    seq = experiment.load(PRIOR / 'protection-sequence.npz')['old_ids'].reshape(-1)
    # Include both roles and positions throughout the real protection tape.
    positions = np.unique(np.concatenate([np.arange(64), np.linspace(0,len(seq)-1,256,dtype=int)]))
    rows = old[seq[positions]]
    kwargs = dict(seed=550017, draws=4, scale_count=256)
    start = time.monotonic()
    expected = [original.cohorts(experiment.key(row), int(row[2]), **kwargs) for row in rows]
    baseline_seconds = time.monotonic()-start
    start = time.monotonic()
    actual = [optimized.cohorts(experiment.key(row), int(row[2]), **kwargs) for row in rows]
    optimized_seconds = time.monotonic()-start
    for i,(a,b) in enumerate(zip(expected,actual)):
        if a != b:
            raise AssertionError(('ordered_cohort_mismatch',i,[k for k in a if a[k] != b[k]]))
    prof = cProfile.Profile()
    prof.enable()
    for row in rows[:32]:
        original.cohorts(experiment.key(row), int(row[2]), **kwargs)
    prof.disable()
    output = io.StringIO()
    pstats.Stats(prof,stream=output).strip_dirs().sort_stats('cumulative').print_stats(25)
    (ROOT / 'original-profile.txt').write_text(output.getvalue())
    report = {'real_cohorts':len(rows),'all_fields_ordered_equal':True,
              'baseline_seconds':baseline_seconds,'optimized_seconds':optimized_seconds,
              'cohort_speedup':baseline_seconds/optimized_seconds}
    experiment.write(ROOT/'cohort-comparison.json',report)
    print('REAL_COHORT_COMPARISON',json.dumps(report),flush=True)
    initial = torch.load(experiment.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    tape = experiment.load(experiment.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    prepared = experiment.prepare(initial,old,new,tape)
    count = 0
    max_difference = 0.0
    for j in range(740):
        a = experiment.load(PRIOR/'cache'/f'{j:04d}.npz')
        b = experiment.load(ROOT/'cache'/f'{j:04d}.npz')
        if a.keys() != b.keys():
            raise AssertionError(('cache_fields',j))
        for key in a:
            if not np.array_equal(a[key],b[key]):
                raise AssertionError(('cache_mismatch',j,key))
            count += 1
    old_prepare = json.loads((PRIOR/'prepare.json').read_text())
    report.update(full_preparation=prepared,original_full_preparation_seconds=old_prepare['seconds'],
                  preparation_speedup=old_prepare['seconds']/prepared['seconds'],
                  all_740_cache_batches_exactly_equal=True,cache_arrays_compared=count,
                  scope='Preparation only, no training or VALID/TEST reads; different-time timing comparison')
    experiment.write(ROOT/'comparison.json',report)
    experiment.write(ROOT/'status.json',{'stage':'complete','preparation_seconds':prepared['seconds']})
    print('COMPLETE',json.dumps(report),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        experiment.write(ROOT/'status.json',{'stage':'failed','error':repr(exc)})
        raise
