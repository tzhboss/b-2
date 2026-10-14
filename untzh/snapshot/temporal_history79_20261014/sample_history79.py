"""Draw a fixed probability sample from existing ranks; no model evaluation."""
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/lc/untzh/temporal_drift76_20261014')

def main():
    started = time.perf_counter()
    mapping = json.loads((SOURCE/'prepared/mapping.json').read_text())
    queries = np.load(SOURCE/'prepared/data.npz')['test']
    with np.load(ROOT/'icews/TComplEx_finetune-history.npz') as z:
        h = {k:z[k] for k in z.files}
    population = np.flatnonzero(h['final_rank'] > h['original_rank'])
    assert len(population) == 42851
    seed, sample_n = 20261014, 40
    chosen = np.random.default_rng(seed).choice(population, sample_n, replace=False)
    provenance = {}
    for path in sorted(ROOT.glob('raw-matched-tuples-*.json')):
        for row in json.loads(path.read_text())['matches']:
            provenance[tuple(row['tuple'])] = row['record']
    nr = len(mapping['relations'])
    records = []
    for number, slot in enumerate(chosen, 1):
        idx = int(h['ids'][slot])
        entity, relation, gold, date = map(int, queries[idx])
        direction = 'tail' if relation < nr else 'head'
        qname = mapping['entities'][entity]
        rname = mapping['relations'][relation % nr]
        fixed_date = mapping['dates'][date]
        def fact(answer):
            name = mapping['entities'][int(answer)]
            return [qname, rname, name, fixed_date] if direction == 'tail' else [name, rname, qname, fixed_date]
        gold_fact = fact(gold)
        before_fact = fact(h['prediction_before'][slot])
        after_fact = fact(h['prediction_after'][slot])
        assert tuple(gold_fact) in provenance
        records.append({
            'sample_id':f'S{number:02d}', 'test_directional_index':idx,
            'direction':direction, 'gold_fact':gold_fact,
            'before_top1_fact':before_fact, 'after_top1_fact':after_fact,
            'original_gold_rank':float(h['original_rank'][slot]),
            'final_gold_rank':float(h['final_rank'][slot]),
            'before_recorded':bool(h['before_top1_recorded_true'][slot]),
            'after_recorded':bool(h['after_top1_recorded_true'][slot]),
            'gold_original_event_metadata':provenance[tuple(gold_fact)],
            'before_original_event_metadata':provenance.get(tuple(before_fact)),
            'after_original_event_metadata':provenance.get(tuple(after_fact)),
        })
    report = {
        'arm':'TComplEx_finetune', 'population':'all 2005-2014 fixed-date directional queries with worsened filtered gold rank',
        'population_n':len(population), 'sample_n':sample_n, 'seed':seed,
        'design':'simple random sample without replacement from cached directional-query population; no source-based exclusion or replacement',
        'inclusion_probability':sample_n/len(population),
        'source_model_experiment':str(SOURCE),
        'no_retraining_or_rescoring':True,
        'labels':'pending; record evidence-supported final truth, explicit contradiction, or unknown independently of rank mechanism',
        'query_dependence':'directional queries can share original facts, actors, or articles; sample counts are not independent event counts',
        'extraction_seconds':time.perf_counter()-started,
        'records':records,
    }
    (ROOT/'representative-history-sample.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for c in records:
        meta=c['gold_original_event_metadata']
        print(c['sample_id'],c['after_top1_fact'],f"rank {c['original_gold_rank']:g}->{c['final_gold_rank']:g}",
              'before/after recorded',c['before_recorded'],c['after_recorded'],
              'publisher',meta.get('publisher'),'story',meta.get('story_id'),flush=True)

if __name__=='__main__':
    main()
