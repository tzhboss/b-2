"""Materialize original coded-story rows for all fixed sampled queries."""
import json
import time
from pathlib import Path
from raw_icews79 import raw_lines

ROOT=Path(__file__).resolve().parent
sample=json.loads((ROOT/'representative-history-sample.json').read_text())
outpath=ROOT/'all-sample-coded-context.json'
prior=json.loads(outpath.read_text()) if outpath.exists() else {'stories':{},'years_scanned':[]}
if not prior['stories']:
    small=json.loads((ROOT/'sample-coded-event-context.json').read_text())
    prior['stories'].update(small['gold_stories'])
by_year={}
for c in sample['records']:
    if c['sample_id'] in prior['stories']:continue
    meta=c['gold_original_event_metadata']
    by_year.setdefault(meta['year'],{}).setdefault(meta['story_id'],[]).append(c['sample_id'])
started=time.perf_counter()
for year,wanted in sorted(by_year.items()):
    pathlist=[p for p in (ROOT/'sources').glob(f'events.{year}.*') if p.suffix in {'.zip','.tab'}]
    assert len(pathlist)==1,pathlist
    reader=raw_lines(pathlist[0]);headers=next(reader);sp=headers.index('Story ID')
    for sample_ids in wanted.values():
        for sample_id in sample_ids:prior['stories'][sample_id]=[]
    for row in reader:
        for sample_id in wanted.get(row[sp],[]):
            prior['stories'][sample_id].append(dict(zip(headers,row)))
    assert all(prior['stories'][sample_id] for sample_ids in wanted.values() for sample_id in sample_ids)
    prior['years_scanned'].append(year)
    prior['current_invocation_seconds']=time.perf_counter()-started
    prior['scope']='Original coded-story context for all fixed40; not article body or adjudicated negative truth.'
    outpath.write_text(json.dumps(prior,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CONTEXT_YEAR',year,'stories',len(prior['stories']),flush=True)
for c in sample['records']:
    meta=c['gold_original_event_metadata']
    row=next(r for r in prior['stories'][c['sample_id']] if r['Event ID']==meta['event_id'])
    print(c['sample_id'],'source sectors:',row['Source Sectors'],'target sectors:',row['Target Sectors'],'rows:',len(prior['stories'][c['sample_id']]),flush=True)
print('COMPLETE',prior['current_invocation_seconds'],flush=True)
