"""Read original coded-event context for named sampled cases, not model scoring."""
import json
import time
from pathlib import Path
from raw_icews79 import raw_lines

ROOT=Path(__file__).resolve().parent
sample=json.loads((ROOT/'representative-history-sample.json').read_text())
focus={'S04','S10','S39'}
wanted={c['gold_original_event_metadata']['story_id']:c['sample_id'] for c in sample['records'] if c['sample_id'] in focus}
years={c['gold_original_event_metadata']['year'] for c in sample['records'] if c['sample_id'] in focus}
output={'scope':'all coded events in original gold stories for S04/S10/S39; extra same-date Iraq/Yemen country-pair events',
        'gold_stories':{s:[] for s in focus},'iraq_yemen_same_date':[],
        'warning':'coded events are not original article sentences; country-level article support is not exact actor-dictionary equivalence'}
start=time.perf_counter()
for year in sorted(years):
    paths=list((ROOT/'sources').glob(f'events.{year}.*.zip'))
    assert len(paths)==1, paths
    reader=raw_lines(paths[0]);headers=next(reader);p={s:headers.index(s) for s in headers}
    for row in reader:
        story=row[p['Story ID']]
        if story in wanted:
            output['gold_stories'][wanted[story]].append(dict(zip(headers,row)))
        if year==2009 and row[p['Event Date']]=='2009-04-19' and {row[p['Source Country']],row[p['Target Country']]}=={'Iraq','Yemen'}:
            output['iraq_yemen_same_date'].append(dict(zip(headers,row)))
    print('YEAR',year,'read',flush=True)
output['seconds']=time.perf_counter()-start
(ROOT/'sample-coded-event-context.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)
