"""Link named dated witnesses to the published original event metadata."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'history-interpretation.json').read_text())
records={}
for p in ROOT.glob('raw-matched-tuples-*.json'):
    records.update({tuple(r['tuple']):r['record'] for r in json.loads(p.read_text())['matches']})
output=[]
for c in report['arms']['TComplEx_finetune']['temporal_incompatibility_witnesses']:
    def fact(answer):
        if c['direction']=='tail':return(c['query_entity'],c['relation'],answer,c['fixed_date'])
        return(answer,c['relation'],c['query_entity'],c['fixed_date'])
    output.append({**c,'gold_raw_record':records.get(fact(c['gold'])),
                   'original_top1_raw_record':records.get(fact(c['before_top1'])),
                   'final_top1_raw_record':records.get(fact(c['after_top1'])),
                   'record_semantics':'original automatic coded event plus internal Story ID; article text not published in event file'})
(ROOT/'witness-news-provenance.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
print(json.dumps(output,ensure_ascii=False,indent=2))
