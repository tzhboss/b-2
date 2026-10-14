"""Read existing raw2005 post-death coding metadata, without model scoring."""
import json,time
from pathlib import Path
from raw_icews79 import raw_lines
ROOT=Path(__file__).resolve().parent
target=ROOT/'pope-identity-context-round18.json'
if target.exists():
    print('EXISTING_OUTPUT_NO_RESCAN')
    raise SystemExit(0)
paths=[p for p in (ROOT/'sources').glob('events.2005.*') if p.suffix in {'.zip','.tab'}]
assert len(paths)==1,paths
started=time.perf_counter();reader=raw_lines(paths[0]);headers=next(reader)
si,ti,di=[headers.index(k) for k in ['Source Name','Target Name','Event Date']]
records=[]
for row in reader:
    if 'John Paul II' in {row[si],row[ti]} and row[di]>'2005-04-02':
        records.append(dict(zip(headers,row)))
        if len(records)>=5:break
out={'records':records,'seconds':time.perf_counter()-started,
     'scope':'First five available raw2005 post-death rows; metadata and identity context only, not source article bodies or exhaustive history.',
     'death_date_source':'https://www.vatican.va/content/john-paul-ii/en/angelus/2005/documents/hf_jp-ii_reg_20050403_divina-misericordia.html',
     'training':False,'neural_rescoring':False}
target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'records_n':len(records),'seconds':out['seconds'],'dates':[r['Event Date'] for r in records]}))
