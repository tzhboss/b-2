"""Inspect actual interval evidence, keeping missing endpoints unknown."""
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data/src_data/wikidata'


def parse_year(value):
    match=re.match(r'([+-]\d+)-',value)
    return int(match.group(1)) if match else None


def main():
    start=time.perf_counter()
    predicates={}
    for row in (RAW/'predicates').read_text().splitlines():
        p=row.split('\t');predicates[p[0]]={'property':'P'+p[1],'label':p[2]}
    entities={}
    for row in (RAW/'entities').read_text().splitlines():
        p=row.split('\t');entities[p[0]]={'qid':'Q'+p[1],'label':p[2] if len(p)>2 else ''}
    totals=Counter();relation_counts=defaultdict(Counter);scoped=[]
    for split in ('train','valid','test'):
        for line_number,row in enumerate((RAW/split).open(),1):
            p=row.rstrip('\n').split('\t')
            assert len(p)==5
            h,r,t,b,e=p;by,ey=parse_year(b),parse_year(e)
            state=('both_known' if by is not None and ey is not None else
                   'begin_only' if by is not None else 'end_only' if ey is not None else 'neither_known')
            totals[split]+=1;totals[state]+=1;relation_counts[r][state]+=1
            if by is not None and ey is not None and ey<by:
                totals['reversed_interval']+=1
            prop=predicates[r]['property']
            if prop in ('P6','P35','P1308') and (ey is None or ey>=2005) and (by is None or by<=2015):
                scoped.append({'split':split,'line':line_number,'h':h,'r':r,'t':t,
                               'begin':b,'end':e,'begin_year':by,'end_year':ey,'state':state,
                               'subject':entities[h],'relation':predicates[r],'object':entities[t]})
        print('SCANNED',split,totals[split],flush=True)
    target_counts=Counter((x['relation']['property'],x['state']) for x in scoped)
    summary={'source':'TKBC original Wikidata dump, not silently treated as TIE WIKI benchmark',
             'scope':'Calendar window 2005-2015 intersects record or missing endpoint; missing does NOT prove validity',
             'all_raw_interval_counts':dict(totals),
             'target_counts':{'|'.join(k):v for k,v in target_counts.items()},
             'target_records':len(scoped),'seconds':time.perf_counter()-start,
             'relations':{predicates[k]['property']:{**predicates[k],**dict(v)} for k,v in relation_counts.items()}}
    (ROOT/'prepared').mkdir(exist_ok=True)
    (ROOT/'interval-statistics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'prepared/leadership-records.json').write_text(json.dumps(scoped,ensure_ascii=False,indent=2),encoding='utf-8')
    named_subjects={'Q30','Q145','Q142','Q183','Q408','Q16','Q668','Q148'}
    selected=[r for r in scoped if r['subject']['qid'] in named_subjects and r['state']=='both_known']
    (ROOT/'prepared/named-country-records.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='relations'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
