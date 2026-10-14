"""Download official yearly event records and match cached dated predictions.

Full ICEWS positives remain machine-coded observations, not adjudicated reality.
No KGE training or rank rescoring. No artificial process time limits/retries.
"""
import csv
import io
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/temporal_drift76_20261014')

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2))

def download(file):
    path=ROOT/'sources'/file['filename']
    if path.exists():
        if path.stat().st_size!=file['filesize']:
            raise ValueError(f'existing size mismatch {path}')
        return path
    url=f'https://dataverse.harvard.edu/api/access/datafile/{file["id"]}'
    print('DOWNLOAD_START',file['filename'],file['filesize'],flush=True)
    subprocess.run(['curl','-L','-f','-o',str(path.with_suffix(path.suffix+'.part')),url],check=True)
    n=path.with_suffix(path.suffix+'.part').stat().st_size
    if n!=file['filesize']:raise ValueError(f'download size mismatch {path} {n}')
    path.with_suffix(path.suffix+'.part').rename(path)
    print('DOWNLOAD_COMPLETE',file['filename'],n,flush=True)
    return path

def raw_lines(path):
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as z:
            members=[m for m in z.namelist() if m.lower().endswith(('.tab','.tsv'))]
            if len(members)!=1:raise ValueError(members)
            with z.open(members[0]) as f, io.TextIOWrapper(f,encoding='utf-8-sig',newline='') as text:
                yield from csv.reader(text,delimiter='\t')
    else:
        with path.open(encoding='utf-8-sig',newline='') as f:
            yield from csv.reader(f,delimiter='\t')

def dated_tuple(q,mapping,answer):
    h,r,t,d=map(int,q);nr=len(mapping['relations'])
    a=mapping['entities'][h];b=mapping['entities'][int(answer)]
    if r>=nr:a,b=b,a
    return (a,mapping['relations'][r%nr],b,mapping['dates'][d])

def main():
    start=time.perf_counter()
    meta=json.loads((ROOT/'sources/icews-dataverse-metadata.json').read_text())
    files={int(x['dataFile']['filename'].split('.')[1]):x['dataFile']
           for x in meta['data']['latestVersion']['files']
           if x['dataFile']['filename'].startswith('events.')
           and x['dataFile']['filename'].split('.')[1].isdigit()}
    mapping=json.loads((SOURCE/'prepared/mapping.json').read_text())
    with np.load(SOURCE/'prepared/data.npz') as packed:
        data={k:packed[k] for k in ('test','test_year')}
    arms={f.name.removesuffix('-history.npz'):dict(np.load(f)) for f in (ROOT/'icews').glob('*-history.npz')}
    ids=arms['TComplEx_finetune']['ids']
    assert all(np.array_equal(h['ids'],ids) for h in arms.values())
    print('PREPARE_KEYS',len(ids),flush=True)
    gold=[dated_tuple(data['test'][idx],mapping,data['test'][idx,2]) for idx in ids]
    before={arm:[dated_tuple(data['test'][idx],mapping,h['prediction_before'][i]) for i,idx in enumerate(ids)] for arm,h in arms.items()}
    after={arm:[dated_tuple(data['test'][idx],mapping,h['prediction_after'][i]) for i,idx in enumerate(ids)] for arm,h in arms.items()}
    wanted=set(gold)
    for rows in list(before.values())+list(after.values()):wanted.update(rows)
    # Exact strings only. No approximate aliases, relation merging or date windows.
    wanted_by_year={year:{t for t in wanted if t[3].startswith(str(year))} for year in range(2005,2015)}
    matched={};summary={'source_doi':'10.7910/DVN/28075','version':37,
                        'matching':'exact actor names, Event Text and date; no fuzzy matches',
                        'source_semantics':'automatic BBN ACCENT event coding; a match is an additional recorded positive, not independently validated truth',
                        'years':{},'arms':{}}
    for year in range(2005,2015):
        save(ROOT/'raw-icews-status.json',{'stage':'download_and_match','year':year,'pid':os.getpid()})
        cache=ROOT/f'raw-matched-tuples-{year}.json'
        if cache.exists():
            prior=json.loads(cache.read_text())
            summary['years'][str(year)]=prior['year_summary']
            matched.update({tuple(x['tuple']):x['record'] for x in prior['matches']})
            print('YEAR_CACHE_REUSED',year,len(prior['matches']),flush=True)
            continue
        path=download(files[year]);reader=raw_lines(path);header=next(reader)
        print('HEADER',year,json.dumps(header),flush=True)
        positions={k:header.index(k) for k in ['Source Name','Event Text','Target Name','Event Date','Event ID','Story ID','Sentence Number','Publisher']}
        count=0;matched_n=0;want=wanted_by_year[year]
        for row in reader:
            count+=1
            date=row[positions['Event Date']]
            if len(date)==8 and date.isdigit():date=date[:4]+'-'+date[4:6]+'-'+date[6:]
            key=(row[positions['Source Name']],row[positions['Event Text']],row[positions['Target Name']],date)
            if key in want:
                matched_n+=1
                if key not in matched:matched[key]={'event_id':row[positions['Event ID']], 'year':year,'row':count+1,
                                                  'story_id':row[positions['Story ID']],
                                                  'sentence_number':row[positions['Sentence Number']],
                                                  'publisher':row[positions['Publisher']]}
            if count%200000==0:print('MATCH_PROGRESS',year,count,matched_n,flush=True)
        summary['years'][str(year)]={'file_id':files[year]['id'],'filename':files[year]['filename'],
                                     'bytes':files[year]['filesize'],'columns':header,'records_n':count,
                                     'matched_records_n':matched_n,'wanted_unique_tuples_n':len(want),
                                     'matched_unique_tuples_n':sum(k in matched for k in want)}
        print('YEAR_COMPLETE',year,json.dumps(summary['years'][str(year)]),flush=True)
        save(cache,{'year_summary':summary['years'][str(year)],
                    'matches':[{'tuple':list(k),'record':v} for k,v in matched.items() if v['year']==year]})
        save(ROOT/'raw-icews-summary.json',summary)
    gmatch=np.array([t in matched for t in gold])
    summary['gold_exact_match_n']=int(gmatch.sum());summary['gold_n']=len(gmatch)
    summary['gold_exact_match_fraction']=float(gmatch.mean())
    masks_common={'all_historical':np.ones(len(ids),dtype=bool),'anchor2005':data['test_year'][ids]==2005}
    for arm,h in arms.items():
        bf=np.array([t in matched for t in before[arm]])
        af=np.array([t in matched for t in after[arm]])
        masks={**masks_common,'all_rank_worsening':h['final_rank']>h['original_rank'],
               'anchor2005_rank_worsening':(data['test_year'][ids]==2005)&(h['final_rank']>h['original_rank'])}
        groups={}
        for group,mask in masks.items():
            benchmark_new_error=mask&h['before_top1_recorded_true']&~h['after_top1_recorded_true']
            transitions={'full_recorded_true_to_full_recorded_true':int((mask&bf&af).sum()),
                         'full_recorded_true_to_unrecorded':int((mask&bf&~af).sum()),
                         'unrecorded_to_full_recorded_true':int((mask&~bf&af).sum()),
                         'unrecorded_to_unrecorded':int((mask&~bf&~af).sum())}
            assert sum(transitions.values())==int(mask.sum())
            groups[group]={'n':int(mask.sum()),'gold_full_recorded_n':int((mask&gmatch).sum()),
                           'before_full_recorded_true_n':int((mask&bf).sum()),
                           'after_full_recorded_true_n':int((mask&af).sum()),
                           'transitions':transitions,
                           'benchmark_new_error_n':int(benchmark_new_error.sum()),
                           'benchmark_new_error_but_full_recorded_n':int((benchmark_new_error&af).sum()),
                           'benchmark_recorded_before_not_in_current_raw_n':int((mask&h['before_top1_recorded_true']&~bf).sum()),
                           'benchmark_recorded_after_not_in_current_raw_n':int((mask&h['after_top1_recorded_true']&~af).sum())}
            combined_bf=bf|h['before_top1_recorded_true'];combined_af=af|h['after_top1_recorded_true']
            groups[group]['union_recorded_transitions']={
                'recorded_to_recorded':int((mask&combined_bf&combined_af).sum()),
                'recorded_to_unrecorded':int((mask&combined_bf&~combined_af).sum()),
                'unrecorded_to_recorded':int((mask&~combined_bf&combined_af).sum()),
                'unrecorded_to_unrecorded':int((mask&~combined_bf&~combined_af).sum())}
            groups[group]['after_union_recorded_true_n']=int((mask&combined_af).sum())
        summary['arms'][arm]=groups
        np.savez_compressed(ROOT/f'icews/{arm}-raw-matches.npz',ids=ids,gold_full_recorded=gmatch,
                            before_full_recorded=bf,after_full_recorded=af)
    # Export formerly benchmark-unrecorded predictions with exact raw-record links.
    arm='TComplEx_finetune';h=arms[arm]
    mask=(h['final_rank']>h['original_rank'])&h['before_top1_recorded_true']&~h['after_top1_recorded_true']
    rescued=[]
    for i in np.flatnonzero(mask):
        if after[arm][i] in matched:
            rescued.append({'test_directional_index':int(ids[i]),'query_tuple':after[arm][i],
                            'original_rank':float(h['original_rank'][i]),'final_rank':float(h['final_rank'][i]),
                            'raw_record':matched[after[arm][i]],
                            'judgment':'same historical dated prediction additionally recorded in full ICEWS; not verified real-world truth'})
    save(ROOT/'full-icews-rescued-predictions.json',rescued)
    summary['seconds']=time.perf_counter()-start
    save(ROOT/'raw-icews-summary.json',summary)
    save(ROOT/'raw-icews-status.json',{'stage':'complete','pid':os.getpid(),'seconds':summary['seconds']})
    print('ALL_COMPLETE',summary['seconds'],summary['gold_exact_match_fraction'],json.dumps(summary['arms']['TComplEx_finetune']),flush=True)

if __name__=='__main__':main()
