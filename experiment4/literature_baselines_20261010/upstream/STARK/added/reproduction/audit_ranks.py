"""Recompute native test summaries from saved query ranks and validate query coverage."""
import argparse,hashlib,json,pathlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run-name',required=True);a=p.parse_args()
base=pathlib.Path(__file__).resolve().parent;c=json.loads((base/'config.json').read_text());repo=(base/c['repo_root']).resolve()
assert pathlib.Path(a.run_name).name==a.run_name and a.run_name not in {'.','..'}
run=base/c['output_root']/a.run_name;obs=run/'observations'
entities={};relations={};queries={};n_entities={}
for i in range(5):
    split={}
    for name in ['train','valid','test']:
        triples=[tuple(x.split()) for x in (repo/'data'/c['dataset']/str(i)/(name+'.txt')).read_text().splitlines()];split[name]=triples
        for h,r,t in triples:
            for e in [h,t]:
                if e not in entities:entities[e]=len(entities)
            if r not in relations:
                v=len(relations);relations[r]=v;relations[r+'_inv']=v+1
    mapped=sorted([(entities[h],relations[r],entities[t]) for h,r,t in split['test']],key=lambda t:t[1])
    queries[i]=np.asarray(mapped+[(t,r+1,h) for h,r,t in mapped],dtype=np.int64);n_entities[i]=len(entities)
rows=[]
expected={(i,j,'after') for i in range(5) for j in range(i+1)} | {(i,i,'before') for i in range(1,5)}
files=sorted(obs.glob('train*_test*_*.npz'));assert len(files)==19, len(files)
found=set()
for file in files:
    with np.load(file,allow_pickle=False) as z:
        i=int(z['training_snapshot']);j=int(z['test_snapshot']);phase=str(z['phase']);key=(i,j,phase);assert key in expected and key not in found;found.add(key)
        assert int(z['candidate_count'])==n_entities[j]
        q=np.column_stack([z['head'],z['relation'],z['tail']]);assert np.array_equal(q,queries[j]),file.name
        r=z['rank'];assert np.issubdtype(r.dtype,np.integer) and np.all((r>=1)&(r<=n_entities[j]))
        metrics={'mr':float(r.mean()),'mrr':float((1/r).mean()),**{f'hits{k}':float((r<=k).mean()) for k in [1,3,5,10]}}
        rows.append({'training_snapshot':i,'test_snapshot':j,'phase':phase,'queries':len(r),'raw_float64_metrics':metrics,'native_precision_metrics':{k:round(v,4) for k,v in metrics.items()},'rank_file':str(file.relative_to(run)),'rank_file_sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
assert found==expected
checkpoints=[]
for i in range(5):
    file=obs/f'selected_checkpoint_snapshot{i}.json';record=json.loads(file.read_text());checkpoint=(obs/record['checkpoint']).resolve();assert checkpoint.is_file() and checkpoint.is_relative_to(run.resolve())
    checkpoints.append({**record,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
report=[]
for i in range(5):
    row=[x for x in rows if x['training_snapshot']==i and x['phase']=='after'];total=sum(x['queries'] for x in row)
    report.append({'snapshot':i,**{k:round(sum(x['queries']*x['native_precision_metrics'][k] for x in row)/total,3) for k in ['mrr','hits1','hits3','hits10']}})
lookup={(x['training_snapshot'],x['test_snapshot'],x['phase']):x for x in rows}
bwt=sum(lookup[(4,i,'after')]['native_precision_metrics']['mrr']-lookup[(i,i,'after')]['native_precision_metrics']['mrr'] for i in range(4))/4
fwt=sum(lookup[(i,i,'before')]['native_precision_metrics']['mrr'] for i in range(1,5))/4
output={'kind':'RANK_RECOMPUTATION_PENDING_CONTROLLER_AUDIT','queries_exact_native_test_order':True,'coverage_19_expected_evaluations':True,'native_double_argsort_tie_policy_unchanged':True,'snapshot_report':report,'BWT':bwt,'FWT':fwt,'evaluations':rows,'selected_checkpoints':checkpoints,'limitations':'This independently recomputes metrics from native exported ranks and checks raw-query coverage, not independent checkpoint inference. Float32 batch reduction in native logger can differ near rounding boundaries.'}
(run/'rank_recomputation.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k not in ['evaluations','selected_checkpoints']},indent=2))
