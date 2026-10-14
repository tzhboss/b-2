"""Score the fixed40 against actual filtered competitors; no training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/temporal_drift76_20261014')
sys.path.insert(0,str(SOURCE))
from train76 import setup
from evaluate76 import load_model

@torch.no_grad()
def main():
 outpath=ROOT/'fixed40-filtered-competitors.json'
 if outpath.exists():
  print('EXISTING_OUTPUT_NO_RESCORING',str(outpath));return
 setup();start=time.perf_counter()
 sample=json.loads((ROOT/'representative-history-sample.json').read_text(encoding='utf-8'))
 datafile=np.load(SOURCE/'prepared/data.npz')
 data={k:datafile[k] for k in datafile.files}
 stats=json.loads((SOURCE/'data-statistics.json').read_text(encoding='utf-8'))
 mapping=json.loads((SOURCE/'prepared/mapping.json').read_text(encoding='utf-8'))
 nr=len(mapping['relations']);bounds=np.searchsorted(data['filter_rows'],np.arange(len(data['test'])+1))
 ids=np.array([c['test_directional_index'] for c in sample['records']]);x=torch.as_tensor(data['test'][ids],device='cuda')
 n=int((data['eyear']<=2015).sum());model,_=load_model('TComplEx','TComplEx_finetune',2015,stats)
 scores=(model.query(x)@model.E[:n].T).cpu().numpy();del model
 records=[];cand_ids=[]
 for i,c in enumerate(sample['records']):
  idx=c['test_directional_index'];h,r,g,d=map(int,data['test'][idx]);gold=float(scores[i,g])
  truth=data['filter_cols'][bounds[idx]:bounds[idx+1]];masked=scores[i].copy()
  masked[truth[truth<n]]=-np.inf;masked[g]=-np.inf
  rank=1+int((masked>gold).sum())+.5*int((masked==gold).sum())
  assert rank==c['final_gold_rank'],(c['sample_id'],rank,c['final_gold_rank'])
  top=np.argsort(-masked,kind='stable')[:5];cand_ids.append(top)
  def fact(a):
   name=mapping['entities'][int(a)];qname=mapping['entities'][h];rn=mapping['relations'][r%nr];day=mapping['dates'][d]
   return [qname,rn,name,day] if r<nr else [name,rn,qname,day]
  records.append({'sample_id':c['sample_id'],'test_directional_index':idx,'gold_fact':c['gold_fact'],
   'query_year':int(data['test_year'][idx]),'original_filtered_gold_rank':c['original_gold_rank'],
   'final_filtered_gold_rank':rank,'final_gold_score':gold,'unfiltered_top1_fact':c['after_top1_fact'],
   'filtered_strictly_above_gold_n':int((masked>gold).sum()),'filtered_tied_with_gold_n':int((masked==gold).sum()),
   'top_filtered_competitors':[{'entity_id':int(a),'fact':fact(a),'final_score':float(masked[a]),
    'final_score_minus_gold':float(masked[a]-gold),'train_first_year':int(data['eyear'][a]),
    'above_gold':bool(masked[a]>gold),'semantic_truth':'unresolved_pending_external_evidence'} for a in top]})
 for year in sorted({r['query_year'] for r in records}):
  slots=[i for i,r in enumerate(records) if r['query_year']==year]
  m,_=load_model('TComplEx','TComplEx_finetune',year,stats)
  active=int((data['eyear']<=year).sum())
  bs=(m.query(x[slots])@m.E[:active].T).cpu().numpy();del m
  for row,i in enumerate(slots):
   idx=records[i]['test_directional_index'];gold_id=int(data['test'][idx,2]);gscore=float(bs[row,gold_id])
   masked=bs[row].copy();truth=data['filter_cols'][bounds[idx]:bounds[idx+1]]
   masked[truth[truth<active]]=-np.inf;masked[gold_id]=-np.inf
   rank=1+int((masked>gscore).sum())+.5*int((masked==gscore).sum())
   assert rank==records[i]['original_filtered_gold_rank'],(records[i]['sample_id'],rank,records[i]['original_filtered_gold_rank'])
   records[i]['original_gold_score']=gscore
   for a,cand in zip(cand_ids[i],records[i]['top_filtered_competitors']):
    existed=a<active;cand['originally_in_candidate_space']=bool(existed)
    cand['original_score_minus_gold']=float(bs[row,a]-gscore) if existed else None
    cand['crossed_from_not_above_gold']=bool(cand['above_gold'] and (not existed or bs[row,a]<=gscore))
  print('ORIGIN_SCORED',year,len(slots),flush=True)
 out={'population':'same fixed40 random worsened queries, no replacement','sample_n':40,
  'source_model_experiment':str(SOURCE),'scope':'one diagnostic extraction of original-year and2015 filtered top5 competitors; no training or model mutation',
  'cached_rank_checks':'all40 original and final filtered ranks match original archived ranks exactly',
  'train_first_year_note':'First seen in benchmark TRAIN is not real entity creation date.',
  'seconds':time.perf_counter()-start,'records':records}
 outpath.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 for r in records:
  c=r['top_filtered_competitors'][0]
  print(r['sample_id'],' / '.join(c['fact']),'crossed',c['crossed_from_not_above_gold'],'newcand',not c['originally_in_candidate_space'],flush=True)
 print('COMPLETE',out['seconds'],flush=True)
if __name__=='__main__':main()
