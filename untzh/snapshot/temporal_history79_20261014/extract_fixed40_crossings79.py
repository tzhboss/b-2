"""Materialize all new gold outrankers in fixed40, to support temporal checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
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
 target=ROOT/'fixed40-new-outrankers.json'
 if target.exists(): print('EXISTING_OUTPUT_NO_RESCORING');return
 setup();started=time.perf_counter()
 sample=json.loads((ROOT/'representative-history-sample.json').read_text(encoding='utf-8'))
 with np.load(SOURCE/'prepared/data.npz') as z:data={k:z[k] for k in z.files}
 stats=json.loads((SOURCE/'data-statistics.json').read_text(encoding='utf-8'))
 mapping=json.loads((SOURCE/'prepared/mapping.json').read_text(encoding='utf-8'))
 ids=np.array([c['test_directional_index'] for c in sample['records']]);x=torch.as_tensor(data['test'][ids],device='cuda')
 active=int((data['eyear']<=2015).sum());nr=len(mapping['relations'])
 bounds=np.searchsorted(data['filter_rows'],np.arange(len(data['test'])+1))
 final_model,_=load_model('TComplEx','TComplEx_finetune',2015,stats)
 after=(final_model.query(x)@final_model.E[:active].T).cpu().numpy();del final_model
 before=np.full(after.shape,np.nan,dtype=after.dtype)
 ns=np.zeros(40,dtype=int)
 for year in sorted(set(data['test_year'][ids])):
  slots=np.flatnonzero(data['test_year'][ids]==year);n=int((data['eyear']<=year).sum());ns[slots]=n
  model,_=load_model('TComplEx','TComplEx_finetune',int(year),stats)
  before[slots,:n]=(model.query(x[slots])@model.E[:n].T).cpu().numpy();del model
 np.savez_compressed(ROOT/'fixed40-score-cache.npz',ids=ids,before=before,after=after,origin_active_n=ns)
 out=[]
 for i,c in enumerate(sample['records']):
  idx=int(ids[i]);h,r,g,d=map(int,data['test'][idx]);ags=float(after[i,g]);bgs=float(before[i,g]);n=int(ns[i])
  truth=data['filter_cols'][bounds[idx]:bounds[idx+1]];valid=np.ones(active,dtype=bool);valid[truth[truth<active]]=False;valid[g]=False
  a=after[i]-ags;b=before[i]-bgs
  ar=1+int((valid&(a>0)).sum())+.5*int((valid&(a==0)).sum())
  br=1+int((valid[:n]&(b[:n]>0)).sum())+.5*int((valid[:n]&(b[:n]==0)).sum())
  assert ar==c['final_gold_rank'] and br==c['original_gold_rank']
  crossed=valid&(a>0)&((np.arange(active)>=n)|(b<=0))
  fallen=valid&(np.arange(active)<n)&(b>0)&(a<=0)
  cs=np.flatnonzero(crossed);cs=cs[np.argsort(-a[cs],kind='stable')]
  def fact(k):
   ent=mapping['entities'][int(k)];qent=mapping['entities'][h];rel=mapping['relations'][r%nr];day=mapping['dates'][d]
   return [qent,rel,ent,day] if r<nr else [ent,rel,qent,day]
  out.append({'sample_id':c['sample_id'],'gold_fact':c['gold_fact'],'original_rank':br,'final_rank':ar,
   'newly_strictly_above_gold_n':len(cs),'ceased_strictly_above_gold_n':int(fallen.sum()),
   'gold_ties_before':int((valid[:n]&(b[:n]==0)).sum()),'gold_ties_after':int((valid&(a==0)).sum()),
   'new_outrankers':[{'entity_id':int(k),'fact':fact(k),'after_gap':float(a[k]),
    'before_gap':float(b[k]) if k<n else None,'was_new_candidate':bool(k>=n),'train_first_year':int(data['eyear'][k])} for k in cs]})
 result={'sample_n':40,'scope':'All newly strictly outranking candidates on same fixed40; not reality truth labels. Float score matrices retained for later source checks without further rescoring.',
  'rank_checks':'All40 before/after cached ranks agree exactly','records':out,'seconds':time.perf_counter()-started}
 target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 known=['Aam Aadmi Party','Yasser Arafat','Benazir Bhutto','Saddam Hussein','Osama bin Laden','Moammar Gadhafi','Muammar Gaddafi','Kim Jong Il','Ariel Sharon','Islamic State','Al Nusra','Syrian National Coalition']
 for rec in out:
  hits=[k for k in rec['new_outrankers'] if any(n.lower() in ' / '.join(k['fact']).lower() for n in known)]
  if hits:print(rec['sample_id'],json.dumps(hits,ensure_ascii=False),flush=True)
 print('COMPLETE',result['seconds'],'crossings',sum(r['newly_strictly_above_gold_n'] for r in out),flush=True)
if __name__=='__main__':main()
