from __future__ import annotations
import glob,re,json,yaml
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def speaker_id(x):
 m=re.match(r'(Ses\d\d)[FM]_.+_([FM])\d+\.wav$',str(x));
 if not m: raise ValueError(x)
 return m.group(1)+'_'+m.group(2)
def speaker_folds(speakers,n,seed):
 s=np.array(sorted(map(str,speakers)),dtype=object); np.random.default_rng(seed).shuffle(s); return {sp:i%n for i,sp in enumerate(s)}
def weights(df):
 c=df.speaker_id.astype(str).value_counts(); w=df.speaker_id.astype(str).map(lambda s:1/c[s]).to_numpy(float); return w/w.mean()
def fit(xtr,xte,ytr,w,alpha):
 sc=StandardScaler(); sc.fit(xtr,sample_weight=w); m=Ridge(alpha=alpha); m.fit(sc.transform(xtr),ytr,sample_weight=w); return m.predict(sc.transform(xte))
def ccc_m(m):
 my,mp,my2,mp2,myp=m; vy=max(my2-my*my,0); vp=max(mp2-mp*mp,0); den=vy+vp+(my-mp)**2; return 2*(myp-my*mp)/den if den>0 else np.nan
def moments(df):
 x=df.copy(); x['y2']=x.y_true*x.y_true; x['p2']=x.pred*x.pred; x['yp']=x.y_true*x.pred
 g=x.groupby('speaker_id',sort=True).agg(my=('y_true','mean'),mp=('pred','mean'),my2=('y2','mean'),mp2=('p2','mean'),myp=('yp','mean'))
 o=g[['my','mp','my2','mp2','myp']].to_numpy(float)
 b=np.column_stack([g.my,g.mp,g.my**2,g.mp**2,g.my*g.mp]).astype(float)
 w=np.column_stack([np.zeros(len(g)),np.zeros(len(g)),np.maximum(g.my2-g.my**2,0),np.maximum(g.mp2-g.mp**2,0),g.myp-g.my*g.mp]).astype(float)
 return o,b,w
cfg=yaml.safe_load(Path('configs/experiments/EXP-20260923-03.yaml').read_text())
root=Path(cfg['output_root']); root.mkdir(parents=True,exist_ok=True)
frames=[pd.read_parquet(f,columns=['file','EmoAct','EmoDom']) for f in sorted(glob.glob(cfg['dataset_glob']))]
d=pd.concat(frames,ignore_index=True).dropna().copy(); d['sample_id']=d.file.astype(str); d['speaker_id']=d.file.map(speaker_id); d=d.rename(columns={'EmoAct':'arousal','EmoDom':'dominance'})
ids=[]; blocks=[]
for f in sorted(Path(cfg['embedding_file']).parent.joinpath('embeddings').glob('*.npz')):
 z=np.load(f); ids.extend(z['sample_id'].astype(str)); blocks.append(z['embedding'].astype(np.float32))
emb=np.concatenate(blocks); em=pd.DataFrame({'sample_id':ids,'row':np.arange(len(ids))}); d=d.merge(em,on='sample_id',validate='one_to_one')
if len(d)!=10039: raise RuntimeError(f'coverage {len(d)}')
X=emb[d.row.to_numpy()]; speakers=sorted(d.speaker_id.unique()); fmap=speaker_folds(speakers,int(cfg['outer_folds']),int(cfg['split_seed'])); folds=d.speaker_id.map(fmap).to_numpy(); Y=d[['arousal','dominance']].to_numpy(float); pred=np.empty_like(Y)
for fold in range(int(cfg['outer_folds'])):
 tr=folds!=fold; te=folds==fold; pred[te]=fit(X[tr],X[te],Y[tr],weights(d.loc[tr]),float(cfg['ridge_alpha']))
rows=[]
for j,t in enumerate(['arousal','dominance']):
 q=pd.DataFrame({'sample_id':d.sample_id,'speaker_id':d.speaker_id,'target':t,'y_true':Y[:,j],'pred':pred[:,j],'fold':folds}); o,b,w=moments(q)
 rng=np.random.default_rng(int(cfg['bootstrap_seed'])+j); ns=len(speakers); counts=rng.multinomial(ns,np.full(ns,1/ns),size=int(cfg['bootstrap_reps'])).astype(float)
 for name,m in [('overall',o),('between',b),('within',w)]:
  pt=ccc_m(m.mean(0)); bm=(counts@m)/ns; vals=np.array([ccc_m(x) for x in bm]); lo,hi=np.nanquantile(vals,[.025,.975]); rows.append({'target':t,'metric':name,'ccc':pt,'ci95_low':lo,'ci95_high':hi})
 q.to_parquet(Path(cfg['output_root'])/f'oof_{t}.parquet',index=False)
root=Path(cfg['output_root']); root.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(root/'component_metrics.csv',index=False); pd.DataFrame({'sample_id':d.sample_id,'speaker_id':d.speaker_id,'fold':folds}).to_csv(root/'fold_assignments.csv',index=False); (root/'run_metadata.json').write_text(json.dumps({'experiment_id':cfg['experiment_id'],'rows':len(d),'speakers':len(speakers),'split_seed':cfg['split_seed'],'ridge_alpha':cfg['ridge_alpha'],'bootstrap_reps':cfg['bootstrap_reps'],'bootstrap_unit':'speaker'},indent=2)+'\n'); print(pd.DataFrame(rows).to_string(index=False))
