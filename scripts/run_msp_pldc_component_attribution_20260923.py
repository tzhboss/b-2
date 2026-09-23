#!/usr/bin/env python3
import os, json, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
ABS=['pitch_abs','f0var_abs','loud_abs','rate_abs','pause_abs','voiced_abs']
def seed(*x): return int.from_bytes(hashlib.sha256('|'.join(map(str,x)).encode()).digest()[:8],'big')%(2**32)
def ccc(a):
 m=np.asarray(a,float); y,p,y2,p2,yp=[m[...,i] for i in range(5)]; vy=np.maximum(y2-y*y,0); vp=np.maximum(p2-p*p,0); d=vy+vp+(y-p)**2; return np.divide(2*(yp-y*p),d,out=np.full_like(d,np.nan),where=d>0)
def arr(g,k):
 if k=='overall': return g[['my','mp','my2','mp2','myp']].to_numpy(float)
 if k=='between': return np.c_[g.my,g.mp,g.my**2,g.mp**2,g.my*g.mp]
 return np.c_[np.zeros(len(g)),np.zeros(len(g)),g.my2-g.my**2,g.mp2-g.mp**2,g.myp-g.my*g.mp]
def prep(path):
 C=['dataset','speaker_id','arousal_mean_1_7','dominance_mean_1_7','f0_median_hz','f0_iqr_semitone','speech_lufs','phoneme_articulation_rate','pause_ratio','voiced_ratio','pitch_reference_scope','loudness_reference_scope','rate_reference_scope']
 d=pd.read_parquet(path,columns=C); d=d[d.dataset.eq('msp')].dropna().copy(); d=d[(d.speaker_id.astype(str)!='Unknown')&(d.f0_median_hz>0)&(d.phoneme_articulation_rate>0)&d.pitch_reference_scope.isin(['speaker_neutral','speaker_neutral_shrunk'])&d.loudness_reference_scope.isin(['speaker_neutral','speaker_neutral_shrunk'])&d.rate_reference_scope.isin(['speaker_neutral','speaker_neutral_shrunk'])]; n=d.groupby('speaker_id').size(); d=d[d.speaker_id.isin(n[n>10].index)].copy(); d['pitch_abs']=12*np.log2(d.f0_median_hz); d['f0var_abs']=d.f0_iqr_semitone; d['loud_abs']=d.speech_lufs; d['rate_abs']=np.log(d.phoneme_articulation_rate); d['pause_abs']=d.pause_ratio; d['voiced_abs']=d.voiced_ratio; return d.rename(columns={'arousal_mean_1_7':'arousal','dominance_mean_1_7':'dominance'}).reset_index(drop=True)
def folds(S,n,s):
 x=np.array(sorted(S),object); np.random.default_rng(s).shuffle(x); return {v:i%n for i,v in enumerate(x)}
def moments(y,p): return [y.mean(),p.mean(),np.mean(y*y),np.mean(p*p),np.mean(y*p),len(y)]
def main():
 q=argparse.ArgumentParser(); q.add_argument('--config',required=True); c=yaml.safe_load(Path(q.parse_args().config).read_text()); d=prep(os.environ[c['dataset']['source_env']]); seeds=list(map(int,c['seed'])); ks=list(map(int,c['parameters']['top_k_values'])); modes=c['parameters']['shift_modes']; targets=c['parameters']['targets']; nf=int(c['parameters']['outer_folds']); rows=[]; audit=[]; S=sorted(d.speaker_id.astype(str).unique())
 for sd in seeds:
  fm=folds(S,nf,sd); ff=d.speaker_id.astype(str).map(fm).to_numpy()
  for f in range(nf):
   tr=d[ff!=f].copy().reset_index(drop=True); te=d[ff==f].copy().reset_index(drop=True); cnt=tr.speaker_id.astype(str).value_counts(); w=tr.speaker_id.astype(str).map(lambda x:1/cnt[x]).to_numpy(); w/=w.mean(); sc=StandardScaler().fit(tr[ABS],sample_weight=w); X=sc.transform(tr[ABS]); Z=sc.transform(te[ABS]); model=Ridge(alpha=float(c['parameters']['ridge_alpha'])).fit(X,tr[targets],sample_weight=w); P=model.predict(Z); TS=sorted(tr.speaker_id.astype(str).unique()); V=np.vstack([X[tr.speaker_id.astype(str).eq(s)].mean(0) for s in TS]); stats={t:np.array([[tr.loc[tr.speaker_id.astype(str).eq(s),t].mean(),tr.loc[tr.speaker_id.astype(str).eq(s),t].std(ddof=0)] for s in TS]) for t in targets}
   for sp in sorted(te.speaker_id.astype(str).unique()):
    mask=te.speaker_id.astype(str).eq(sp).to_numpy(); v=Z[mask].mean(0); sim=(V@v)/(np.maximum(np.linalg.norm(V,axis=1),1e-12)*max(np.linalg.norm(v),1e-12)); order=np.argsort(-sim); audit.append({'seed':sd,'fold':f,'speaker_id':sp,'nearest':TS[order[0]]})
    for j,t in enumerate(targets):
     y=te.loc[mask,t].to_numpy(float); p=P[mask,j]; rows.append([sd,f,sp,t,'baseline',*moments(y,p)]); pm=p.mean(); ps=p.std(ddof=0)
     for k in ks:
      ix=order[:min(k,len(order))]; mu=stats[t][ix,0].mean(); sig=stats[t][ix,1].mean()
      for mode in modes:
       if mode=='mu': z=p-pm+mu
       elif mode=='sigma': z=(p-pm)/max(ps,1e-8)*sig+pm
       else: z=(p-pm)/max(ps,1e-8)*sig+mu
       rows.append([sd,f,sp,t,f'pldc_k{k}_{mode}',*moments(y,z)])
 M=pd.DataFrame(rows,columns=['seed','fold','speaker_id','target','condition','my','mp','my2','mp2','myp','n']); root=Path(c['outputs']['artifact_root']); root.mkdir(parents=True,exist_ok=True); M.to_parquet(root/'speaker_moments.parquet',index=False); pd.DataFrame(audit).to_csv(root/'retrieval_audit.csv',index=False); point=[]; delta=[]; B=int(c['parameters']['bootstrap_reps']); bs=int(c['parameters']['bootstrap_seed']); conds=sorted(M.condition.unique())
 for t in targets:
  for sd in seeds:
   z=M[(M.target==t)&(M.seed==sd)]
   for co in conds:
    g=z[z.condition==co].set_index('speaker_id').loc[S]
    for met in ['overall','between','within']: point.append({'seed':sd,'target':t,'condition':co,'metric':met,'ccc':float(ccc(arr(g,met).mean(0)))})
  for co in [x for x in conds if x!='baseline']:
   for met in ['overall','between','within']:
    rng=np.random.default_rng(seed(bs,t,co,met)); W=rng.multinomial(len(S),np.full(len(S),1/len(S)),size=B)/len(S); boots=[]; pts=[]
    for sd in seeds:
     z=M[(M.target==t)&(M.seed==sd)]; A=arr(z[z.condition==co].set_index('speaker_id').loc[S],met); C=arr(z[z.condition=='baseline'].set_index('speaker_id').loc[S],met); boots.append(ccc(W@A)-ccc(W@C)); pts.append(float(ccc(A.mean(0))-ccc(C.mean(0))))
    b=np.mean(np.vstack(boots),0); lo,hi=np.nanquantile(b,[.025,.975]); delta.append({'target':t,'condition':co,'metric':met,'delta_ccc_mean_across_seeds':np.mean(pts),'ci95_low':lo,'ci95_high':hi,'n_speakers':len(S),'bootstrap_reps':B})
 pd.DataFrame(point).to_csv(root/'component_metrics_by_seed.csv',index=False); pd.DataFrame(delta).to_csv(root/'component_deltas_vs_baseline.csv',index=False); (root/'run_metadata.json').write_text(json.dumps({'experiment_id':c['experiment_id'],'rows':len(d),'speakers':len(S),'bootstrap_unit':'speaker','modeling_seeds':seeds,'method':'PLDC common-representation reimplementation'},indent=2)+'\n'); print(pd.DataFrame(delta).query("condition.str.startswith('pldc_k1_')",engine='python').to_string(index=False))
if __name__=='__main__': main()
