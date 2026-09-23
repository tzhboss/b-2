#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ATTRS=["pitch","f0var","loud","rate","pause","voiced"]
ABS=[f"{a}_abs" for a in ATTRS]

def stable_seed(*parts):
    b="|".join(map(str,parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(b).digest()[:8],"big")%(2**32)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def equal_speaker_weights(df):
    c=df.speaker_id.astype(str).value_counts()
    w=df.speaker_id.astype(str).map(lambda s:1.0/c[s]).to_numpy(float)
    return w/w.mean()

def prepare(path):
    cols=["dataset","sample_id","speaker_id","arousal_mean_1_7","dominance_mean_1_7",
          "f0_median_hz","f0_iqr_semitone","speech_lufs","phoneme_articulation_rate",
          "pause_ratio","voiced_ratio","pitch_reference_scope","loudness_reference_scope","rate_reference_scope"]
    d=pd.read_parquet(path,columns=cols)
    d=d[d.dataset.eq("msp")].dropna().copy()
    d=d[(d.speaker_id.astype(str)!="Unknown")&(d.f0_median_hz>0)&
        (d.phoneme_articulation_rate>0)&
        d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
    sizes=d.groupby("speaker_id").size()
    d=d[d.speaker_id.isin(sizes[sizes>10].index)].copy()
    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
    d["f0var_abs"]=d.f0_iqr_semitone.astype(float)
    d["loud_abs"]=d.speech_lufs.astype(float)
    d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
    d["pause_abs"]=d.pause_ratio.astype(float)
    d["voiced_abs"]=d.voiced_ratio.astype(float)
    return d.rename(columns={"arousal_mean_1_7":"arousal","dominance_mean_1_7":"dominance"}).reset_index(drop=True)

def fit_fa(X,speaker_ids,q,iters):
    mu=X.mean(axis=0); Xc=X-mu
    speakers=np.asarray(speaker_ids,dtype=object)
    uniq=sorted(set(speakers.tolist()))
    groups={sp:np.flatnonzero(speakers==sp) for sp in uniq}
    M=np.vstack([Xc[idx].mean(axis=0) for idx in groups.values()])
    _,_,vt=np.linalg.svd(M,full_matrices=False)
    F=0.5*vt[:q].T.copy()
    psi=np.maximum(Xc.var(axis=0),1e-3)
    eye=np.eye(q)
    for _ in range(iters):
        invpsi=1.0/np.maximum(psi,1e-6)
        post={}; cross=np.zeros((X.shape[1],q)); yy=np.zeros((q,q))
        for sp,idx in groups.items():
            Xi=Xc[idx]; n=len(idx); S=Xi.sum(axis=0)
            C=np.linalg.inv(eye+n*(F.T*invpsi)@F)
            m=C@(F.T@(invpsi*S))
            Eyy=C+np.outer(m,m)
            post[sp]=(m,Eyy)
            cross+=np.outer(S,m)
            yy+=n*Eyy
        F=cross@np.linalg.pinv(yy)
        residual=np.zeros(X.shape[1])
        for sp,idx in groups.items():
            Xi=Xc[idx]; n=len(idx); S=Xi.sum(axis=0)
            m,Eyy=post[sp]
            residual+=np.sum(Xi*Xi,axis=0)
            residual+=-2.0*S*(F@m)
            residual+=n*np.diag(F@Eyy@F.T)
        psi=np.maximum(residual/len(X),1e-5)
    return {"mu":mu,"F":F,"psi":psi}

def posterior_normalize(X,speaker_ids,fa):
    mu,F,psi=fa["mu"],fa["F"],fa["psi"]
    invpsi=1.0/np.maximum(psi,1e-6)
    q=F.shape[1]; eye=np.eye(q)
    sp=np.asarray(speaker_ids,dtype=object); out=np.empty_like(X,float)
    for s in sorted(set(sp.tolist())):
        idx=np.flatnonzero(sp==s); Xi=X[idx]-mu
        n=len(idx); S=Xi.sum(axis=0)
        C=np.linalg.inv(eye+n*(F.T*invpsi)@F)
        m=C@(F.T@(invpsi*S))
        center=mu+F@m
        out[idx]=X[idx]-center
    return out

def ccc5(m):
    m=np.asarray(m,float); my,mp,my2,mp2,myp=[m[...,i] for i in range(5)]
    vy=np.maximum(my2-my*my,0); vp=np.maximum(mp2-mp*mp,0)
    cov=myp-my*mp; den=vy+vp+(my-mp)**2
    return np.divide(2*cov,den,out=np.full_like(den,np.nan,dtype=float),where=den>0)

def speaker_moments(y,p,speaker_ids):
    rows=[]; sp=np.asarray(speaker_ids,dtype=object)
    for s in sorted(set(sp.tolist())):
        idx=np.flatnonzero(sp==s); yy=y[idx]; pp=p[idx]
        rows.append([s,yy.mean(),pp.mean(),np.mean(yy*yy),np.mean(pp*pp),np.mean(yy*pp),len(idx)])
    return rows

def metric_array(g,metric):
    if metric=="overall":
        return g[["my","mp","my2","mp2","myp"]].to_numpy(float)
    if metric=="between":
        return np.column_stack([g.my,g.mp,g.my**2,g.mp**2,g.my*g.mp]).astype(float)
    return np.column_stack([np.zeros(len(g)),np.zeros(len(g)),
        g.my2-g.my**2,g.mp2-g.mp**2,g.myp-g.my*g.mp]).astype(float)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    args=ap.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text())
    d=prepare(Path(os.environ[cfg["dataset"]["source_env"]]))
    seeds=[int(x) for x in cfg["seed"]]
    nfold=int(cfg["parameters"]["outer_folds"])
    dims=[int(x) for x in cfg["parameters"]["latent_dims"]]
    iters=int(cfg["parameters"]["em_iterations"])
    alpha=float(cfg["parameters"]["ridge_alpha"])
    targets=list(cfg["parameters"]["targets"])
    speakers=sorted(d.speaker_id.astype(str).unique())
    rows=[]; audits=[]
    for seed in seeds:
        fmap=speaker_folds(speakers,nfold,seed)
        sf=d.speaker_id.astype(str).map(fmap).to_numpy()
        for fold in range(nfold):
            tr=sf!=fold; te=sf==fold
            train=d.loc[tr].copy(); test=d.loc[te].copy()
            if set(train.speaker_id.astype(str)) & set(test.speaker_id.astype(str)):
                raise RuntimeError("speaker leakage")
            wtr=equal_speaker_weights(train)
            Xtr=train[ABS].to_numpy(float); Xte=test[ABS].to_numpy(float)
            sc=StandardScaler(); sc.fit(Xtr,sample_weight=wtr)
            Ztr=sc.transform(Xtr); Zte=sc.transform(Xte)
            Ytr=train[targets].to_numpy(float); Yte=test[targets].to_numpy(float)
            base=Ridge(alpha=alpha).fit(Ztr,Ytr,sample_weight=wtr).predict(Zte)
            for j,t in enumerate(targets):
                for rec in speaker_moments(Yte[:,j],base[:,j],test.speaker_id.astype(str).to_numpy()):
                    rows.append([seed,fold,0,t,"baseline",*rec])
            for q in dims:
                fa=fit_fa(Ztr,train.speaker_id.astype(str).to_numpy(),q,iters)
                Ntr=posterior_normalize(Ztr,train.speaker_id.astype(str).to_numpy(),fa)
                Nte=posterior_normalize(Zte,test.speaker_id.astype(str).to_numpy(),fa)
                pred=Ridge(alpha=alpha).fit(Ntr,Ytr,sample_weight=wtr).predict(Nte)
                audits.append({"seed":seed,"fold":fold,"latent_dim":q,
                               "psi_min":float(np.min(fa["psi"])),"psi_max":float(np.max(fa["psi"])),
                               "F_fro":float(np.linalg.norm(fa["F"])),
                               "train_rows":int(tr.sum()),"test_rows":int(te.sum())})
                for j,t in enumerate(targets):
                    for rec in speaker_moments(Yte[:,j],pred[:,j],test.speaker_id.astype(str).to_numpy()):
                        rows.append([seed,fold,q,t,f"fa_q{q}",*rec])
    cols=["seed","fold","latent_dim","target","condition","speaker_id","my","mp","my2","mp2","myp","n"]
    mom=pd.DataFrame(rows,columns=cols)
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    mom.to_parquet(root/"speaker_moments.parquet",index=False)
    pd.DataFrame(audits).to_csv(root/"fa_audit.csv",index=False)
    B=int(cfg["parameters"]["bootstrap_reps"]); bseed=int(cfg["parameters"]["bootstrap_seed"])
    point=[]; delta=[]; conditions=["baseline"]+[f"fa_q{q}" for q in dims]
    for target in targets:
        for seed in seeds:
            z=mom[(mom.target==target)&(mom.seed==seed)]; sp=sorted(z.speaker_id.unique())
            for cond in conditions:
                g=z[z.condition==cond].set_index("speaker_id").loc[sp]
                for metric in ["overall","between","within"]:
                    arr=metric_array(g,metric)
                    point.append({"seed":seed,"target":target,"condition":cond,
                                  "metric":metric,"ccc":float(ccc5(arr.mean(0)))})
        for cond in [f"fa_q{q}" for q in dims]:
            for metric in ["overall","between","within"]:
                rng=np.random.default_rng(stable_seed(bseed,target,cond,metric))
                counts=rng.multinomial(len(speakers),np.full(len(speakers),1/len(speakers)),size=B).astype(float)/len(speakers)
                boots=[]; pts=[]
                for seed in seeds:
                    z=mom[(mom.target==target)&(mom.seed==seed)]
                    A=metric_array(z[z.condition==cond].set_index("speaker_id").loc[speakers],metric)
                    C=metric_array(z[z.condition=="baseline"].set_index("speaker_id").loc[speakers],metric)
                    boots.append(ccc5(counts@A)-ccc5(counts@C))
                    pts.append(float(ccc5(A.mean(0))-ccc5(C.mean(0))))
                boot=np.mean(np.vstack(boots),axis=0); lo,hi=np.nanquantile(boot,[.025,.975])
                delta.append({"target":target,"condition":cond,"metric":metric,
                              "delta_ccc_mean_across_seeds":float(np.mean(pts)),
                              "ci95_low":float(lo),"ci95_high":float(hi),
                              "n_speakers":len(speakers),"bootstrap_reps":B})
    pd.DataFrame(point).to_csv(root/"component_metrics_by_seed.csv",index=False)
    out=pd.DataFrame(delta); out.to_csv(root/"component_deltas_vs_baseline.csv",index=False)
    meta={"experiment_id":cfg["experiment_id"],"rows":len(d),"speakers":len(speakers),
          "method":"Dang2016 factor-analysis normalization mechanism on common six-feature Ridge",
          "em_iterations":iters,"latent_dims":dims,"bootstrap_unit":"speaker","modeling_seeds":seeds}
    (root/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(out.to_string(index=False))

if __name__=="__main__":
    main()
