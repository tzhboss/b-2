#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def iemocap_speaker(x):
    m=re.match(r'(Ses\d\d)[FM]_.+_([FM])\d+\.wav$',str(x))
    if not m: raise ValueError(x)
    return m.group(1)+'_'+m.group(2)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def equal_speaker_weights(df):
    c=df.speaker_id.astype(str).value_counts()
    w=df.speaker_id.astype(str).map(lambda s:1.0/c[s]).to_numpy(float)
    return w/w.mean()

def weighted_ccc(y,p,w):
    y=np.asarray(y,float); p=np.asarray(p,float); w=np.asarray(w,float); w=w/w.sum()
    my=np.sum(w*y); mp=np.sum(w*p)
    dy=y-my; dp=p-mp
    vy=np.sum(w*dy*dy); vp=np.sum(w*dp*dp); cov=np.sum(w*dy*dp)
    den=vy+vp+(my-mp)**2
    return float(2*cov/den) if den>0 else np.nan

def fit_predict(train,test,cols,ytr,wtr,alpha):
    xtr=train[cols].to_numpy(np.float32); xte=test[cols].to_numpy(np.float32)
    sc=StandardScaler(); sc.fit(xtr,sample_weight=wtr)
    model=Ridge(alpha=alpha); model.fit(sc.transform(xtr),ytr,sample_weight=wtr)
    return model.predict(sc.transform(xte))

def prepare(pattern):
    files=sorted(glob.glob(pattern))
    cols=["file","EmoAct","EmoVal","EmoDom","speaking_rate","pitch_mean","relative_db"]
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in files],ignore_index=True).dropna().copy()
    d=d[(d.pitch_mean>0)&(d.speaking_rate>0)].copy()
    d["sample_id"]=d.file.astype(str)
    d["speaker_id"]=d.file.map(iemocap_speaker)
    d["pitch_abs"]=12*np.log2(d.pitch_mean.astype(float))
    d["loud_abs"]=d.relative_db.astype(float)
    d["rate_abs"]=np.log(d.speaking_rate.astype(float))
    d=d.rename(columns={"EmoVal":"valence","EmoAct":"arousal","EmoDom":"dominance"})
    for t in ["valence","arousal","dominance"]:
        spmean=d.groupby("speaker_id")[t].mean()
        gm=float(d[t].mean())
        d[f"{t}_spmean"]=d.speaker_id.map(spmean)
        d[f"{t}_global"]=gm
        d[f"{t}_within"]=d[t]-d[f"{t}_spmean"]
    return d.reset_index(drop=True)

def reserved_order(d,n,seed):
    out={}
    for sp,idx in d.groupby("speaker_id",sort=True).indices.items():
        idx=np.asarray(idx,dtype=int)
        if len(idx)<=n: raise RuntimeError(f"{sp}: only {len(idx)} rows")
        rng=np.random.default_rng(stable_seed(seed,sp,"iemocap-reserve100"))
        p=idx.copy(); rng.shuffle(p)
        out[str(sp)]=p[:n]
    return out

def bootstrap_slopes(mat,lambdas,reps,seed):
    mat=np.asarray(mat,float); lambdas=np.asarray(lambdas,float)
    slopes=np.array([np.polyfit(lambdas,row,1)[0] for row in mat])
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(slopes),size=(reps,len(slopes)))
    bs=slopes[idx].mean(1)
    lo,hi=np.quantile(bs,[.025,.975])
    return float(slopes.mean()),float(lo),float(hi)

def row_hash(df):
    txt="\n".join(sorted(df.sample_id.astype(str).tolist()))
    return hashlib.sha256(txt.encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    d=prepare(cfg["dataset"]["iemocap_source_glob"])
    abs_cols=["pitch_abs","loud_abs","rate_abs"]
    speakers=sorted(d.speaker_id.astype(str).unique())
    oracle=d.groupby("speaker_id")[abs_cols].median()
    k_values=[int(x) for x in cfg["parameters"]["k_values"]]
    lambdas=[float(x) for x in cfg["parameters"]["lambda_values"]]
    targets=list(cfg["parameters"]["targets"])
    nreserve=int(cfg["parameters"]["reserve_enrollment"])
    nfold=int(cfg["parameters"]["outer_folds"])
    metrics=[]; audits=[]; errors=[]

    for seed in cfg["seed"]:
        orders=reserved_order(d,nreserve,int(seed))
        reserved=np.zeros(len(d),bool)
        for idx in orders.values(): reserved[idx]=True
        work0=d.loc[~reserved].copy().reset_index(drop=True)
        fmap=speaker_folds(speakers,nfold,int(seed))
        sf=work0.speaker_id.astype(str).map(fmap).to_numpy()

        # Oracle condition once per seed/fold/lambda on the exact fixed downstream pool.
        oracle_work=work0.copy()
        for j,nm in enumerate(["pitch","loud","rate"]):
            oracle_work[f"{nm}_oracle_center"]=oracle_work.speaker_id.map(oracle.iloc[:,j])
            oracle_work[f"{nm}_oracle_rel"]=oracle_work[abs_cols[j]]-oracle_work[f"{nm}_oracle_center"]

        for fold in range(nfold):
            tr=sf!=fold; te=sf==fold
            otr=oracle_work.loc[tr].copy(); ote=oracle_work.loc[te].copy()
            wtr=equal_speaker_weights(otr); wte=equal_speaker_weights(ote)
            audits.append({"seed":int(seed),"fold":fold,"K":0,
                           "train_hash":row_hash(otr),"test_hash":row_hash(ote),
                           "n_train":len(otr),"n_test":len(ote)})
            for lam in lambdas:
                ytr=np.column_stack([
                    otr[f"{t}_global"].to_numpy(float)+otr[f"{t}_within"].to_numpy(float)+
                    lam*(otr[f"{t}_spmean"].to_numpy(float)-otr[f"{t}_global"].to_numpy(float))
                    for t in targets])
                yte=np.column_stack([
                    ote[f"{t}_global"].to_numpy(float)+ote[f"{t}_within"].to_numpy(float)+
                    lam*(ote[f"{t}_spmean"].to_numpy(float)-ote[f"{t}_global"].to_numpy(float))
                    for t in targets])
                reps={
                  "absolute":abs_cols,
                  "oracle_relative":["pitch_oracle_rel","loud_oracle_rel","rate_oracle_rel"],
                  "oracle_hybrid":["pitch_oracle_rel","loud_oracle_rel","rate_oracle_rel",
                                   "pitch_oracle_center","loud_oracle_center","rate_oracle_center"]
                }
                for rep,cols in reps.items():
                    pred=fit_predict(otr,ote,cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))
                    for jj,t in enumerate(targets):
                        metrics.append({"seed":int(seed),"fold":fold,"K":0,"lambda":lam,
                                        "target":t,"representation":rep,
                                        "ccc":weighted_ccc(yte[:,jj],pred[:,jj],wte)})

        # K-shot conditions on the same downstream pool.
        for K in k_values:
            center={}
            for sp,idx in orders.items():
                center[sp]=d.loc[idx[:K],abs_cols].median().to_numpy(float)
            kw=work0.copy()
            for j,nm in enumerate(["pitch","loud","rate"]):
                kw[f"{nm}_center"]=np.array([center[str(sp)][j] for sp in kw.speaker_id])
                kw[f"{nm}_rel"]=kw[abs_cols[j]]-kw[f"{nm}_center"]
            for sp in speakers:
                for j,nm in enumerate(["pitch","loudness","log_rate"]):
                    errors.append({"seed":int(seed),"K":K,"speaker_id":sp,"attribute":nm,
                                   "error_to_oracle":abs(center[sp][j]-float(oracle.loc[sp].iloc[j]))})
            for fold in range(nfold):
                tr=sf!=fold; te=sf==fold
                train=kw.loc[tr].copy(); test=kw.loc[te].copy()
                wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
                audits.append({"seed":int(seed),"fold":fold,"K":K,
                               "train_hash":row_hash(train),"test_hash":row_hash(test),
                               "n_train":len(train),"n_test":len(test)})
                for lam in lambdas:
                    ytr=np.column_stack([
                        train[f"{t}_global"].to_numpy(float)+train[f"{t}_within"].to_numpy(float)+
                        lam*(train[f"{t}_spmean"].to_numpy(float)-train[f"{t}_global"].to_numpy(float))
                        for t in targets])
                    yte=np.column_stack([
                        test[f"{t}_global"].to_numpy(float)+test[f"{t}_within"].to_numpy(float)+
                        lam*(test[f"{t}_spmean"].to_numpy(float)-test[f"{t}_global"].to_numpy(float))
                        for t in targets])
                    reps={
                      "absolute":abs_cols,
                      "kshot_relative":["pitch_rel","loud_rel","rate_rel"],
                      "kshot_hybrid":["pitch_rel","loud_rel","rate_rel","pitch_center","loud_center","rate_center"]
                    }
                    for rep,cols in reps.items():
                        pred=fit_predict(train,test,cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))
                        for jj,t in enumerate(targets):
                            metrics.append({"seed":int(seed),"fold":fold,"K":K,"lambda":lam,
                                            "target":t,"representation":rep,
                                            "ccc":weighted_ccc(yte[:,jj],pred[:,jj],wte)})

    m=pd.DataFrame(metrics)
    audits=pd.DataFrame(audits)
    e=pd.DataFrame(errors)
    es=e.groupby(["K","attribute"],as_index=False).agg(mae_to_oracle=("error_to_oracle","mean"))

    # Verify row hashes fixed across K within seed/fold.
    chk=audits.groupby(["seed","fold"]).agg(train_hash_n=("train_hash","nunique"),test_hash_n=("test_hash","nunique"))
    if (chk.train_hash_n!=1).any() or (chk.test_hash_n!=1).any():
        raise RuntimeError("downstream row hash changed across K")

    slope_rows=[]
    # Oracle slope from K=0.
    for t in targets:
        g=m[(m.K==0)&(m.target==t)].pivot_table(
            index=["seed","fold","lambda"],columns="representation",values="ccc").reset_index()
        g["effect"]=g.oracle_relative-g.absolute
        mat=g.pivot_table(index=["seed","fold"],columns="lambda",values="effect")[lambdas].to_numpy()
        slope,lo,hi=bootstrap_slopes(mat,lambdas,int(cfg["parameters"]["bootstrap_reps"]),stable_seed("oracle",t))
        slope_rows.append({"K":0,"target":t,"effect":"relative_minus_absolute",
                           "slope_mean":slope,"ci95_low":lo,"ci95_high":hi})

    for K in k_values:
        for t in targets:
            g=m[(m.K==K)&(m.target==t)].pivot_table(
                index=["seed","fold","lambda"],columns="representation",values="ccc").reset_index()
            g["ra"]=g.kshot_relative-g.absolute
            g["hr"]=g.kshot_hybrid-g.kshot_relative
            for effect,col in [("relative_minus_absolute","ra"),("hybrid_minus_relative","hr")]:
                mat=g.pivot_table(index=["seed","fold"],columns="lambda",values=col)[lambdas].to_numpy()
                slope,lo,hi=bootstrap_slopes(mat,lambdas,int(cfg["parameters"]["bootstrap_reps"]),
                                             stable_seed(K,t,effect))
                slope_rows.append({"K":K,"target":t,"effect":effect,
                                   "slope_mean":slope,"ci95_low":lo,"ci95_high":hi})
    slopes=pd.DataFrame(slope_rows)
    oracle_sl=slopes[(slopes.K==0)&(slopes.effect=="relative_minus_absolute")][["target","slope_mean"]].rename(columns={"slope_mean":"oracle_slope"})
    gap=slopes[(slopes.K>0)&(slopes.effect=="relative_minus_absolute")].merge(oracle_sl,on="target")
    gap["abs_gap_to_oracle"]=(gap.slope_mean-gap.oracle_slope).abs()

    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{"rows_total":len(d),"speakers":len(speakers),"reserved_per_speaker":nreserve,
                   "downstream_rows_per_seed":len(d)-nreserve*len(speakers)}]).to_csv(root/"data_inventory.csv",index=False)
    audits.to_csv(root/"audit_rows.csv",index=False)
    es.to_csv(root/"center_error_summary.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    slopes.to_csv(root/"slope_tests.csv",index=False)
    gap.to_csv(root/"slope_gap_summary.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"reserve_enrollment":nreserve,
      "k_values":k_values,"lambda_values":lambdas,
      "fixed_downstream_pool":True,"nested_enrollment_prefixes":True
    },indent=2)+"\n")
    print("CENTER ERROR"); print(es.to_string(index=False))
    print("\nSLOPES"); print(slopes.to_string(index=False))
    print("\nGAPS"); print(gap.to_string(index=False))

if __name__=="__main__":
    main()
