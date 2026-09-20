#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLS=["sample_id","dataset","speaker_id","emotion_5class_candidate","emotion_training_usable","f0_median_hz"]

def stable_int(*parts):
    h=hashlib.sha256("|".join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:8],"little")

def assign_speaker_folds(speakers,n_splits,seed):
    sp=np.asarray(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(sp)
    return {s:i%n_splits for i,s in enumerate(sp)}

def load_embeddings(root):
    ids=[]; arr=[]
    for p in sorted(Path(root).glob("shard-*-of-*.npz")):
        z=np.load(p,allow_pickle=False)
        ids.extend(z["sample_id"].astype(str).tolist())
        arr.append(z["embedding"].astype(np.float32))
    e=np.concatenate(arr,axis=0)
    if len(ids)!=len(set(ids)): raise RuntimeError("duplicate embeddings")
    return ids,e

def baseline_map(df,enr,k):
    lookup=dict(zip(df.sample_id.astype(str),df.abs_pitch_st.astype(float)))
    out={}
    for sp,g in enr.groupby("speaker_id"):
        g=g.sort_values("rank")
        ids=g.sample_id.astype(str).tolist()[:k]
        out[str(sp)]=float(np.median([lookup[x] for x in ids]))
    return out

def bootstrap(v,reps,seed):
    v=np.asarray(v,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v)))
    m=v[idx].mean(1)
    return tuple(map(float,np.quantile(m,[.025,.975])))

def fit_eval(Xtr,ytr,Xte,yte,labels,cfg):
    model=Pipeline([
      ("scale",StandardScaler()),
      ("clf",RidgeClassifier(alpha=float(cfg["alpha"]),class_weight=cfg["class_weight"],solver=cfg["solver"]))
    ])
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    return {
      "macro_f1":float(f1_score(yte,pred,labels=labels,average="macro",zero_division=0)),
      "accuracy":float(accuracy_score(yte,pred)),
      "balanced_accuracy":float(balanced_accuracy_score(yte,pred))
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--dataset",default=None)
    ap.add_argument("--output-dir",default=None)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(cfg["outputs"]["artifact_root"])
    out=Path(args.output_dir) if args.output_dir else root
    out.mkdir(parents=True,exist_ok=True)

    src=Path(os.environ[cfg["dataset"]["source_env"]])
    enr_path=Path(os.environ[cfg["dataset"]["enrollment_env"]])
    exp05_hash_path=Path(os.environ[cfg["dataset"]["exp05_hash_env"]])
    emb_root=Path(os.environ[cfg["embeddings"]["root_env"]])
    emb_inv=Path(os.environ[cfg["embeddings"]["inventory_env"]])

    df=pd.read_parquet(src,columns=COLS)
    df["speaker_id"]=df.speaker_id.astype(str)
    df["abs_pitch_st"]=12*np.log2(df.f0_median_hz.astype(float))
    enrollment=pd.read_csv(enr_path); enrollment["speaker_id"]=enrollment.speaker_id.astype(str)
    exp05_hash=pd.read_csv(exp05_hash_path)
    ids,emb=load_embeddings(emb_root)
    pos={s:i for i,s in enumerate(ids)}
    inv=json.loads(emb_inv.read_text())

    datasets=[args.dataset] if args.dataset else cfg["dataset"]["datasets"]
    rows=[]; hashes=[]; inventories=[]
    n_splits=int(cfg["parameters"]["n_splits"])

    for dataset in datasets:
        all_d=df[(df.dataset==dataset)&df.f0_median_hz.notna()].copy().sort_values("sample_id").reset_index(drop=True)
        speakers=sorted(all_d.speaker_id.unique())
        for seed in cfg["seed"]:
            enr=enrollment[(enrollment.dataset==dataset)&(enrollment.seed==int(seed))].copy()
            eids=set(enr.sample_id.astype(str))
            target=all_d[
                all_d.emotion_training_usable.fillna(False)&
                all_d.emotion_5class_candidate.fillna("").ne("")&
                ~all_d.sample_id.astype(str).isin(eids)
            ].copy().sort_values("sample_id").reset_index(drop=True)
            if not set(target.sample_id.astype(str)).issubset(pos):
                raise RuntimeError("missing WavLM target embeddings")
            target["emb_row"]=target.sample_id.astype(str).map(pos)
            labels=sorted(target.emotion_5class_candidate.astype(str).unique().tolist())
            folds=assign_speaker_folds(speakers,n_splits,int(seed))
            target["fold"]=target.speaker_id.map(folds)
            bm1=baseline_map(all_d,enr,1); bm10=baseline_map(all_d,enr,10)

            for fold in range(n_splits):
                test_sp={s for s,f in folds.items() if f==fold}; train_sp=set(speakers)-test_sp
                tr=target[target.speaker_id.isin(train_sp)].copy()
                te=target[target.speaker_id.isin(test_sp)].copy()
                Etr=emb[tr.emb_row.to_numpy()]; Ete=emb[te.emb_row.to_numpy()]
                abs_tr=tr.abs_pitch_st.to_numpy(); abs_te=te.abs_pitch_st.to_numpy()
                rel1_tr=abs_tr-tr.speaker_id.map(bm1).to_numpy(); rel1_te=abs_te-te.speaker_id.map(bm1).to_numpy()
                rel10_tr=abs_tr-tr.speaker_id.map(bm10).to_numpy(); rel10_te=abs_te-te.speaker_id.map(bm10).to_numpy()

                mats={
                  "wavlm_only":(Etr,Ete),
                  "absolute":(np.column_stack([Etr,abs_tr]),np.column_stack([Ete,abs_te])),
                  "relative_k1":(np.column_stack([Etr,rel1_tr]),np.column_stack([Ete,rel1_te])),
                  "relative_k10":(np.column_stack([Etr,rel10_tr]),np.column_stack([Ete,rel10_te])),
                  "hybrid_k1":(np.column_stack([Etr,abs_tr,rel1_tr]),np.column_stack([Ete,abs_te,rel1_te])),
                  "hybrid_k10":(np.column_stack([Etr,abs_tr,rel10_tr]),np.column_stack([Ete,abs_te,rel10_te])),
                }
                ytr=tr.emotion_5class_candidate.astype(str).to_numpy()
                yte=te.emotion_5class_candidate.astype(str).to_numpy()
                th=hashlib.sha256("\n".join(te.sample_id.astype(str)).encode()).hexdigest()
                expected=exp05_hash[(exp05_hash.dataset==dataset)&(exp05_hash.seed==int(seed))&
                                    (exp05_hash.fold==fold)].test_sample_id_sha256.unique()
                if len(expected)!=1 or expected[0]!=th:
                    raise RuntimeError(f"EXP05 target hash mismatch {dataset}/{seed}/{fold}")
                for rep,(Xtr,Xte) in mats.items():
                    sc=fit_eval(Xtr,ytr,Xte,yte,labels,cfg["parameters"]["classifier"])
                    rows.append({"dataset":dataset,"seed":int(seed),"fold":fold,"representation":rep,
                                 "n_train":len(tr),"n_test":len(te),
                                 "n_train_speakers":len(train_sp),"n_test_speakers":len(test_sp),**sc})
                    hashes.append({"dataset":dataset,"seed":int(seed),"fold":fold,
                                   "representation":rep,"test_sample_id_sha256":th})
            inventories.append({"dataset":dataset,"seed":int(seed),"target_rows":len(target),
                                "speakers":len(speakers),"embedding_dim":int(emb.shape[1])})

    m=pd.DataFrame(rows); h=pd.DataFrame(hashes); invdf=pd.DataFrame(inventories)
    summ=m.groupby(["dataset","representation"],as_index=False).agg(
        macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),
        accuracy_mean=("accuracy","mean"),balanced_accuracy_mean=("balanced_accuracy","mean"),n_eval=("macro_f1","size"))
    p=m.pivot_table(index=["dataset","seed","fold"],columns="representation",values="macro_f1")
    comps=[("relative_k1","absolute"),("relative_k10","absolute"),("relative_k10","relative_k1"),
           ("hybrid_k1","wavlm_only"),("hybrid_k10","wavlm_only"),("relative_k10","wavlm_only")]
    dr=[]
    for dataset,g in p.groupby(level=0):
        for a,b in comps:
            d=(g[a]-g[b]).dropna().to_numpy()
            lo,hi=bootstrap(d,int(cfg["parameters"]["bootstrap_reps"]),stable_int(dataset,a,b))
            dr.append({"dataset":dataset,"comparison":f"{a}-{b}","delta_macro_f1_mean":float(d.mean()),
                       "ci95_low":lo,"ci95_high":hi,"n_pairs":len(d)})
    delta=pd.DataFrame(dr)

    invdf.to_csv(out/"data_inventory.csv",index=False)
    m.to_csv(out/"metrics_by_fold.csv",index=False)
    summ.to_csv(out/"summary.csv",index=False)
    delta.to_csv(out/"paired_deltas.csv",index=False)
    h.to_csv(out/"target_hash_checks.csv",index=False)
    meta={"experiment_id":cfg["experiment_id"],"datasets":datasets,
          "embedding_inventory":inv,"embedding_rows":len(ids),"embedding_dim":int(emb.shape[1]),
          "seeds":cfg["seed"],"n_splits":n_splits}
    (out/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(summ.to_string(index=False))
    print("\nDELTAS\n",delta.to_string(index=False))

if __name__=="__main__":
    main()
