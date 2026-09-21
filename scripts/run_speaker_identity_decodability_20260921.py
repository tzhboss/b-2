#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score

COLS=[
 "dataset","sample_id","speaker_id","transcript","audio_path",
 "f0_median_hz","pitch_relative_st","pitch_reference_scope",
 "speech_lufs","loudness_relative_lu","loudness_reference_scope",
 "phoneme_articulation_rate","rate_relative_ratio","rate_reference_scope"
]

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def content_id(row):
    ds=row.dataset
    if ds=="esd_english":
        t=str(row.transcript).strip()
        return "txt:"+hashlib.sha1(t.encode()).hexdigest()[:16]
    if ds=="mead_part0":
        return "sent:"+Path(str(row.audio_path)).stem
    if ds=="ravdess_speech":
        parts=Path(str(row.audio_path)).stem.split("-")
        if len(parts)!=7: raise ValueError(row.audio_path)
        # canonical RAVDESS: modality-channel-emotion-intensity-statement-repetition-actor
        return "stmt_rep:"+parts[4]+"_"+parts[5]
    raise ValueError(ds)

def prepare(raw,ds,aset):
    d=raw[raw.dataset.eq(ds)].copy()
    req=["sample_id","speaker_id","f0_median_hz","pitch_relative_st","pitch_reference_scope"]
    if aset=="all":
        req += ["speech_lufs","loudness_relative_lu","loudness_reference_scope",
                "phoneme_articulation_rate","rate_relative_ratio","rate_reference_scope"]
    if ds=="esd_english": req += ["transcript"]
    else: req += ["audio_path"]
    d=d.dropna(subset=req).copy()
    d=d[(d.speaker_id.astype(str)!="Unknown")&(d.f0_median_hz>0)&
        d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
    d["pitch_rel"]=d.pitch_relative_st.astype(float)
    d["pitch_base"]=d.pitch_abs-d.pitch_rel
    if aset=="all":
        d=d[(d.phoneme_articulation_rate>0)&(d.rate_relative_ratio>0)&
            d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
            d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
        d["loud_abs"]=d.speech_lufs.astype(float)
        d["loud_rel"]=d.loudness_relative_lu.astype(float)
        d["loud_base"]=d.loud_abs-d.loud_rel
        d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
        d["rate_rel"]=np.log(d.rate_relative_ratio.astype(float))
        d["rate_base"]=d.rate_abs-d.rate_rel
    d["content_id"]=d.apply(content_id,axis=1)
    return d.reset_index(drop=True)

def rep_cols(aset,rep):
    roots={"pitch":("pitch_abs","pitch_rel","pitch_base"),
           "loud":("loud_abs","loud_rel","loud_base"),
           "rate":("rate_abs","rate_rel","rate_base")}
    attrs=["pitch"] if aset=="pitch" else ["pitch","loud","rate"]
    a=[roots[x][0] for x in attrs]; r=[roots[x][1] for x in attrs]; b=[roots[x][2] for x in attrs]
    if rep=="absolute": return a
    if rep=="relative": return r
    if rep=="baseline_only": return b
    if rep=="relative_plus_baseline": return r+b
    raise ValueError(rep)

def make_fold_map(contents,ds,seed):
    contents=np.array(sorted(contents),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(contents)
    nfold=4 if ds=="ravdess_speech" else 5
    return {c:i%nfold for i,c in enumerate(contents)},nfold

def bootstrap(v,reps,seed):
    v=np.asarray(v,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v)))
    z=v[idx].mean(1)
    lo,hi=np.quantile(z,[.025,.975])
    return float(v.mean()),float(lo),float(hi)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]])
    raw=pd.read_parquet(src,columns=COLS)
    datasets=["esd_english","mead_part0","ravdess_speech"]
    metrics=[]; inv=[]; finv=[]
    for ds in datasets:
      for aset in cfg["parameters"]["attribute_sets"]:
        d=prepare(raw,ds,aset)
        speakers=sorted(d.speaker_id.astype(str).unique())
        contents=sorted(d.content_id.unique())
        inv.append({"dataset":ds,"attribute_set":aset,"rows":len(d),"speakers":len(speakers),
                    "content_groups":len(contents),"min_rows_per_speaker":int(d.groupby("speaker_id").size().min())})
        for seed in cfg["seed"]:
            fmap,nfold=make_fold_map(contents,ds,stable_seed(ds,aset,seed))
            folds=d.content_id.map(fmap).to_numpy()
            for fold in range(nfold):
                tr=folds!=fold; te=folds==fold
                train=d.loc[tr].copy(); test=d.loc[te].copy()
                trc=set(train.content_id); tec=set(test.content_id)
                if trc&tec: raise RuntimeError("content leakage")
                train_sp=set(train.speaker_id.astype(str)); test_sp=set(test.speaker_id.astype(str))
                all_sp=set(speakers)
                if train_sp!=all_sp or test_sp!=all_sp:
                    # Drop unusable fold rather than silently changing class universe.
                    finv.append({"dataset":ds,"attribute_set":aset,"seed":int(seed),"fold":fold,
                                 "usable":False,"train_contents":len(trc),"test_contents":len(tec),
                                 "train_speakers":len(train_sp),"test_speakers":len(test_sp)})
                    continue
                finv.append({"dataset":ds,"attribute_set":aset,"seed":int(seed),"fold":fold,
                             "usable":True,"train_contents":len(trc),"test_contents":len(tec),
                             "train_speakers":len(train_sp),"test_speakers":len(test_sp)})
                ytr=train.speaker_id.astype(str).to_numpy(); yte=test.speaker_id.astype(str).to_numpy()
                for rep in cfg["parameters"]["representations"]:
                    cols=rep_cols(aset,rep)
                    xtr=train[cols].to_numpy(np.float32); xte=test[cols].to_numpy(np.float32)
                    sc=StandardScaler(); xtr=sc.fit_transform(xtr); xte=sc.transform(xte)
                    clf=RidgeClassifier(alpha=float(cfg["parameters"]["ridge_alpha"]),class_weight="balanced")
                    clf.fit(xtr,ytr); pred=clf.predict(xte)
                    metrics.append({"dataset":ds,"attribute_set":aset,"seed":int(seed),"fold":fold,
                                    "representation":rep,
                                    "macro_f1":f1_score(yte,pred,average="macro"),
                                    "balanced_accuracy":balanced_accuracy_score(yte,pred),
                                    "n_train":len(train),"n_test":len(test)})
    m=pd.DataFrame(metrics); inventory=pd.DataFrame(inv); fi=pd.DataFrame(finv)
    s=m.groupby(["dataset","attribute_set","representation"],as_index=False).agg(
        macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),
        bal_acc_mean=("balanced_accuracy","mean"),n_eval=("macro_f1","size"))
    p=m.pivot_table(index=["dataset","attribute_set","seed","fold"],columns="representation",values="macro_f1")
    rows=[]
    for (ds,aset),g in p.groupby(level=[0,1]):
        for a,b in [("relative","absolute"),("baseline_only","relative"),
                    ("relative_plus_baseline","relative"),("relative_plus_baseline","absolute")]:
            v=(g[a]-g[b]).dropna().to_numpy()
            mean,lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(ds,aset,a,b))
            rows.append({"dataset":ds,"attribute_set":aset,"comparison":f"{a}-{b}",
                         "delta_macro_f1_mean":mean,"ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)})
    delta=pd.DataFrame(rows)
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    inventory.to_csv(root/"data_inventory.csv",index=False)
    fi.to_csv(root/"fold_inventory.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    s.to_csv(root/"summary.csv",index=False)
    delta.to_csv(root/"paired_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"source":str(src),"seeds":cfg["seed"],
      "split":"content-disjoint; ESD transcript, MEAD sentence index, RAVDESS statement x repetition",
      "role":"representation decodability audit, not deployment speaker identification"
    },indent=2)+"\n")
    print("INVENTORY"); print(inventory.to_string(index=False))
    print("\nSUMMARY"); print(s.to_string(index=False))
    print("\nDELTAS"); print(delta.to_string(index=False))

if __name__=="__main__":
    main()
