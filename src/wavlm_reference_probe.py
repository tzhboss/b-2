from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ABS_COLUMNS={"pitch":"abs_pitch_semitone","loudness":"speech_lufs","rate":"abs_log_rate"}
REL_COLUMNS={"pitch":"pitch_relative_st","loudness":"loudness_relative_lu","rate":"rel_log_rate"}

def prepare(df):
    x=df.copy()
    x["abs_pitch_semitone"]=12.0*np.log2(x["f0_median_hz"].astype(float))
    x["abs_log_rate"]=np.log(x["phoneme_articulation_rate"].astype(float))
    x["rel_log_rate"]=np.log(x["rate_relative_ratio"].astype(float))
    return x

def make_folds(df,label,n_splits,seed):
    rng=np.random.default_rng(seed)
    folds=np.full(len(df),-1,dtype=int)
    groups=df.reset_index(drop=True).groupby(["speaker_id",label],dropna=False,sort=True).indices
    for idx in groups.values():
        idx=np.asarray(idx); rng.shuffle(idx)
        a=np.arange(len(idx))%n_splits; rng.shuffle(a); folds[idx]=a
    if (folds<0).any(): raise RuntimeError("unassigned fold")
    return folds

def prosody_cols(attrs,rep):
    if rep=="wavlm_only": return []
    if rep=="absolute": return [ABS_COLUMNS[a] for a in attrs]
    if rep=="relative": return [REL_COLUMNS[a] for a in attrs]
    if rep=="hybrid": return [ABS_COLUMNS[a] for a in attrs]+[REL_COLUMNS[a] for a in attrs]
    raise ValueError(rep)

def matrix(df,emb,attrs,rep):
    p=prosody_cols(attrs,rep)
    if not p: return emb
    return np.concatenate([emb,df[p].to_numpy(dtype=np.float32)],axis=1)

def fit_eval(train_x,train_y,test_x,test_y,labels,alpha,class_weight,solver):
    model=Pipeline([
        ("scale",StandardScaler()),
        ("clf",RidgeClassifier(alpha=alpha,class_weight=class_weight,solver=solver))
    ])
    model.fit(train_x,train_y)
    pred=model.predict(test_x)
    return {
        "macro_f1":float(f1_score(test_y,pred,labels=labels,average="macro",zero_division=0)),
        "accuracy":float(accuracy_score(test_y,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(test_y,pred))
    }

def bootstrap(values,reps,seed):
    v=np.asarray(values,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v))); m=v[idx].mean(1)
    return tuple(map(float,np.quantile(m,[.025,.975])))
