#!/usr/bin/env python3
import argparse, hashlib, itertools, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd, yaml

def ccc(m):
    m=np.asarray(m,float); my,mp,my2,mp2,myp=[m[...,i] for i in range(5)]
    vy=np.maximum(my2-my*my,0); vp=np.maximum(mp2-mp*mp,0); den=vy+vp+(my-mp)**2
    return np.divide(2*(myp-my*mp),den,out=np.full_like(den,np.nan,dtype=float),where=den>0)

def moments(d):
    x=d.copy(); x["y2"]=x.y_true**2; x["p2"]=x.pred**2; x["yp"]=x.y_true*x.pred
    g=x.groupby("speaker_id",sort=True).agg(my=("y_true","mean"),mp=("pred","mean"),my2=("y2","mean"),mp2=("p2","mean"),myp=("yp","mean"))
    o=g[["my","mp","my2","mp2","myp"]].to_numpy(float)
    b=np.column_stack([g.my,g.mp,g.my**2,g.mp**2,g.my*g.mp])
    w=np.column_stack([np.zeros(len(g)),np.zeros(len(g)),np.maximum(g.my2-g.my**2,0),np.maximum(g.mp2-g.mp**2,0),g.myp-g.my*g.mp])
    return list(g.index.astype(str)),{"overall":o,"between":b,"within":w}

def point(m): return float(ccc(m.mean(0)))

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def qci(x): return [float(v) for v in np.nanquantile(x,[.025,.975])]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); root=Path(cfg["output_root"])
    if root.exists() and any(root.iterdir()): raise RuntimeError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True,exist_ok=True)
    lam=float(cfg["lambda_filter"]); reps=int(cfg["bootstrap_reps"]); seed=int(cfg["bootstrap_seed"])
    pr=pd.read_parquet(cfg["pitch_rate_oof"]); pr=pr[(pr.dataset=="iemocap")&np.isclose(pr["lambda"].astype(float),lam)].copy()
    w=pd.read_parquet(cfg["wavlm_oof"]); w=w[np.isclose(w["lambda"].astype(float),lam)].copy()
    tables={}
    for fam,name in [("linear_ridge","pitch_rate_linear"),("quadratic_ridge","pitch_rate_quadratic")]:
        z=pr[pr.model_family==fam][["sample_id","speaker_id","target","y_true","pred_absolute","fold"]].rename(columns={"pred_absolute":"pred"}); tables[name]=z
    for layer in [12,24]:
        name=f"wavlm_layer{layer}"; z=w[w.layer==layer][["sample_id","speaker_id","target","y_true","pred_absolute","fold"]].rename(columns={"pred_absolute":"pred"}); tables[name]=z
    for target,path in cfg["emotion2vec_oof"].items():
        z=pd.read_parquet(path)[["sample_id","speaker_id","target","y_true","pred","fold"]].copy(); tables.setdefault("emotion2vec_plus_large",[]); tables["emotion2vec_plus_large"].append(z)
    tables["emotion2vec_plus_large"]=pd.concat(tables["emotion2vec_plus_large"],ignore_index=True)

    align_rows=[]; metric_rows=[]; delta_rows=[]; contrast_rows=[]; rank_rows=[]; fold_rows=[]
    rng=np.random.default_rng(seed)
    for target in cfg["targets"]:
        ts={k:v[v.target==target].copy() for k,v in tables.items()}
        common=set.intersection(*[set(v.sample_id.astype(str)) for v in ts.values()])
        if not common: raise RuntimeError(f"no common samples for {target}")
        ref=None; aligned={}; foldmaps={}
        for name,z in ts.items():
            z=z[z.sample_id.astype(str).isin(common)].copy().sort_values("sample_id")
            if z.sample_id.duplicated().any(): raise RuntimeError(f"duplicate {name}/{target}")
            key=z[["sample_id","speaker_id","y_true"]].reset_index(drop=True)
            if ref is None: ref=key
            else:
                if not np.array_equal(key.sample_id.to_numpy(),ref.sample_id.to_numpy()): raise RuntimeError("sample mismatch")
                if not np.array_equal(key.speaker_id.astype(str).to_numpy(),ref.speaker_id.astype(str).to_numpy()): raise RuntimeError("speaker mismatch")
                if not np.allclose(key.y_true.to_numpy(float),ref.y_true.to_numpy(float),atol=1e-8,rtol=0): raise RuntimeError("truth mismatch")
            sp,mm=moments(z); aligned[name]=(sp,mm)
            fm=z.groupby("speaker_id")["fold"].agg(lambda s:int(s.iloc[0]))
            if z.groupby("speaker_id")["fold"].nunique().max()!=1: raise RuntimeError(f"speaker crosses folds {name}")
            foldmaps[name]=fm.sort_index()
        speakers=next(iter(aligned.values()))[0]
        if any(v[0]!=speakers for v in aligned.values()): raise RuntimeError("speaker-set mismatch")
        ns=len(speakers); counts=rng.multinomial(ns,np.full(ns,1/ns),size=reps).astype(float)
        boot={}
        for name,(sp,mm) in aligned.items():
            boot[name]={}
            for metric in ["overall","between","within"]:
                vals=ccc((counts@mm[metric])/ns); boot[name][metric]=vals; lo,hi=qci(vals)
                metric_rows.append(dict(target=target,model=name,metric=metric,ccc=point(mm[metric]),ci95_low=lo,ci95_high=hi,n_speakers=ns,n_samples=len(common)))
        for name in sorted(foldmaps):
            fold_rows.append(dict(target=target,model=name,fold_map=";".join(f"{s}:{int(v)}" for s,v in foldmaps[name].items())))
        base=foldmaps["wavlm_layer12"]
        if not foldmaps["emotion2vec_plus_large"].equals(base): raise RuntimeError("emotion2vec/WavLM fold maps differ")
        align_rows.append(dict(target=target,common_samples=len(common),n_speakers=ns,fold_map_match_emotion2vec_wavlm=True,**{f"n_{k}":len(v) for k,v in ts.items()}))
        for metric in ["overall","between","within"]:
            scores={n:point(v[1][metric]) for n,v in aligned.items()}
            rank_rows.append(dict(target=target,metric=metric,ranking=" > ".join(sorted(scores,key=scores.get,reverse=True)),**scores))
            for x,y in itertools.combinations(sorted(aligned),2):
                dv=boot[x][metric]-boot[y][metric]; lo,hi=qci(dv)
                delta_rows.append(dict(target=target,metric=metric,comparison=f"{x}-{y}",delta_ccc=scores[x]-scores[y],ci95_low=lo,ci95_high=hi,ci_excludes_zero=bool(lo>0 or hi<0)))
        for x,y in itertools.combinations(sorted(aligned),2):
            for cname,m1,m2 in [("between_minus_within","between","within"),("overall_minus_within","overall","within")]:
                dv=(boot[x][m1]-boot[y][m1])-(boot[x][m2]-boot[y][m2]); lo,hi=qci(dv)
                pt=(point(aligned[x][1][m1])-point(aligned[y][1][m1]))-(point(aligned[x][1][m2])-point(aligned[y][1][m2]))
                contrast_rows.append(dict(target=target,comparison=f"{x}-{y}",contrast=cname,delta_of_deltas=pt,ci95_low=lo,ci95_high=hi,ci_excludes_zero=bool(lo>0 or hi<0)))

    pd.DataFrame(align_rows).to_csv(root/"sample_alignment.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(root/"fold_alignment.csv",index=False)
    pd.DataFrame(metric_rows).to_csv(root/"component_metrics.csv",index=False)
    pd.DataFrame(delta_rows).to_csv(root/"pairwise_component_deltas.csv",index=False)
    pd.DataFrame(contrast_rows).to_csv(root/"pairwise_profile_contrasts.csv",index=False)
    pd.DataFrame(rank_rows).to_csv(root/"rank_summary.csv",index=False)
    sources=[cfg["pitch_rate_oof"],cfg["wavlm_oof"],*cfg["emotion2vec_oof"].values()]
    meta={"experiment_id":cfg["experiment_id"],"prediction_condition":"absolute","relative_role":"diagnostic_only","bootstrap_unit":"speaker","bootstrap_reps":reps,"bootstrap_seed":seed,"source_sha256":{p:sha256(p) for p in sources},"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    (root/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(pd.DataFrame(metric_rows).to_string(index=False)); print("\nPAIRWISE SIGNIFICANT"); print(pd.DataFrame(delta_rows).query("ci_excludes_zero").to_string(index=False)); print("\nPROFILE CONTRASTS SIGNIFICANT"); print(pd.DataFrame(contrast_rows).query("ci_excludes_zero").to_string(index=False))

if __name__=="__main__": main()

