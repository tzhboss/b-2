#!/usr/bin/env python3
from __future__ import annotations
import argparse,glob,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_iemocap_absolute_relative_hybrid_component_20260924 import speaker_id,fold_map,weights,load_simple,load_wavlm
from run_iemocap_paired_crossmodel_component_attribution_20260924 import ccc,moments,point,qci

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); root=Path(cfg["output_root"])
    if root.exists() and any(root.iterdir()): raise RuntimeError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True,exist_ok=True)
    labs=[pd.read_parquet(f,columns=["file","EmoAct","EmoDom"]) for f in sorted(glob.glob(cfg["dataset_glob"]))]
    d=pd.concat(labs,ignore_index=True).dropna().copy(); d["sample_id"]=d.file.astype(str); d["speaker_id"]=d.file.map(speaker_id)
    d=d.rename(columns={"EmoAct":"arousal","EmoDom":"dominance"}); speakers=sorted(d.speaker_id.unique())
    fmap=fold_map(speakers,int(cfg["outer_folds"]),int(cfg["split_seed"]))
    models=load_wavlm(cfg["embeddings"]["wavlm"])
    for name,pat in cfg["embeddings"].items():
        if name!="wavlm": models[name]=load_simple(pat)
    alphas=[float(x) for x in cfg["alphas"]]; methods=["absolute","relative","hybrid"]; metrics=["overall","between","within"]
    point_rows=[]; delta_rows=[]; contrast_rows=[]; stability=[]; rng=np.random.default_rng(int(cfg["bootstrap_seed"]))
    for model_name,(em,H0) in models.items():
        q=d.merge(em,on="sample_id",validate="one_to_one")
        if len(q)!=10039: raise RuntimeError(f"coverage {model_name}")
        H=H0[q.row.to_numpy()]; sf=q.speaker_id.map(fmap).to_numpy(); Y=q[["arousal","dominance"]].to_numpy(float)
        centers={sp:np.median(H[q.speaker_id.to_numpy()==sp],axis=0) for sp in speakers}; C=np.vstack([centers[sp] for sp in q.speaker_id]); R=H-C
        feats={"absolute":H,"relative":R,"hybrid":np.concatenate([R,C],axis=1)}
        pred={(alpha,m):np.empty_like(Y) for alpha in alphas for m in methods}
        for fold in range(int(cfg["outer_folds"])):
            tr=sf!=fold; te=sf==fold; w=weights(q.loc[tr])
            for meth,X in feats.items():
                sc=StandardScaler(); sc.fit(X[tr],sample_weight=w); xtr=sc.transform(X[tr]); xte=sc.transform(X[te])
                for alpha in alphas:
                    mdl=Ridge(alpha=alpha); mdl.fit(xtr,Y[tr],sample_weight=w); pred[(alpha,meth)][te]=mdl.predict(xte)
        for j,target in enumerate(["arousal","dominance"]):
            ns=len(speakers); counts=rng.multinomial(ns,np.full(ns,1/ns),size=int(cfg["bootstrap_reps"])).astype(float); store={}
            for alpha in alphas:
                store[alpha]={}
                for meth in methods:
                    z=pd.DataFrame({"speaker_id":q.speaker_id,"y_true":Y[:,j],"pred":pred[(alpha,meth)][:,j]})
                    sp,mm=moments(z); store[alpha][meth]=mm
                    for metric in metrics:
                        vals=ccc((counts@mm[metric])/ns); lo,hi=qci(vals)
                        point_rows.append({"model":model_name,"target":target,"alpha":alpha,"method":meth,"metric":metric,"ccc":point(mm[metric]),"ci95_low":lo,"ci95_high":hi})
                for comp,x,y in [("relative_minus_absolute","relative","absolute"),("hybrid_minus_relative","hybrid","relative"),("hybrid_minus_absolute","hybrid","absolute")]:
                    for metric in metrics:
                        dv=ccc((counts@store[alpha][x][metric])/ns)-ccc((counts@store[alpha][y][metric])/ns); lo,hi=qci(dv)
                        pt=point(store[alpha][x][metric])-point(store[alpha][y][metric])
                        delta_rows.append({"model":model_name,"target":target,"alpha":alpha,"comparison":comp,"metric":metric,"delta_ccc":pt,"ci95_low":lo,"ci95_high":hi,"ci_excludes_zero":bool(lo>0 or hi<0)})
                    dv=(ccc((counts@store[alpha][x]["between"])/ns)-ccc((counts@store[alpha][y]["between"])/ns))-(ccc((counts@store[alpha][x]["within"])/ns)-ccc((counts@store[alpha][y]["within"])/ns)); lo,hi=qci(dv)
                    pt=(point(store[alpha][x]["between"])-point(store[alpha][y]["between"]))-(point(store[alpha][x]["within"])-point(store[alpha][y]["within"]))
                    contrast_rows.append({"model":model_name,"target":target,"alpha":alpha,"comparison":comp,"contrast":"between_minus_within","delta_of_deltas":pt,"ci95_low":lo,"ci95_high":hi,"ci_excludes_zero":bool(lo>0 or hi<0)})
            for comp in ["relative_minus_absolute","hybrid_minus_relative","hybrid_minus_absolute"]:
                rows=[x for x in contrast_rows if x["model"]==model_name and x["target"]==target and x["comparison"]==comp]
                vals=[x["delta_of_deltas"] for x in rows]
                od=[x for x in delta_rows if x["model"]==model_name and x["target"]==target and x["comparison"]==comp and x["metric"]=="overall"]
                ovs=[x["delta_ccc"] for x in od]
                stability.append({"model":model_name,"target":target,"comparison":comp,"profile_all_negative":all(v<0 for v in vals),"profile_all_positive":all(v>0 for v in vals),"profile_sign_changes":len(set(np.sign(vals)))>1,"overall_all_nonpositive":all(v<=0 for v in ovs),"overall_all_nonnegative":all(v>=0 for v in ovs),"overall_sign_changes":len(set(np.sign(ovs)))>1,"profile_min":min(vals),"profile_max":max(vals),"overall_min":min(ovs),"overall_max":max(ovs)})
    pd.DataFrame(point_rows).to_csv(root/"alpha_component_metrics.csv",index=False)
    pd.DataFrame(delta_rows).to_csv(root/"alpha_method_deltas.csv",index=False)
    pd.DataFrame(contrast_rows).to_csv(root/"alpha_profile_contrasts.csv",index=False)
    pd.DataFrame(stability).to_csv(root/"sign_stability.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({"experiment_id":cfg["experiment_id"],"alphas":alphas,"task":"raw absolute VAD","bootstrap_unit":"speaker","bootstrap_reps":cfg["bootstrap_reps"],"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},indent=2)+"\n")
    print(pd.DataFrame(stability).to_string(index=False))
if __name__=="__main__": main()

