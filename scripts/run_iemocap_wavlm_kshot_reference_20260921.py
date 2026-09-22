#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def speaker_id(x):
    m=re.match(r'(Ses\d\d)[FM]_.+_([FM])\d+\.wav$',str(x))
    if not m: raise ValueError(x)
    return m.group(1)+'_'+m.group(2)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def equal_speaker_weights(ids):
    s=pd.Series(ids,dtype=str)
    c=s.value_counts()
    w=s.map(lambda x:1.0/c[x]).to_numpy(float)
    return w/w.mean()

def weighted_ccc(y,p,w):
    y=np.asarray(y,float); p=np.asarray(p,float); w=np.asarray(w,float); w=w/w.sum()
    my=np.sum(w*y); mp=np.sum(w*p)
    vy=np.sum(w*(y-my)**2); vp=np.sum(w*(p-mp)**2); cov=np.sum(w*(y-my)*(p-mp))
    den=vy+vp+(my-mp)**2
    return float(2*cov/den) if den>0 else np.nan

def fit_predict(xtr,xte,ytr,wtr,alpha):
    sc=StandardScaler()
    sc.fit(xtr,sample_weight=wtr)
    model=Ridge(alpha=alpha)
    model.fit(sc.transform(xtr),ytr,sample_weight=wtr)
    return model.predict(sc.transform(xte))

def per_speaker_moments(frame,speakers,pred_col):
    rows=[]
    for sp in speakers:
        z=frame[frame.speaker_id==sp]
        y=z.y_true.to_numpy(float); p=z[pred_col].to_numpy(float)
        rows.append([y.mean(),np.mean(y*y),p.mean(),np.mean(p*p),np.mean(y*p)])
    return np.asarray(rows,float)

def ccc_from_moments(a):
    my=a[:,0]; ey2=a[:,1]; mp=a[:,2]; ep2=a[:,3]; eyp=a[:,4]
    vy=ey2-my*my; vp=ep2-mp*mp; cov=eyp-my*mp
    den=vy+vp+(my-mp)**2
    out=np.full(len(my),np.nan,float); ok=den>0
    out[ok]=2*cov[ok]/den[ok]
    return out

def effect_from_counts(counts,abs_stats,rel_stats,S):
    w=counts.astype(float)/float(S)
    a=w@abs_stats; r=w@rel_stats
    return ccc_from_moments(r)-ccc_from_moments(a)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    pars=cfg["parameters"]

    labs=[]
    for f in sorted(glob.glob(cfg["dataset"]["source_glob"])):
        labs.append(pd.read_parquet(f,columns=["file","EmoAct","EmoDom"]))
    d=pd.concat(labs,ignore_index=True).dropna().copy()
    d["sample_id"]=d.file.astype(str)
    d["speaker_id"]=d.file.map(speaker_id)
    d=d.rename(columns={"EmoAct":"arousal","EmoDom":"dominance"})

    ids=[]; blocks=[]
    for f in sorted(Path(cfg["inputs"]["embedding_root"]).glob("*.npz")):
        z=np.load(f)
        ids.extend(z["sample_id"].astype(str).tolist())
        blocks.append(z["embedding"].astype(np.float32))
    emb=np.concatenate(blocks,axis=0)
    em=pd.DataFrame({"sample_id":ids,"row_idx":np.arange(len(ids))})
    d=d.merge(em,on="sample_id",validate="one_to_one")
    if len(d)!=10039: raise RuntimeError(f"unexpected rows {len(d)}")

    speakers=sorted(d.speaker_id.unique())
    min_n=int(d.groupby("speaker_id").size().min())
    pool=int(pars["enrollment_pool_size"])
    if min_n<=pool: raise RuntimeError(f"speaker has <= enrollment pool rows: {min_n}")

    fmap=speaker_folds(speakers,int(pars["outer_folds"]),int(pars["split_seed"]))
    folds=d.speaker_id.map(fmap).to_numpy()
    lambdas=np.asarray(pars["lambda_values"],float)
    k_values=list(map(int,pars["k_values"]))
    targets=list(pars["targets"])
    layers=list(map(int,pars["layers"]))
    alpha=float(pars["ridge_alpha"])
    seed_rows=[]

    # Results are stored at utterance level only temporarily per enrollment seed.
    all_frames={}
    oracle_frames={}

    for eseed in map(int,pars["enrollment_seeds"]):
        enroll_idx={}
        downstream_mask=np.ones(len(d),dtype=bool)
        for sp in speakers:
            idx=np.flatnonzero(d.speaker_id.to_numpy()==sp)
            rng=np.random.default_rng(stable_seed("enroll",eseed,sp))
            order=idx.copy(); rng.shuffle(order)
            chosen=order[:pool]
            enroll_idx[sp]=chosen
            downstream_mask[chosen]=False

        ds_idx=np.flatnonzero(downstream_mask)
        dd=d.iloc[ds_idx].reset_index(drop=True)
        fold_ds=folds[ds_idx]

        # Target components use downstream labels only; enrollment labels are not used.
        target_endpoints={}
        for t in targets:
            raw=dd[t].to_numpy(float)
            spmean=dd.groupby("speaker_id")[t].mean()
            sm=dd.speaker_id.map(spmean).to_numpy(float)
            gm=float(raw.mean())
            within=raw-sm
            target_endpoints[t]=(gm+within, raw)

        for lpos,layer in enumerate(layers):
            Hfull=emb[d.row_idx.to_numpy(),lpos,:]
            H=Hfull[ds_idx]
            # Oracle center is diagnostic full-speaker median.
            oracle_centers={sp:np.median(Hfull[np.flatnonzero(d.speaker_id.to_numpy()==sp)],axis=0)
                            for sp in speakers}
            Coracle=np.vstack([oracle_centers[sp] for sp in dd.speaker_id])
            Roracle=H-Coracle

            # Absolute endpoint predictions are independent of K within an enrollment seed.
            abs_pred={t:{} for t in targets}
            oracle_pred={t:{} for t in targets}
            for fold in range(int(pars["outer_folds"])):
                tr=fold_ds!=fold; te=fold_ds==fold
                wtr=equal_speaker_weights(dd.loc[tr,"speaker_id"].to_numpy())
                Y0=np.column_stack([target_endpoints[t][0] for t in targets])
                Y1=np.column_stack([target_endpoints[t][1] for t in targets])
                pa0=fit_predict(H[tr],H[te],Y0[tr],wtr,alpha)
                pa1=fit_predict(H[tr],H[te],Y1[tr],wtr,alpha)
                po0=fit_predict(Roracle[tr],Roracle[te],Y0[tr],wtr,alpha)
                po1=fit_predict(Roracle[tr],Roracle[te],Y1[tr],wtr,alpha)
                test_pos=np.flatnonzero(te)
                for j,t in enumerate(targets):
                    abs_pred[t][fold]=(test_pos,pa0[:,j],pa1[:,j])
                    oracle_pred[t][fold]=(test_pos,po0[:,j],po1[:,j])

            # Oracle utterance frames for same downstream pool.
            for t in targets:
                rec=[]
                y0,y1=target_endpoints[t]
                for fold in range(int(pars["outer_folds"])):
                    pos,a0,a1=abs_pred[t][fold]
                    _,o0,o1=oracle_pred[t][fold]
                    for q,ii in enumerate(pos):
                        rec.append((ii,fold,a0[q],a1[q],o0[q],o1[q],y0[ii],y1[ii]))
                oracle_frames[(eseed,layer,t)]=pd.DataFrame(
                    rec,columns=["pos","fold","abs0","abs1","rel0","rel1","y0","y1"])

            for K in k_values:
                centers={sp:np.median(Hfull[enroll_idx[sp][:K]],axis=0) for sp in speakers}
                C=np.vstack([centers[sp] for sp in dd.speaker_id])
                R=H-C
                rel_pred={t:{} for t in targets}
                for fold in range(int(pars["outer_folds"])):
                    tr=fold_ds!=fold; te=fold_ds==fold
                    wtr=equal_speaker_weights(dd.loc[tr,"speaker_id"].to_numpy())
                    Y0=np.column_stack([target_endpoints[t][0] for t in targets])
                    Y1=np.column_stack([target_endpoints[t][1] for t in targets])
                    pr0=fit_predict(R[tr],R[te],Y0[tr],wtr,alpha)
                    pr1=fit_predict(R[tr],R[te],Y1[tr],wtr,alpha)
                    test_pos=np.flatnonzero(te)
                    for j,t in enumerate(targets):
                        rel_pred[t][fold]=(test_pos,pr0[:,j],pr1[:,j])

                for t in targets:
                    rows=[]
                    y0,y1=target_endpoints[t]
                    for fold in range(int(pars["outer_folds"])):
                        pos,a0,a1=abs_pred[t][fold]
                        _,r0,r1=rel_pred[t][fold]
                        for q,ii in enumerate(pos):
                            rows.append({
                              "enrollment_seed":eseed,"K":K,"layer":layer,"target":t,
                              "speaker_id":dd.iloc[ii].speaker_id,"sample_id":dd.iloc[ii].sample_id,
                              "y0":y0[ii],"y1":y1[ii],
                              "abs0":a0[q],"abs1":a1[q],"rel0":r0[q],"rel1":r1[q]
                            })
                    all_frames[(eseed,K,layer,t)]=pd.DataFrame(rows)
                seed_rows.append({"enrollment_seed":eseed,"K":K,"layer":layer,
                                  "downstream_rows":len(dd),"speakers":len(speakers)})

    # Effect curves per enrollment seed/K/layer/target plus oracle.
    curves=[]
    oracle_curves=[]
    for eseed in map(int,pars["enrollment_seeds"]):
      for layer in layers:
        for t in targets:
          # Oracle
          fr=oracle_frames[(eseed,layer,t)].copy()
          spids=d.iloc[np.flatnonzero(np.ones(len(d),dtype=bool))]  # unused placeholder
          # map downstream position back through all_frames shared structure
          ref=all_frames[(eseed,k_values[0],layer,t)]
          fr["speaker_id"]=ref.sort_values("sample_id").speaker_id.to_numpy() if False else ref.speaker_id.to_numpy()
          for lam in lambdas:
            y=(1-lam)*fr.y0.to_numpy()+lam*fr.y1.to_numpy()
            pa=(1-lam)*fr.abs0.to_numpy()+lam*fr.abs1.to_numpy()
            pr=(1-lam)*fr.rel0.to_numpy()+lam*fr.rel1.to_numpy()
            w=equal_speaker_weights(fr.speaker_id.to_numpy())
            eff=weighted_ccc(y,pr,w)-weighted_ccc(y,pa,w)
            oracle_curves.append({"enrollment_seed":eseed,"layer":layer,"target":t,
                                  "lambda":lam,"relative_minus_absolute":eff})
          for K in k_values:
            fr=all_frames[(eseed,K,layer,t)]
            for lam in lambdas:
                y=(1-lam)*fr.y0.to_numpy()+lam*fr.y1.to_numpy()
                pa=(1-lam)*fr.abs0.to_numpy()+lam*fr.abs1.to_numpy()
                pr=(1-lam)*fr.rel0.to_numpy()+lam*fr.rel1.to_numpy()
                w=equal_speaker_weights(fr.speaker_id.to_numpy())
                eff=weighted_ccc(y,pr,w)-weighted_ccc(y,pa,w)
                curves.append({"enrollment_seed":eseed,"K":K,"layer":layer,"target":t,
                               "lambda":lam,"relative_minus_absolute":eff})

    curves=pd.DataFrame(curves)
    oracle_curves=pd.DataFrame(oracle_curves)

    # Summary slopes and speaker bootstrap. For bootstrap, use each seed-specific utterance frame
    # and average seed-level effects within each replicate.
    summaries=[]
    B=int(pars["bootstrap_reps"])
    x=lambdas; xc=x-x.mean(); denom=float(xc@xc)
    for K in k_values:
      for layer in layers:
        for t in targets:
          seed_effects=[]
          for eseed in map(int,pars["enrollment_seeds"]):
            g=curves[(curves.enrollment_seed==eseed)&(curves.K==K)&
                     (curves.layer==layer)&(curves.target==t)].sort_values("lambda")
            seed_effects.append(g.relative_minus_absolute.to_numpy(float))
          point=np.mean(np.vstack(seed_effects),axis=0)
          point_slope=float(point@xc/denom)
          rng=np.random.default_rng(stable_seed("boot",K,layer,t))
          bs_s=np.empty(B); bs_l0=np.empty(B)
          pvec=np.full(len(speakers),1.0/len(speakers))
          for b in range(B):
            counts=rng.multinomial(len(speakers),pvec)
            seed_curves=[]
            for eseed in map(int,pars["enrollment_seeds"]):
                fr=all_frames[(eseed,K,layer,t)]
                vals=[]
                for lam in lambdas:
                    y=(1-lam)*fr.y0.to_numpy()+lam*fr.y1.to_numpy()
                    pa=(1-lam)*fr.abs0.to_numpy()+lam*fr.abs1.to_numpy()
                    pr=(1-lam)*fr.rel0.to_numpy()+lam*fr.rel1.to_numpy()
                    # per-speaker moments
                    tmp=fr[["speaker_id"]].copy()
                    tmp["y"]=y; tmp["pa"]=pa; tmp["pr"]=pr
                    arr=[]
                    for sp in speakers:
                        z=tmp[tmp.speaker_id==sp]
                        yy=z.y.to_numpy(float); aa=z.pa.to_numpy(float); rr=z.pr.to_numpy(float)
                        arr.append([yy.mean(),np.mean(yy*yy),aa.mean(),np.mean(aa*aa),np.mean(yy*aa),
                                    rr.mean(),np.mean(rr*rr),np.mean(yy*rr)])
                    arr=np.asarray(arr,float)
                    w=counts.astype(float)/len(speakers)
                    a=w@arr
                    def c(myp):
                        my,ey2,mp,ep2,eyp=myp
                        vy=ey2-my*my; vp=ep2-mp*mp; cov=eyp-my*mp
                        den2=vy+vp+(my-mp)**2
                        return 2*cov/den2 if den2>0 else np.nan
                    vals.append(c([a[0],a[1],a[5],a[6],a[7]])-
                                c([a[0],a[1],a[2],a[3],a[4]]))
                seed_curves.append(vals)
            eff=np.mean(np.asarray(seed_curves,float),axis=0)
            bs_s[b]=eff@xc/denom
            bs_l0[b]=eff[0]
          slo,shi=np.quantile(bs_s,[.025,.975]); elo,ehi=np.quantile(bs_l0,[.025,.975])
          summaries.append({"K":K,"layer":layer,"target":t,
                            "slope_mean":point_slope,"slope_ci95_low":float(slo),"slope_ci95_high":float(shi),
                            "lambda0_effect":float(point[0]),"lambda0_ci95_low":float(elo),
                            "lambda0_ci95_high":float(ehi)})

    summary=pd.DataFrame(summaries)

    # Oracle slope averaged across enrollment seeds for the same downstream pools.
    oracle_sum=[]
    for layer in layers:
      for t in targets:
        es=[]
        for eseed in map(int,pars["enrollment_seeds"]):
            g=oracle_curves[(oracle_curves.enrollment_seed==eseed)&
                            (oracle_curves.layer==layer)&(oracle_curves.target==t)].sort_values("lambda")
            es.append(g.relative_minus_absolute.to_numpy(float))
        eff=np.mean(np.vstack(es),axis=0)
        oracle_sum.append({"layer":layer,"target":t,"oracle_slope":float(eff@xc/denom),
                           "oracle_lambda0":float(eff[0])})
    oracle_sum=pd.DataFrame(oracle_sum)
    comp=summary.merge(oracle_sum,on=["layer","target"])
    comp["slope_gap_abs"]=(comp.slope_mean-comp.oracle_slope).abs()
    comp["lambda0_gap_abs"]=(comp.lambda0_effect-comp.oracle_lambda0).abs()

    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{"rows":len(d),"speakers":len(speakers),"min_rows_per_speaker":min_n,
                   "enrollment_pool_size":pool,"enrollment_seeds":len(pars["enrollment_seeds"])}]).to_csv(
        root/"data_inventory.csv",index=False)
    curves.to_csv(root/"effect_curves.csv",index=False)
    oracle_curves.to_csv(root/"oracle_effect_curves.csv",index=False)
    summary.to_csv(root/"slope_summary.csv",index=False)
    comp.to_csv(root/"oracle_comparison.csv",index=False)
    pd.DataFrame(seed_rows).drop_duplicates().to_csv(root/"enrollment_inventory.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"layers":layers,"targets":targets,
      "k_values":k_values,"enrollment_pool_size":pool,
      "enrollment_seeds":pars["enrollment_seeds"],"bootstrap_unit":"speaker",
      "lambda_interpolation":"exact Ridge endpoint interpolation"
    },indent=2)+"\n")
    print(summary.to_string(index=False))
    print("\\nORACLE COMPARISON")
    print(comp.to_string(index=False))

if __name__=="__main__":
    main()
