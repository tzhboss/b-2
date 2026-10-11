"""Common strict-ranking transition metrics, independent of model architecture."""
import numpy as np

def transitions(before,after,score_before=None,score_after=None,std_before=None):
    b=np.asarray(before,dtype=np.int64);a=np.asarray(after,dtype=np.int64)
    assert b.shape==a.shape and b.ndim==1 and len(b)>0 and np.all(b>=1) and np.all(a>=1)
    old=b<=10;loss=old&(a>10);delta=a-b;w=old&(delta>0)
    def quant(x,p):return int(np.quantile(x,p,method='higher')) if len(x) else None
    counts=dict(queries=len(b),old_top10=int(old.sum()),lost_top10=int(loss.sum()),unchanged=int((b==a).sum()),worsened=int((a>b).sum()),improved=int((a<b).sum()))
    scalar=dict.fromkeys(counts,0)
    for x,y in zip(b.tolist(),a.tolist()):
        scalar['queries']+=1;scalar['old_top10']+=int(x<=10);scalar['lost_top10']+=int(x<=10<y)
        scalar['unchanged']+=int(x==y);scalar['worsened']+=int(y>x);scalar['improved']+=int(y<x)
    assert scalar==counts and counts['queries']==counts['unchanged']+counts['worsened']+counts['improved']
    d=dict(**counts,old_top10_loss_fraction=float(loss.sum()/old.sum()) if old.any() else None,
        old_top10_delta_p50=quant(delta[old],.5),old_top10_delta_p95=quant(delta[old],.95),
        worsened_old_top10_n=int(w.sum()),worsened_old_top10_loss_p50=quant(delta[w],.5),worsened_old_top10_loss_p95=quant(delta[w],.95),
        worsened_old_top10_loss_max=int(delta[w].max()) if w.any() else None,
        before_hits10=float(old.mean()),after_hits10=float((a<=10).mean()))
    if score_before is not None:
        sb=np.asarray(score_before);sa=np.asarray(score_after);sd=np.asarray(std_before)
        assert sb.shape==sa.shape==sd.shape==b.shape and np.isfinite(sb).all() and np.isfinite(sa).all() and np.isfinite(sd).all()
        valid=sd>0;normalized=(sa-sb)/np.maximum(sd,np.finfo(float).tiny)
        small=old&valid&(np.abs(normalized)<=1e-3)
        d.update(old_top10_small_standardized_score_drift_n=int(small.sum()),small_score_drift_lost_top10_n=int((small&loss).sum()),
          small_score_drift_loss_fraction=float((small&loss).sum()/small.sum()) if small.any() else None,
          old_top10_abs_standardized_score_drift_p50=float(np.quantile(np.abs(normalized[old&valid]),.5)) if np.any(old&valid) else None)
    return d
