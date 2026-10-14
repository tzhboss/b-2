import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/admission_risk124_20261014')
import experiment124 as prior
cx,base,core=prior.cx,prior.base,prior.core
ROOT=Path(__file__).resolve().parent
SEEDS=[20261014,20261015,20261016]
MODES=[('standard',.12),('low',.03),('half',.015)]

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic();status('predict');(ROOT/'cache').mkdir(exist_ok=True);bundles=[];fixed=[]
    for parent in [43,17,29,'WN_DistMult','WN_ComplEx']:
        fb=isinstance(parent,int);complex_model=parent=='WN_ComplEx';n,total=(11258,13134) if fb else (24567,28660)
        src=Path('/data/lc/untzh')/({43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[parent] if fb else 'complex_transfer118_20261014' if complex_model else 'wn_transfer112_20261014')
        state=torch.load(src/'cache/initial.pt',map_location='cpu',weights_only=True)
        oldE,R=state['E'][:n].double().cuda(),state['R'].double().cuda();del state
        z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz' if fb else '/data/lc/untzh/wn_transfer112_20261014/cache/train.npz')
        truths=core.truthmap(z['old'],np.concatenate([z['new'],z['extra']]) if fb else z['new'])
        queries=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['old_valid'] if fb and parent!=43 else base.load(src/'evaluation/queries.npz')['old_valid']
        predictions={}
        for mode,sigma in MODES:
            p=base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{parent}-old_valid-{mode}-prediction.npz')
            np.savez(ROOT/'cache'/f'{parent}-{mode}-prediction.npz',**p);predictions[mode]=p
            fixed.append(dict(parent=str(parent),mode=mode,sigma=sigma,n=len(p['ids']),expected_drops=float(p['probability'].sum())))
        ids=predictions['low']['ids'];a=queries[ids];q=[]
        for lo in range(0,len(a),256):
            b=a[lo:lo+256];v=cx.tail_query(oldE[b[:,0]],R[b[:,1]]) if complex_model else oldE[b[:,0]]*R[b[:,1]];q.append(v.cpu())
        q=torch.cat(q);legal=np.ones((len(a),total-n),bool)
        for i,(h,r,g) in enumerate(a):legal[i,[c-n for c in truths[int(h),int(r)] if c>=n]]=False
        bundles.append((parent,total-n,q,legal,predictions))
    # Every forecast is immutable on disk before any new candidate realization is generated.
    base.write(ROOT/'frozen-predictions.json',dict(seeds=SEEDS,predictions=fixed,predicted_before_noise_generation=True))
    reports={};generation_cost={};score_cost={}
    for parent,newcount,q,legal,predictions in bundles:
        q=q.cuda();mask=torch.as_tensor(legal,device='cuda');g=torch.as_tensor(predictions['low']['gold'],device='cuda')
        capacity=10-predictions['low']['oldrank'];parent_reports={}
        for seed in SEEDS:
            status('generate',parent=str(parent),seed=seed);st=time.monotonic()
            # One standard-normal FP32 tensor per parent/replicate shared across fixed scales.
            noise=torch.randn((newcount,200),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
            torch.save(noise.cpu(),ROOT/'cache'/f'{parent}-noise-{seed}.pt');generation_cost[f'{parent}-{seed}']=time.monotonic()-st
            labels={mode:[] for mode,sigma in MODES};st=time.monotonic();status('score',parent=str(parent),seed=seed)
            scaled={mode:(noise*sigma).double() for mode,sigma in MODES}
            for lo in range(0,len(q),128):
                qb=q[lo:lo+128];gold=g[lo:lo+128];legalb=mask[lo:lo+128]
                for mode,sigma in MODES:
                    above=((qb@scaled[mode].T)>gold[:,None])&legalb
                    labels[mode].append(above.sum(-1).cpu().numpy())
            score_cost[f'{parent}-{seed}']=time.monotonic()-st;rep={}
            for mode,sigma in MODES:
                counts=np.concatenate(labels[mode]);y=counts>capacity;p=predictions[mode]
                scores=dict(gaussian_capacity=p['probability'],normal_tail=p['p'],negative_gold=-p['gold'],boundary=p['oldrank'].astype(float),random=np.random.default_rng(seed+17).random(len(y)))
                methods={k:prior.evaluate(y,v,k=='gaussian_capacity') for k,v in scores.items()}
                np.savez(ROOT/'cache'/f'{parent}-{seed}-{mode}-labels.npz',counts=counts,event=y)
                rep[mode]=dict(n=len(y),drops=int(y.sum()),methods=methods)
                print('FORWARD125',str(parent),seed,mode,'observed',int(y.sum()),'expected',float(p['probability'].sum()),'AP',methods['gaussian_capacity']['ap'],flush=True)
            parent_reports[str(seed)]=rep
        reports[str(parent)]=parent_reports;base.write(ROOT/'summary-running.json',dict(parents=reports))
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,generation_seconds=generation_cost,scoring_seconds=score_cost,
        predictions_saved_before_realizations=True,no_training=True,fixed_development_reference_queries=True,historical_assets_not_free=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
