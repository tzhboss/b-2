import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/risk_link136_20261014')
import experiment136 as risk
worker,base,core,cx=risk.worker,risk.base,risk.core,risk.cx
ROOT=Path(__file__).resolve().parent

def status(stage,**kw):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

@torch.no_grad()
def diagnose(parent,n,initial,prefix,endpoints,queries,pred,truths,gate,out):
    tick=time.monotonic()
    ids=pred['ids']
    facts=queries[ids]
    E,R=initial['E'].cuda().double(),initial['R'].cuda().double()
    prefixrows=prefix['E'][n:].cuda().double()
    models={k:v['E'][n:].cuda().double() for k,v in endpoints.items()}
    d=gate['degree']
    groups=dict(one=d==1,two_three=(d>=2)&(d<=3),four_plus=d>=4,
                few=d<=3,no_train_conflict=~gate['risk'],unknown=~gate['query_support'],selected_protection=gate['action']>0)
    masks={k:torch.as_tensor(v,device='cuda') for k,v in groups.items()}
    prefixcount=[]
    endpointcount={k:[] for k in models}
    counts={arm:{group:[] for group in groups} for arm in models}
    newpairs={arm:{group:0 for group in groups} for arm in models}
    rescuedpairs={arm:{group:0 for group in groups} for arm in models}
    affectedrows={arm:np.zeros(len(d),bool) for arm in models}
    for lo in range(0,len(facts),128):
        b=facts[lo:lo+128]
        q=risk.query(E,R,b,parent=='WN_ComplEx')
        gs=torch.as_tensor(pred['gold'][lo:lo+len(b)],device='cuda')
        before=q@prefixrows.T>gs[:,None]
        legal=torch.ones_like(before)
        for j,(h,r,g) in enumerate(b):
            cols=[v-n for v in truths[int(h),int(r)] if v>=n]
            if cols:legal[j,cols]=False
        before&=legal
        prefixcount.append(before.sum(1).cpu().numpy())
        for arm,rows in models.items():
            after=(q@rows.T>gs[:,None])&legal
            endpointcount[arm].append(after.sum(1).cpu().numpy())
            affectedrows[arm]|=after.any(0).cpu().numpy()
            new=after&~before
            rescued=before&~after
            for group,mask in masks.items():
                counts[arm][group].append(after[:,mask].sum(1).cpu().numpy())
                newpairs[arm][group]+=int(new[:,mask].sum())
                rescuedpairs[arm][group]+=int(rescued[:,mask].sum())
    prefixcount=np.concatenate(prefixcount)
    oldrank=pred['oldrank']
    prefixrank=oldrank+prefixcount
    reports={}
    output=dict(ids=ids,prefix_rank=prefixrank,oldrank=oldrank)
    cached_feature=base.load(f'/data/lc/untzh/risk_link136_20261014/{parent}/cache/old_valid-features.npz')
    assert np.array_equal(prefixrank,cached_feature['prefix_rank'])
    for arm in models:
        total=np.concatenate(endpointcount[arm])
        final=oldrank+total
        rankpath=Path('/data/lc/untzh/entity_action137_20261014')/str(parent)/'evaluation'/f'{arm}.npz'
        cached=base.load(rankpath)['old_valid'][ids,1]
        assert np.array_equal(final,cached)
        drop=final>10
        firstdrop=prefixrank>10
        row={}
        for group,mask in groups.items():
            contribution=np.concatenate(counts[arm][group])
            without=final-contribution
            row[group]=dict(entities=int(mask.sum()),entities_crossing_any_valid=int((affectedrows[arm]&mask).sum()),
                final_crossing_pairs=int(contribution.sum()),late_new_crossing_pairs=newpairs[arm][group],rescued_crossing_pairs=rescuedpairs[arm][group],
                queries_affected=int((contribution>0).sum()),dropped_queries_affected=int((drop&(contribution>0)).sum()),
                drops_resolved_by_candidate_removal=int((drop&(without<=10)).sum()),
                mean_counterfactual_rr_gain=float((1/without-1/final).mean()))
            output[f'{arm}-{group}-counts']=contribution
        reports[arm]=dict(eligible=len(ids),prefix_drops=int(firstdrop.sum()),final_drops=int(drop.sum()),
            late_new_drops=int((~firstdrop&drop).sum()),rescued_queries=int((firstdrop&~drop).sum()),groups=row,
            eligible_mrr=float((1/final).mean()),cached_rank_equal=True)
        output[arm+'-final_rank']=final
    out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'counts.npz',**output)
    return dict(models=reports,seconds=time.monotonic()-tick,
        diagnostic_candidate_removal_not_deployable=True,overlapping_groups_not_additive=True)

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic()
    reports={}
    for parent in [17,'WN_ComplEx']:
        status('prepare',parent=str(parent))
        n,total,src,new,tape,truths,_,_=worker.training_assets(parent)
        ip,pp,_,_,_=risk.previous.paths(parent)
        initial=torch.load(ip,map_location='cpu',weights_only=True)
        prefix=torch.load(pp,map_location='cpu',weights_only=True)
        endpoints={arm:torch.load(Path('/data/lc/untzh/entity_action137_20261014')/str(parent)/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True) for arm in ['none','fixed_anchor','fixed_replay']}
        queries=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['old_valid'] if parent==17 else base.load(src/'evaluation/queries.npz')['old_valid']
        pred=base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{parent}-old_valid-half-prediction.npz')
        gate=base.load(f'/data/lc/untzh/risk_link136_20261014/{parent}/cache/row-gate.npz')
        status('diagnose',parent=str(parent))
        report=diagnose(parent,n,initial,prefix,endpoints,queries,pred,truths,gate,ROOT/str(parent))
        reports[str(parent)]=report
        base.write(ROOT/str(parent)/'summary.json',report)
        print('LOWDEGREE138',parent,{arm:{group:(g['dropped_queries_affected'],g['drops_resolved_by_candidate_removal']) for group,g in m['groups'].items()} for arm,m in report['models'].items()},flush=True)
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,new_training_configs=0,
        decision='NO-GO current TRAIN-pressure row gate; diagnostic only'))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
