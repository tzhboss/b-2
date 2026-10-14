import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/forecast_protect146_20261014')
import experiment146 as prior
base,worker,core,cx=prior.base,prior.worker,prior.core,prior.cx
ROOT=Path(__file__).resolve().parent
N,TOTAL=prior.N,prior.TOTAL
ARMS=prior.ARMS
DIRECTIONS=core.DIRECTIONS

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))
def sync():torch.cuda.synchronize();return time.monotonic()

def direction_losses(E,R,new,tape,j):
    lo,hi=tape['bounds'][j:j+2]
    p,nh,nt=[torch.as_tensor(a,device='cuda',dtype=torch.long) for a in
             (new[tape['pos'][lo:hi]],tape['nh'][lo:hi],tape['nt'][lo:hi])]
    h,r,t=p.unbind(1)
    ps=(E[h]*R[r]*E[t]).sum(-1)
    ns=(E[nh]*R[r[:,None]]*E[nt]).sum(-1)
    losses=F.softplus(-ps)+F.softplus(ns).mean(1)+.0005*(E[h].square().mean(1)+R[r].square().mean(1))
    masks=[(h<N)&(t>=N),(h>=N)&(t<N),(h>=N)&(t>=N)]
    assert torch.stack(masks).sum(0).eq(1).all()
    counts=[int(m.sum()) for m in masks]
    return {d:losses[m].mean() if c else None for d,m,c in zip(DIRECTIONS,masks,counts)},counts,len(p)

def cosine(a,b):
    aa,bb=a.double().flatten(),b.double().flatten()
    den=aa.norm()*bb.norm()
    return float(aa.dot(bb)/den) if den>0 else None

@torch.no_grad()
def probe(rows,ref):
    gap=ref['q']@rows.double().T-ref['gold'][:,None]
    rank=ref['oldrank']+((gap>0)&ref['legal']).sum(1)
    loss=(F.softplus(gap/ref['qnorm'][:,None]).masked_fill(~ref['legal'],0).sum(1)/ref['legal'].sum(1)).mean()
    return dict(drops=int((rank>10).sum()),mrr=float((1/rank.double()).mean()),softgap=float(loss))

def prepare(seed,src,truths,out):
    tick=sync()
    c=base.load(src/'cache/cohort.npz')
    legal=np.ones((len(c['ids']),TOTAL-N),bool)
    for i,(h,r,g) in enumerate(c['queries']):legal[i,[v-N for v in truths[int(h),int(r)] if v>=N]]=False
    anchor=dict(q=torch.as_tensor(c['q'],device='cuda'),legal=torch.as_tensor(legal,device='cuda'),
                selections=base.load(src/'cache/protection.npz')['selections'][:220])
    gates=base.load(f'/data/lc/untzh/forecast_protect146_20261014/seed{seed}/gates-frozen.npz')
    assert np.array_equal(gates['cohort_ids'],c['ids'])
    masks={a:torch.as_tensor(gates[a][:22],device='cuda') for a in ['forecast_anchor','current_anchor','random_matched']}
    spacepath={43:src,17:Path('/data/lc/untzh/anchor_confirm95_20261014'),29:Path('/data/lc/untzh/query_space92_20261014')}[seed]
    space=prior.controls.online.replay.previous.load_space(spacepath/'cache/query-space.pt')
    replaypath=src/'cache/full-replay.npz' if seed==43 else Path(f'/data/lc/untzh/full_replay105_20261014/seed{seed}/cache/full-replay.npz')
    z=base.load(replaypath)
    translate=dict(qt='qtail',qh='qhead',mt='mask_tail',mh='mask_head',dt='denominator_tail',dh='denominator_head')
    replay={k:torch.as_tensor(z[v][:22],device='cuda') for k,v in translate.items()}
    forecastpath=Path('/data/lc/untzh/trajectory_risk144_20261014/17') if seed==17 else Path(f'/data/lc/untzh/trajectory_confirm145_20261014/seed{seed}')
    f=torch.load(forecastpath/'forecast-rows.pt',map_location='cpu',weights_only=True)
    gradients=f['gradient_batches'].cuda()
    ids=f['batch_ids'].numpy()
    pred=base.load(forecastpath/'old_train-predictions.npz')
    probe_ix=np.argsort(c['ids'],kind='stable')[:256]
    pos=np.searchsorted(pred['ids'],c['ids'][probe_ix])
    assert np.array_equal(pred['ids'][pos],c['ids'][probe_ix])
    q=anchor['q'][probe_ix]
    ref=dict(q=q,qnorm=q.norm(dim=1).clamp_min(1e-30),gold=torch.as_tensor(pred['gold'][pos],device='cuda'),
             oldrank=torch.as_tensor(pred['oldrank'][pos],device='cuda'),legal=anchor['legal'][probe_ix])
    np.savez(out/'fixed-reference.npz',ids=c['ids'][probe_ix],q=q.cpu().numpy(),gold=pred['gold'][pos],oldrank=pred['oldrank'][pos])
    return anchor,replay,space,masks,gradients,ids,ref,dict(seconds=sync()-tick,reference_probe=256,gradient_cache_reused=True,
         reference_rule='smallest original cohort fact ids, no resampling',no_valid_input=True)

def diagnose(seed,initial,prefix,new,tape,anchor,replay,space,masks,gradients,ids,ref,out):
    E,R,opt=worker.model(initial,N,prefix)
    ce,cr,co=worker.model(initial,N,prefix)
    common=copy.deepcopy(opt.state_dict())
    before=E.detach()[N:].clone()
    records=[]
    checks=[]
    begin=sync()
    for k,j in enumerate(ids):
        status('proposal',seed=seed,batch=int(j))
        dl,counts,total=direction_losses(E,R,new,tape,int(j))
        gd={d:torch.autograd.grad(v,E,retain_graph=True)[0][N:] if v is not None else torch.zeros_like(before) for d,v in dl.items()}
        gl=torch.zeros_like(E)
        gl[N:]=gradients[k]
        if k==0:
            combined=sum(gd[d]*counts[i]/total for i,d in enumerate(DIRECTIONS))
            err=float((combined-gradients[k]).abs().max())
            assert torch.allclose(combined,gradients[k],rtol=2e-5,atol=1e-9)
            checks.append(dict(batch=int(j),direction_gradient_rebuild_max_error=err))
        chosen=anchor['selections'][k]
        legal=anchor['legal'][chosen]
        ss=anchor['q'][chosen]@E[N:].double().T
        loss=F.softplus(ss).masked_fill(~legal,0).sum(1)/legal.sum(1)
        ga=torch.autograd.grad(loss.mean(),E,retain_graph=True)[0]
        sparse={a:torch.autograd.grad((loss*masks[a][k]).mean(),E,retain_graph=True)[0] for a in masks}
        tail=F.softplus(replay['qt'][k]@E[N:].double().T).masked_fill(~replay['mt'][k],0).sum(1)/replay['dt'][k]
        head=F.softplus(replay['qh'][k]@E[N:].double().T).masked_fill(~replay['mh'][k],0).sum(1)/replay['dh'][k]
        gr=torch.autograd.grad(.5*(tail+head).mean(),E)[0]
        ordinary_rows=None
        ordinary_loss=None
        ordinary_probe=None
        for arm in ARMS:
            gp=ga if arm=='fixed_anchor' else gr if arm=='fixed_replay' else sparse[arm] if arm in sparse else torch.zeros_like(gl)
            bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm())
            alpha=.1*bn/pn if pn else 0.
            ce.data.copy_(E.detach())
            co.load_state_dict(copy.deepcopy(common))
            co.zero_grad(set_to_none=True)
            ce.grad=gl+alpha*gp
            co.step()
            unprojected=ce.detach()[N:].clone()
            if arm=='none':ordinary_rows=unprojected.clone()
            raw=(unprojected-ordinary_rows).double()
            projected=cx.source.space_module.project(raw,space)
            if arm not in ['none','fixed_replay']:
                with torch.no_grad():ce[N:].copy_(ordinary_rows+projected.to(ce.dtype))
            delta=(ce.detach()[N:]-ordinary_rows).double()
            with torch.no_grad():
                after,_,_=direction_losses(ce,cr,new,tape,int(j))
                values={d:float(v) if v is not None else None for d,v in after.items()}
                current_probe=probe(ce.detach()[N:],ref)
            if arm=='none':ordinary_loss=values;ordinary_probe=current_probe
            sgdcost={d:-alpha*float((gd[d].double()*gp[N:].double()).sum()) for d in DIRECTIONS}
            displacementcost={d:float((gd[d].double()*delta).sum()) for d in DIRECTIONS}
            loss_delta={d:values[d]-ordinary_loss[d] if values[d] is not None else None for d in DIRECTIONS}
            weights=np.array(counts)/total
            global_cost=sum(loss_delta[d]*weights[i] for i,d in enumerate(DIRECTIONS) if loss_delta[d] is not None)
            records.append(dict(batch=int(j),arm=arm,counts=counts,alpha=alpha,gp_norm=pn,
                conflict_cosines={d:cosine(gd[d],gp[N:]) for d in DIRECTIONS},
                sgd_cost=sgdcost,actual_displacement_linear_cost=displacementcost,actual_loss_delta=loss_delta,
                weighted_loss_delta=float(global_cost),raw_relative_norm=float(raw.norm()),accepted_relative_norm=float(delta.norm()),
                retained_norm_fraction=float(delta.norm()/raw.norm()) if raw.norm()>0 else None,
                probe=current_probe,probe_softgap_delta=current_probe['softgap']-ordinary_probe['softgap'],
                selected_queries=int(masks[arm][k].sum()) if arm in masks else 32 if arm in ['fixed_anchor','fixed_replay'] else 0))
        assert torch.equal(E.detach()[N:],before)
    assert torch.equal(ce[:N].cpu(),initial['E'][:N]) and torch.equal(cr.cpu(),initial['R'])
    base.write(out/'proposal-records.json',records)
    return records,dict(seconds=sync()-begin,checks=checks,host_never_advanced=True,virtual_adam_proposals=len(records),no_new_training=True)

def summarize(records):
    result={}
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        bydirection={}
        for d in DIRECTIONS:
            rr=[r for r in rows if r['actual_loss_delta'][d] is not None]
            y=np.array([r['actual_loss_delta'][d]>0 for r in rr])
            prediction={}
            for name in ['sgd_cost','actual_displacement_linear_cost']:
                score=np.array([r[name][d] for r in rr])
                prediction[name]=dict(sign_agreement=float(((score>0)==y).mean()) if len(y) else None,
                    performance=prior.risk.previous.performance(y,score),mean=float(score.mean()) if len(score) else None)
            bydirection[d]=dict(mean_actual_loss_delta=float(np.mean([r['actual_loss_delta'][d] for r in rr])) if rr else None,
                harm_cases=int(y.sum()),n=len(rr),prediction=prediction,
                mean_cosine=float(np.mean([r['conflict_cosines'][d] for r in rr if r['conflict_cosines'][d] is not None])) if any(r['conflict_cosines'][d] is not None for r in rr) else None)
        result[arm]=dict(weighted_loss_delta=float(np.mean([r['weighted_loss_delta'] for r in rows])),directions=bydirection,
            mean_alpha=float(np.mean([r['alpha'] for r in rows])),nonzero_cases=sum(r['gp_norm']>0 for r in rows),
            mean_raw_norm=float(np.mean([r['raw_relative_norm'] for r in rows])),mean_accepted_norm=float(np.mean([r['accepted_relative_norm'] for r in rows])),
            mean_probe_softgap_delta=float(np.mean([r['probe_softgap_delta'] for r in rows])))
    return result

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    begin=time.monotonic()
    _,_,_,new,_,truths=prior.risk.training_assets(17)
    reports={}
    for seed in [17,43,29]:
        status('prepare',seed=seed)
        src=Path('/data/lc/untzh')/{43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[seed]
        models=Path(f'/data/lc/untzh/train_init121_20261014/seed{seed}/models/zero')
        initial=torch.load(models/'initial.pt',map_location='cpu',weights_only=True)
        prefix=torch.load(models/'prefix10.pt',map_location='cpu',weights_only=True)
        tape=base.load(src/'cache/learning.npz')
        out=ROOT/f'seed{seed}'
        out.mkdir(parents=True,exist_ok=True)
        assert not (out/'proposal-records.json').exists()
        anchor,replay,space,masks,gradients,ids,ref,prep=prepare(seed,src,truths,out)
        assert np.array_equal(ids,np.flatnonzero(tape['epochs']==11))
        records,cost=diagnose(seed,initial,prefix,new,tape,anchor,replay,space,masks,gradients,ids,ref,out)
        results=summarize(records)
        report=dict(results=results,costs=dict(preparation=prep,diagnosis=cost))
        reports[str(seed)]=report
        base.write(out/'summary.json',report)
        print('COST147',seed,{a:r['weighted_loss_delta'] for a,r in results.items()},flush=True)
    higher=all(r['results']['forecast_anchor']['weighted_loss_delta']>r['results']['random_matched']['weighted_loss_delta'] for r in reports.values())
    base.write(ROOT/'summary.json',dict(seeds=reports,total_seconds=time.monotonic()-begin,new_training_configs=0,new_parent_trainings=0,
        virtual_adam_proposals=396,common_prefix_states=3,host_never_advanced=True,no_valid_read=True,no_test=True,
        forecast_cost_above_matched_random_all_parents=higher,
        decision='GO paired local cost interpretation only' if higher else 'NO-GO common early local cost explanation',
        no_action_rule_promoted=True,go_no_go_sign_predictor_review_pending=True))
    status('complete',seconds=time.monotonic()-begin)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
