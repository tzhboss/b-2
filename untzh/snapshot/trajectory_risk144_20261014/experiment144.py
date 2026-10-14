import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
import copy
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/risk_link136_20261014')
import experiment136 as previous
worker,base,core,cx=previous.worker,previous.base,previous.core,previous.cx
ROOT=Path(__file__).resolve().parent
ARMS=['one_batch_linear','stationary_mean']

def sync():torch.cuda.synchronize();return time.monotonic()
def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def training_assets(parent):
    fb=parent==17
    src=Path('/data/lc/untzh')/('increment87_20261014' if fb else 'complex_transfer118_20261014')
    train_src=src if fb else Path('/data/lc/untzh/wn_transfer112_20261014')
    n,total=(11258,13134) if fb else (24567,28660)
    z=base.load(train_src/'cache/train.npz')
    truths=core.truthmap(z['old'],np.concatenate([z['new'],z['extra']]) if fb else z['new'])
    return n,total,src,z['new'],base.load(train_src/'cache/learning.npz'),truths

def forecasts(parent,n,initial,prefix,new,tape,out):
    tick=sync()
    E,R,opt=worker.model(initial,n,prefix)
    learning=base.learning if parent==17 else cx.learning
    ids=np.flatnonzero(tape['epochs']==11)
    horizon=int((tape['epochs']>=11).sum())
    gradients=[]
    for j in ids:
        g=torch.autograd.grad(learning(E,R,new,tape,int(j)),E)[0]
        assert torch.count_nonzero(g[:n])==0
        gradients.append(g[n:].detach().clone())
    gradients=torch.stack(gradients)
    gradient_seconds=sync()-tick
    tick=sync()
    # Exactly one next proposal provides the equal-horizon version of the old
    # single-proposal baseline, without changing the original prefix.
    before=E.detach()[n:].clone()
    opt.zero_grad(set_to_none=True)
    full=torch.zeros_like(E)
    full[n:]=gradients[0]
    E.grad=full
    opt.step()
    single=before+horizon*(E.detach()[n:]-before)
    assert torch.equal(E[:n].cpu(),initial['E'][:n])
    single_seconds=sync()-tick
    tick=sync()
    mean=gradients.mean(0)
    second=gradients.square().mean(0)
    scale=horizon**-.5
    gs=torch.cat([mean[None],mean[None]+scale*(gradients-mean[None])],0)
    g2s=torch.cat([second[None],second[None]+scale*(gradients.square()-second[None])],0)
    p=prefix['optimizer']
    group=p['param_groups'][0]
    assert not group['amsgrad'] and group['weight_decay']==0 and not group['maximize']
    state=p['state'][group['params'][0]]
    step0=int(state['step'].item())
    b1,b2=group['betas']
    rows=before[None].expand(len(gs),-1,-1).clone()
    m=state['exp_avg'][n:].cuda()[None].expand_as(rows).clone()
    v=state['exp_avg_sq'][n:].cuda()[None].expand_as(rows).clone()
    for k in range(1,horizon+1):
        m.lerp_(gs,1-b1)
        v.mul_(b2).add_(g2s,alpha=1-b2)
        den=v.sqrt().div_((1-b2**(step0+k))**.5).add_(group['eps'])
        rows.addcdiv_(m,den,value=-group['lr']/(1-b1**(step0+k)))
    assert torch.isfinite(rows).all() and torch.isfinite(single).all()
    moment_seconds=sync()-tick
    payload=dict(current=before.cpu(),one_batch_linear=single.cpu(),stationary_mean=rows[0].cpu(),scenarios=rows[1:].cpu(),
                 gradient_mean=mean.cpu(),gradient_second=second.cpu(),gradient_batches=gradients.cpu(),batch_ids=torch.as_tensor(ids),horizon=horizon)
    torch.save(payload,out/'forecast-rows.pt')
    return payload,dict(gradient_batches=len(ids),hypothetical_horizon=horizon,
        moment_iterations=horizon,scenario_count=len(ids),gradient_seconds=gradient_seconds,single_proposal_seconds=single_seconds,
        moment_forecast_seconds=moment_seconds,save_seconds=sync()-tick-moment_seconds,
        total_seconds=sync()-tick+gradient_seconds+single_seconds,
        no_neural_training_rollout=True,stationary_gradient_approximation=True,candidate_dependence_preserved=True)

@torch.no_grad()
def predict(parent,n,initial,queries,pred,feature,truths,forecast,out):
    tick=sync()
    ids=feature['ids']
    pos=np.searchsorted(pred['ids'],ids)
    assert np.array_equal(pred['ids'][pos],ids)
    facts=queries[ids]
    gold=pred['gold'][pos]
    oldrank=pred['oldrank'][pos]
    qnorm=pred['querynorm'][pos]
    matrices=torch.cat([forecast['current'][None],forecast['one_batch_linear'][None],forecast['stationary_mean'][None],forecast['scenarios']],0).cuda().double()
    E,R=initial['E'].cuda().double(),initial['R'].cuda().double()
    rank_parts=[]
    gap_parts=[]
    for lo in range(0,len(facts),128):
        b=facts[lo:lo+128]
        q=previous.query(E,R,b,parent=='WN_ComplEx')
        scores=torch.einsum('qd,snd->sqn',q,matrices)
        gs=torch.as_tensor(gold[lo:lo+len(b)],device='cuda')
        legal=torch.ones(scores.shape[1:],dtype=torch.bool,device='cuda')
        for i,(h,r,g) in enumerate(b):
            forbidden=[v-n for v in truths[int(h),int(r)] if v>=n]
            if forbidden:legal[i,forbidden]=False
        above=(scores>gs[None,:,None])&legal[None]
        rank_parts.append((torch.as_tensor(oldrank[lo:lo+len(b)],device='cuda')[None]+above.sum(2)).cpu().numpy())
        gap=(scores[0]-gs[:,None]).masked_fill(~legal,-torch.inf).max(1).values
        gap_parts.append(gap.cpu().numpy()/np.maximum(qnorm[lo:lo+len(b)],1e-30))
    ranks=np.concatenate(rank_parts,axis=1)
    assert np.array_equal(ranks[0],feature['prefix_rank'])
    scenario=ranks[3:]
    X=feature['X']
    payload=dict(ids=ids,current_rank=ranks[0],forecast_ranks=ranks,
        oldrank=oldrank,gold=gold,querynorm=qnorm,oldrank_boundary=oldrank,
        current_full_rank=ranks[0],current_full_gap=np.concatenate(gap_parts),sampled_gap128=X[:,3],sampled_excess128=X[:,6],
        one_batch_linear_rank=ranks[1],stationary_mean_rank=ranks[2],
        scenario_drop_fraction=(scenario>10).mean(0),scenario_mean_rank=scenario.mean(0),
        scenario_mean_rr=(1/scenario).mean(0),scenario_rank_std=scenario.std(0),
        current_near_boundary=-np.abs(ranks[0]-10.5),forecast_near_boundary=-np.abs(ranks[2]-10.5),
        scenario_boundary_mass=(scenario==11).mean(0))
    np.savez(out,**payload)
    return payload,dict(seconds=sync()-tick,queries=len(ids),scenario_count=len(scenario),all_new_candidates=True,
        current_rank_cache_equal=True,bytes=sum(v.nbytes for v in payload.values()))

def target_report(y,methods,probabilities):
    n=len(y)
    total=int(y.sum())
    return dict(n=n,events=total,event_fraction=total/n if n else None,
        predictions={name:previous.performance(y,value) for name,value in methods.items()},
        brier={name:float(np.mean((value-y)**2)) if n else None for name,value in probabilities.items()},
        zero_risk_brier=total/n if n else None,
        empirical_label_prevalence_brier=float(y.var()) if n else None,
        prevalence_reference_is_evaluation_only=True)

def evaluate(parent,data,ordinary,anchor):
    ids=data['ids']
    before=data['current_rank']
    after=ordinary[ids,1]
    protected=anchor[ids,1]
    safe=before<=10
    methods={name:data[name] for name in ['oldrank_boundary','current_full_rank','current_full_gap','sampled_gap128','sampled_excess128',
        'one_batch_linear_rank','stationary_mean_rank','scenario_drop_fraction','scenario_mean_rank']}
    probabilities=dict(one_batch_linear=(data['one_batch_linear_rank']>10).astype(float),
                       stationary_mean=(data['stationary_mean_rank']>10).astype(float),scenario=data['scenario_drop_fraction'])
    late=target_report(after[safe]>10,{name:v[safe] for name,v in methods.items()},{name:v[safe] for name,v in probabilities.items()})
    rescue=target_report(after[~safe]<=10,{name:-v[~safe] for name,v in methods.items()},
                         {name:1-v[~safe] for name,v in probabilities.items()})
    late['predicted_count']=float(data['scenario_drop_fraction'][safe].sum())
    rescue['predicted_count']=float((1-data['scenario_drop_fraction'][~safe]).sum())
    extra=(after>10)&(protected<=10)
    gains=target_report(extra,{**methods,**{name:data[name] for name in ['current_near_boundary','forecast_near_boundary','scenario_boundary_mass']}},{})
    harm=(after<=10)&(protected>10)
    errors={name:dict(mean_abs_rank_error=float(np.abs(data[name]-after).mean()),
                     median_abs_rank_error=float(np.median(np.abs(data[name]-after))),
                     mean_rr_abs_error=float(np.abs(1/data[name]-1/after).mean())) for name in ['one_batch_linear_rank','stationary_mean_rank','scenario_mean_rank']}
    return dict(n=len(ids),prefix_drops=int((~safe).sum()),ordinary_drops=int((after>10).sum()),
        late_new=late,self_recovery=rescue,anchor_extra_rescue=gains,anchor_new_harm=int(harm.sum()),errors=errors,
        scenario_std_quantiles={str(p):float(np.quantile(data['scenario_rank_std'],p)) for p in [0,.5,.9,.99,1]},
        note='forecast assumptions uncalibrated; reference scoring not unknown gold discovery; no action trained')

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    begin=time.monotonic()
    bundles=[]
    prep_costs={}
    for parent in [17,'WN_ComplEx']:
        status('prepare',parent=str(parent))
        tick=time.monotonic()
        n,total,src,new,tape,truths=training_assets(parent)
        ip,pp,op,rp,ap=previous.previous.paths(parent)
        initial=torch.load(ip,map_location='cpu',weights_only=True)
        prefix=torch.load(pp,map_location='cpu',weights_only=True)
        out=ROOT/str(parent)
        out.mkdir(parents=True,exist_ok=True)
        assert not (out/'forecast-rows.pt').exists(), 'existing frozen forecasts: do not silently regenerate'
        forecast,cost=forecasts(parent,n,initial,prefix,new,tape,out)
        queries=base.load(src/'evaluation/queries.npz')
        if parent==17:queries['old_valid']=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['old_valid']
        predictions={}
        scoring={}
        for s in ['old_train','old_valid']:
            status('forecast_score',parent=str(parent),split=s)
            pred=base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{parent}-{s}-half-prediction.npz')
            feature=base.load(f'/data/lc/untzh/risk_link136_20261014/{parent}/cache/{s}-features.npz')
            predictions[s],scoring[s]=predict(parent,n,initial,queries[s],pred,feature,truths,forecast,out/f'{s}-predictions.npz')
        prep_costs[str(parent)]=dict(gradient_forecast=cost,scoring=scoring,total_seconds=time.monotonic()-tick)
        bundles.append((parent,rp,ap,op,prefix,n,predictions,out))
    base.write(ROOT/'predictions-frozen.json',dict(all_predictions_saved_before_endpoint_read=True,
        targets=['ordinary_late_new_drop','ordinary_self_recovery','anchor_extra_rescue'],no_current_training=True,no_test=True))
    reports={}
    for parent,rp,ap,op,prefix,n,predictions,out in bundles:
        status('label',parent=str(parent))
        ordinary=base.load(rp)
        anchor=base.load(ap)
        endpoint=torch.load(op,map_location='cpu',weights_only=True)
        forecast=torch.load(out/'forecast-rows.pt',map_location='cpu',weights_only=True)
        actual_rows=endpoint['E'][n:]
        norm_report={name:dict(mean_row_norm=float(forecast[name].double().norm(dim=1).mean()),
                              mean_row_distance_to_actual=float((forecast[name].double()-actual_rows).norm(dim=1).mean())) for name in ['current','one_batch_linear','stationary_mean']}
        norm_report['actual_mean_row_norm']=float(actual_rows.norm(dim=1).mean())
        report=dict(cases={s:evaluate(parent,data,ordinary[s],anchor[s]) for s,data in predictions.items()},
                    costs=prep_costs[str(parent)],forecast_geometry=norm_report)
        reports[str(parent)]=report
        base.write(out/'summary.json',report)
        for s,r in report['cases'].items():
            print('RISK144',parent,s,'late',r['late_new']['events'],'recovery',r['self_recovery']['events'],
                'lateAP',{k:v['ap'] for k,v in r['late_new']['predictions'].items()},
                'recoveryAP',{k:v['ap'] for k,v in r['self_recovery']['predictions'].items()},flush=True)
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-begin,
        new_training_configs=0,new_parent_trainings=0,gradient_batches=33,stationary_moment_prediction_not_training=True,
        no_test=True,all_valid_development=True,go_no_go_pending=True))
    status('complete',seconds=time.monotonic()-begin)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
