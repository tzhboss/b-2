import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/action_cost147_20261014')
import experiment147 as diagnostic
previous=diagnostic.prior
base,worker,core,cx=diagnostic.base,diagnostic.worker,diagnostic.core,diagnostic.cx
ROOT=Path(__file__).resolve().parent
N=diagnostic.N

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))
def sync():torch.cuda.synchronize();return time.monotonic()

def forecast(seed,initial,prefix,new,tape,anchor,replay,space,gradients,ids,out,oldpath):
    tick=sync()
    E,R,opt=worker.model(initial,N,prefix)
    gs=[]
    for k,j in enumerate(ids):
        chosen=anchor['selections'][k]
        ss=anchor['q'][chosen]@E[N:].double().T
        legal=anchor['legal'][chosen]
        ga=torch.autograd.grad((F.softplus(ss).masked_fill(~legal,0).sum(1)/legal.sum(1)).mean(),E)[0][N:]
        tail=F.softplus(replay['qt'][k]@E[N:].double().T).masked_fill(~replay['mt'][k],0).sum(1)/replay['dt'][k]
        head=F.softplus(replay['qh'][k]@E[N:].double().T).masked_fill(~replay['mh'][k],0).sum(1)/replay['dh'][k]
        gr=torch.autograd.grad(.5*(tail+head).mean(),E)[0][N:]
        gl=gradients[k]
        bn=gl.double().norm()
        gs.append(torch.stack([gl,gl+ga*(.1*bn/ga.double().norm()),gl+gr*(.1*bn/gr.double().norm())]))
    g=torch.stack(gs)
    mean=g.mean(0)
    second=g.square().mean(0)
    gradient_time=sync()-tick
    tick=sync()
    group=prefix['optimizer']['param_groups'][0]
    state=prefix['optimizer']['state'][group['params'][0]]
    b1,b2=group['betas']
    step0=int(state['step'].item())
    horizon=int((tape['epochs']>=11).sum())
    rows=prefix['E'][N:].cuda()[None].expand(3,-1,-1).clone()
    m=state['exp_avg'][N:].cuda()[None].expand_as(rows).clone()
    v=state['exp_avg_sq'][N:].cuda()[None].expand_as(rows).clone()
    for k in range(1,horizon+1):
        m.lerp_(mean,1-b1)
        v.mul_(b2).add_(second,alpha=1-b2)
        den=v.sqrt().div_((1-b2**(step0+k))**.5).add_(group['eps'])
        rows.addcdiv_(m,den,value=-group['lr']/(1-b1**(step0+k)))
        delta=cx.source.space_module.project((rows[1]-rows[0]).double(),space)
        rows[1]=rows[0]+delta.to(rows.dtype)
    prior_forecast=torch.load(oldpath/'forecast-rows.pt',map_location='cpu',weights_only=True)
    assert torch.equal(rows[0].cpu(),prior_forecast['stationary_mean']), 'ordinary formula unchanged'
    assert torch.isfinite(rows).all()
    moment_time=sync()-tick
    torch.save(dict(rows=rows.cpu(),mean=mean.cpu(),second=second.cpu(),horizon=horizon,
        batch_ids=torch.as_tensor(ids)),out/'action-forecast.pt')
    return rows,dict(gradient_seconds=gradient_time,moment_seconds=moment_time,total_seconds=gradient_time+sync()-tick,
        protection_gradient_batches=44,horizon=horizon,ordinary_cache_equal=True,no_training_rollout=True)

@torch.no_grad()
def score(initial,rows,facts,pred,current,truths,out):
    tick=sync()
    ids=current['ids']
    pos=np.searchsorted(pred['ids'],ids)
    assert np.array_equal(pred['ids'][pos],ids)
    E,R=initial['E'].cuda().double(),initial['R'].cuda().double()
    matrices=torch.cat([current_rows[None],rows],0).double()
    parts=[]
    for lo in range(0,len(ids),128):
        b=facts[ids[lo:lo+128]]
        q=E[b[:,0]]*R[b[:,1]]
        ss=torch.einsum('qd,snd->sqn',q,matrices)
        gold=torch.as_tensor(pred['gold'][pos[lo:lo+len(b)]],device='cuda')
        legal=torch.ones_like(ss[0],dtype=torch.bool)
        for i,(h,r,g) in enumerate(b):legal[i,[t-N for t in truths[int(h),int(r)] if t>=N]]=False
        rank=torch.as_tensor(pred['oldrank'][pos[lo:lo+len(b)]],device='cuda')[None]+((ss>gold[None,:,None])&legal[None]).sum(2)
        parts.append(rank.cpu().numpy())
    ranks=np.concatenate(parts,axis=1)
    assert np.array_equal(ranks[0],current['current_rank'])
    assert np.array_equal(ranks[1],current['stationary_mean_rank'])
    payload=dict(ids=ids,current_rank=ranks[0],ordinary_forecast=ranks[1],anchor_forecast=ranks[2],replay_forecast=ranks[3],
        anchor_rrgain=1/ranks[2]-1/ranks[1],replay_rrgain=1/ranks[3]-1/ranks[1],
        anchor_rank_gain=ranks[1]-ranks[2],replay_rank_gain=ranks[1]-ranks[3],
        anchor_predicted_rescue=((ranks[1]>10)&(ranks[2]<=10)).astype(float),
        replay_predicted_rescue=((ranks[1]>10)&(ranks[3]<=10)).astype(float),
        oldrank=current['oldrank'],current_gap=current['current_full_gap'])
    np.savez(out,**payload)
    return payload,dict(seconds=sync()-tick,n=len(ids),full_candidates=True,current_rank_equal=True,ordinary_forecast_rank_equal=True)

def evaluate(data,ordinary,protected,action):
    ids=data['ids']
    before=ordinary[ids,1]
    after=protected[ids,1]
    y=(before>10)&(after<=10)
    harm=(before<=10)&(after>10)
    actual=1/after-1/before
    predicted=data[action+'_rrgain']
    methods=dict(rrgain=predicted,rank_gain=data[action+'_rank_gain'],predicted_rescue=data[action+'_predicted_rescue'],
                 current_rank=data['current_rank'],current_gap=data['current_gap'],ordinary_forecast=data['ordinary_forecast'],oldrank=data['oldrank'])
    report={name:previous.risk.previous.performance(y,value) for name,value in methods.items()}
    return dict(n=len(ids),extra_rescues=int(y.sum()),new_top10_harm=int(harm.sum()),prediction=report,
        actual_mean_rrgain=float(actual.mean()),predicted_mean_rrgain=float(predicted.mean()),rrgain_mae=float(np.abs(actual-predicted).mean()),
        actual_rr_improved=int((actual>0).sum()),actual_rr_worsened=int((actual<0).sum()),
        positive_gain_sign_precision=float((actual[predicted>0]>0).mean()) if (predicted>0).any() else None,
        go=bool(report['rrgain']['ap'] is not None and report['rrgain']['ap']>report['current_rank']['ap'] and report['rrgain']['ap']>report['ordinary_forecast']['ap']))

def main():
    global current_rows
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    begin=time.monotonic()
    _,_,_,new,_,truths=previous.risk.training_assets(17)
    old=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')['old']
    bundles=[]
    for seed in [17,43,29]:
        status('prepare',seed=seed)
        src=Path('/data/lc/untzh')/{43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[seed]
        path=Path(f'/data/lc/untzh/train_init121_20261014/seed{seed}/models/zero')
        initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=True)
        prefix=torch.load(path/'prefix10.pt',map_location='cpu',weights_only=True)
        tape=base.load(src/'cache/learning.npz')
        oldpath=Path('/data/lc/untzh/trajectory_risk144_20261014/17') if seed==17 else Path(f'/data/lc/untzh/trajectory_confirm145_20261014/seed{seed}')
        out=ROOT/f'seed{seed}'
        out.mkdir(parents=True,exist_ok=True)
        assert not (out/'action-forecast.pt').exists()
        anchor,replay,space,masks,gradients,ids,ref,prep=diagnostic.prepare(seed,src,truths,out)
        status('forecast',seed=seed)
        rows,fcost=forecast(seed,initial,prefix,new,tape,anchor,replay,space,gradients,ids,out,oldpath)
        current_rows=prefix['E'][N:].cuda()
        queries=dict(old_train=old,old_valid=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['old_valid'])
        predictions={}
        costs={}
        for split,facts in queries.items():
            status('score',seed=seed,split=split)
            pred=base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{seed}-{split}-half-prediction.npz')
            current=base.load(oldpath/f'{split}-predictions.npz')
            predictions[split],costs[split]=score(initial,rows,facts,pred,current,truths,out/f'{split}-predictions.npz')
        preparation=dict(assets=prep,forecast=fcost,scoring=costs)
        base.write(out/'preparation.json',preparation)
        bundles.append((seed,out,predictions,preparation))
    base.write(ROOT/'predictions-frozen.json',dict(all_new_predictions_before_current_label_read=True,retrospective_development=True,no_action_submitted=True))
    reports={}
    for seed,out,predictions,cost in bundles:
        status('label',seed=seed)
        source=Path(f'/data/lc/untzh/forecast_protect146_20261014/seed{seed}/evaluation')
        ordinary=base.load(source/'none.npz')
        protected={a:base.load(source/f'fixed_{a}.npz') for a in ['anchor','replay']}
        cases={s:{a:evaluate(data,ordinary[s],protected[a][s],a) for a in protected} for s,data in predictions.items()}
        report=dict(cases=cases,costs=cost)
        reports[str(seed)]=report
        base.write(out/'summary.json',report)
        print('ACTION149',seed,{a:(r['extra_rescues'],r['prediction']['rrgain']['ap'],r['go']) for a,r in cases['old_valid'].items()},flush=True)
    go=all(v['cases']['old_valid'][a]['go'] for v in reports.values() for a in ['anchor','replay'])
    base.write(ROOT/'summary.json',dict(seeds=reports,total_seconds=time.monotonic()-begin,new_training_configs=0,new_parent_trainings=0,
        new_protection_gradient_batches=132,old_learning_gradients_reused=66,no_test=True,no_action_submitted=True,
        decision='GO benefit diagnostic only; adaptive update unproven' if go else 'NO-GO stationary action forecast; no parameter repair',
        all_valid_development=True))
    status('complete',seconds=time.monotonic()-begin)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
