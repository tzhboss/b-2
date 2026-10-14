import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/data/lc/untzh/anchor_route97_20261014')
import experiment97 as previous
prior, action, route, core, base = previous.prior, previous.action, previous.route, previous.core, previous.base
ROOT = Path(__file__).resolve().parent
previous.ROOT = previous.previous.ROOT = prior.ROOT = action.ROOT = route.ROOT = route.prev.ROOT = core.ROOT = base.ROOT = ROOT
N, SIZE = prior.N, core.SIZE
ARMS = ['risk_actionable', 'actionable_only', 'risk_shuffled_geometry']
route.ARMS = ARMS


@torch.no_grad()
def current_fraction(E, data, fractions):
    parts = []
    for lo in range(0, SIZE, 256):
        hi = min(SIZE,lo+256)
        scores = (data['q'][lo:hi,None,:]*E[data['candidates'][lo:hi]].double()).sum(-1)
        top = scores.argmax(-1)
        parts.append(fractions[lo:hi].gather(1,top[:,None]).squeeze(-1).cpu().numpy())
    return np.concatenate(parts)


def train(arm, initial, prefix, new, tape, data, model, space, anchors, fractions, permutation):
    tick=time.monotonic()
    E,R,opt=core.newmodel(initial,prefix)
    rng=np.random.default_rng(890029)
    records,decisions=[],[]
    detection_seconds=0.
    for j in range(220,440):
        ep=int(tape['epochs'][j])-1
        if j==220 or tape['epochs'][j-1]!=tape['epochs'][j]:
            torch.cuda.synchronize();start=time.monotonic()
            a=current_fraction(E,data,fractions)
            X=route.features(E,data,ep)
            if j==220:
                frame=base.load(route.SOURCE/'cache/frame-10.npz')['X'];cols=list(range(15))+[20]
                assert np.max(np.abs(X[:,cols]-frame[:,cols]))<=1e-6
            risk=model.predict_proba(X)[:,1]
            scores = a if arm=='actionable_only' else risk*(a[permutation] if arm=='risk_shuffled_geometry' else a)
            assert np.isfinite(scores).all()
            pool=np.argsort(-scores,kind='stable')[:int(np.ceil(.05*SIZE))]
            torch.cuda.synchronize();seconds=time.monotonic()-start;detection_seconds+=seconds
            decisions.append(dict(epoch=ep,seconds=seconds,pool=pool.tolist(),mean_pool_actionability=float(a[pool].mean()),mean_pool_risk=float(risk[pool].mean())))
        first=rng.choice(pool,26,replace=False);available=np.ones(SIZE,bool);available[first]=False
        chosen=np.concatenate([first,rng.choice(np.flatnonzero(available),6,replace=False)])
        loss=base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p,_=action.penalty(E,data,chosen,'absolute_full');gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm());alpha=.1*bn/pn if pn else 0.
        assert torch.isfinite(loss) and torch.isfinite(p) and torch.isfinite(gl+alpha*gp).all()
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        reference=anchors[j-220].cuda();raw=(E.detach()[N:]-reference).double();projected=prior.project(raw,space)
        assert float(projected.norm())<=float(raw.norm())*(1+1e-8)+1e-12
        with torch.no_grad():E[N:].copy_(reference+projected.to(E.dtype))
        actual=(E.detach()[N:]-reference).double();changes=(space['vectors']*actual[space['rows']]).sum(-1)
        assert torch.isfinite(E).all()
        records.append(dict(step=j,epoch=ep+1,selected=chosen.tolist(),accepted_anchor_displacement_norm=float(actual.norm()),
                            max_cross_train_score_difference=float(changes.abs().max()),gradient_ratio=alpha*pn/bn if bn else 0.))
        if j==439 or tape['epochs'][j+1]!=tape['epochs'][j]:core.progress('train',arm=arm,epoch=ep+1)
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/arm;out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    base.write(out/'steps.json',records);base.write(out/'decisions.json',decisions)
    report=dict(seconds=time.monotonic()-tick,detection_seconds_included=detection_seconds,steps=220,frozen_equal=True,
                final_anchor_displacement_norm=records[-1]['accepted_anchor_displacement_norm'],
                max_cross_train_score_difference=max(x['max_cross_train_score_difference'] for x in records))
    base.write(out/'training.json',report);return report


def main():
    torch.set_num_threads(2);torch.manual_seed(29);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial,prefix,old,new,truths,tape,data,model,coefficient,ps=route.prepare()
    action.extra_data(truths,data);space,ss=previous.previous.previous.load_space()
    anchors=torch.load('/data/lc/untzh/anchor_space94_20261014/cache/ordinary-anchors.pt',map_location='cpu',weights_only=True)
    geometry=base.load('/data/lc/untzh/actionable99_20261014/cache/actionability.npz')
    ids=base.load(route.SOURCE/'cache/cohort.npz')['ids'];assert np.array_equal(ids,geometry['ids'])
    fractions=torch.as_tensor(geometry['fraction'],device='cuda');assert fractions.shape==(SIZE,128)
    permutation=np.random.default_rng(1000029).permutation(SIZE)
    (ROOT/'cache').mkdir();np.save(ROOT/'cache/permutation.npy',permutation)
    costs=dict(preparation_seconds=time.monotonic()-tick,historical_assets_not_free=True,
               geometry_source='/data/lc/untzh/actionable99_20261014/cache/actionability.npz',training={})
    for arm in ARMS:
        costs['training'][arm]=train(arm,initial,prefix,new,tape,data,model,space,anchors,fractions,permutation)
        core.dump('costs-running.json',costs)
    metrics,es=route.evaluate(truths)
    historical=json.loads(Path('/data/lc/untzh/anchor_route97_20261014/summary.json').read_text())['metrics']
    for name,metric in historical.items():
        if name not in metrics:metrics[name]=metric
    costs.update(evaluation_seconds=es,total_seconds=time.monotonic()-tick)
    core.dump('summary.json',dict(metrics=metrics,costs=costs));core.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
