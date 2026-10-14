"""Broad protection mechanism screen with reused trajectory and basic checks."""
import collections
import copy
import json
import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.nn import functional as F

PRIOR = Path('/data/lc/untzh/risk58_20261013')
sys.path.insert(0, str(PRIOR))
import experiment58 as ex
ROOT = Path(__file__).resolve().parent
ex.ROOT = ROOT; ex.base.ROOT = ROOT
DIRECTIONS = ['head_old_gold_new', 'head_new_gold_old', 'both_new']
ARMS = {
 'absolute_control': dict(signal='absolute'),
 'gap128': dict(signal='gap'),
 'gap_full': dict(signal='gap_full'),
 'margin_full': dict(signal='margin_full'),
 'forecast_margin': dict(signal='forecast_margin'),
 'capacity_full': dict(signal='capacity_full'),
 'capacity_guard': dict(signal='capacity_guard'),
 'risk_margin': dict(signal='margin_full', route='risk'),
 'utility_absolute': dict(signal='absolute', route='utility'),
 'entity_project_absolute': dict(signal='absolute', project=True),
 'entity_project_gap': dict(signal='gap_full', project=True),
 'postadam_absolute': dict(signal='absolute', postadam=True),
 'postadam_margin': dict(signal='margin_full', postadam=True),
 'reweight_only': dict(signal='none', weight=1.5),
 'reweight_absolute': dict(signal='absolute', weight=1.5),
}


def model(initial, prefix):
    E, R, opt = ex.init_model(initial)
    E.data.copy_(prefix['E'].cuda())
    opt.load_state_dict(copy.deepcopy(prefix['optimizer']))
    return E, R, opt


def learning(E, R, new, tape, j, weight=1.):
    if weight == 1.:
        return ex.base.learning(E, R, new, tape, j)
    lo, hi = tape['bounds'][j:j+2]
    p, nh, nt = [torch.as_tensor(a, device='cuda') for a in
                (new[tape['pos'][lo:hi]], tape['nh'][lo:hi], tape['nt'][lo:hi])]
    h, r, t = p.unbind(1)
    pos = (E[h] * R[r] * E[t]).sum(-1)
    neg = (E[nh] * R[r[:, None]] * E[nt]).sum(-1)
    weights = torch.where((h < ex.N) & (t >= ex.N), weight, 1.)
    per = F.softplus(-pos) + F.softplus(neg).mean(-1)
    return (per * weights).mean() + .0005 * (E[h].square().mean() + R[r].square().mean())


def penalty(E, data, selected, signal, delta=None):
    q = data['q_t'][selected]
    gold, tau = data['gold_t'][selected], data['tau_t'][selected]
    full = signal not in ['absolute', 'gap']
    if full:
        scores = q @ E[ex.N:].double().T
        legal = data['legal_t'][selected]
    else:
        candidates = data['candidates_t'][selected]
        scores = (q[:, None] * E[candidates].double()).sum(-1)
        legal = torch.ones_like(scores, dtype=torch.bool)
    if signal == 'absolute':
        return F.softplus(scores).mean()
    z = (scores - gold[:, None]) / tau[:, None]
    if signal == 'forecast_margin':
        z = z + 37 * (q @ delta[ex.N:].double().T) / tau[:, None]
    if signal in ['gap', 'gap_full']:
        values = F.softplus(z)
    elif signal in ['margin_full', 'forecast_margin']:
        values = F.relu(z + .25).square()
    elif signal in ['capacity_full', 'capacity_guard']:
        top = z.masked_fill(~legal, -torch.inf).topk(10, dim=-1).values
        capacity = 10 - torch.as_tensor(data['oldrank'][selected], device='cuda', dtype=torch.long)
        boundary = top.gather(1, capacity[:, None]).squeeze(1)
        val = F.softplus(4 * boundary) / 4
        if signal == 'capacity_guard':
            val = val + .1 * (F.softplus(z).masked_fill(~legal, 0).sum(-1) / legal.sum(-1))
        return val.mean()
    else:
        raise ValueError(signal)
    return (values.masked_fill(~legal, 0).sum(-1) / legal.sum(-1)).mean()


def entity_project(gp, gl):
    a, b = gp[ex.N:].double(), gl[ex.N:].double()
    inner, norm2 = (a*b).sum(-1), b.square().sum(-1)
    coef = torch.where((inner < 0) & (norm2 > 0), inner / norm2.clamp_min(1e-30), 0.)
    projected = a - coef[:, None] * b
    return torch.cat([torch.zeros_like(gp[:ex.N]), projected.to(gp.dtype)]), int((coef < 0).sum())


def route(E, R, opt, gl, data, tree, policy):
    if policy == 'uniform':
        return np.arange(ex.SIZE)
    X, _ = ex.features(E, torch.zeros_like(E), data, 10)
    score = tree.predict_proba(X)[:, 1]
    if policy == 'utility':
        conflicts = []
        for lo in range(0, ex.SIZE, 256):
            ids = data['candidates_t'][lo:lo+256, :32]
            q = data['q_t'][lo:lo+256]
            direction = gl[ids].double()
            # A positive dot(q, learning gradient) means learning is already lowering scores.
            dot = (direction * q[:, None]).sum(-1)
            denom = direction.norm(dim=-1) * q.norm(dim=-1)[:, None]
            cosine = dot / denom.clamp_min(1e-30)
            conflict = F.relu(-cosine).mean(-1)
            conflicts.append(conflict.cpu().numpy())
        score = score / (1 + 10 * np.concatenate(conflicts))
    return np.argsort(-score, kind='stable')[:int(np.ceil(.2 * ex.SIZE))]


def calibrate(initial, prefix, new, tape, data):
    norms = collections.defaultdict(list)
    rng = np.random.default_rng(17)
    E, R, opt = model(initial, prefix)
    for j in range(370, 378):
        selected = rng.choice(ex.SIZE, 32, replace=False)
        gl = torch.autograd.grad(learning(E,R,new,tape,j), E)[0]
        delta = ex.proposal(E,opt,gl)
        for signal in sorted({s['signal'] for s in ARMS.values()} - {'none'}):
            p = penalty(E,data,selected,signal,delta.detach())
            gp = torch.autograd.grad(p,E)[0]
            norms[signal].append([float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())])
    result = {}
    for signal, rows in norms.items():
        a = np.asarray(rows)
        denom = np.sqrt(np.mean(a[:,1]**2))
        # Exact-zero hinges remain inactive; do not divide by a synthetic floor.
        result[signal] = {'lambda': .1*np.sqrt(np.mean(a[:,0]**2))/denom if denom else 0., 'norms':rows}
    result['absolute']['lambda'] = json.loads((PRIOR/'coefficient.json').read_text())['lambda']
    ex.dump('calibration.json',result)
    return result


def train(name, spec, initial, prefix, new, tape, data, tree, calibration):
    started = time.monotonic()
    E,R,opt = model(initial,prefix)
    rng = np.random.default_rng(17)
    records = []
    routepolicy = spec.get('route','uniform')
    pool = np.arange(ex.SIZE)
    signal = spec['signal']
    for j in range(370,740):
        loss = learning(E,R,new,tape,j,spec.get('weight',1.))
        gl = torch.autograd.grad(loss,E)[0]
        if j == 370 or tape['epochs'][j-1] != tape['epochs'][j]:
            pool = route(E,R,opt,gl,data,tree,routepolicy)
        selected = rng.choice(ex.SIZE,32,replace=False) if routepolicy=='uniform' else np.concatenate([
            rng.choice(pool,26,replace=False),rng.choice(ex.SIZE,6,replace=False)])
        projected_rows = 0
        gp = torch.zeros_like(E); p = E.new_zeros((),dtype=torch.float64)
        delta = None
        if signal!='none':
            if signal=='forecast_margin':
                delta = ex.proposal(E,opt,gl).detach()
            p = penalty(E,data,selected,signal,delta)
            gp = torch.autograd.grad(p,E)[0]
            if spec.get('project'):
                gp,projected_rows = entity_project(gp,gl)
        bn,pn = float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
        coefficient = calibration[signal]['lambda'] if signal!='none' else 0.
        alpha = min(coefficient,.1*bn/pn) if pn else 0.
        opt.zero_grad(set_to_none=True)
        if spec.get('postadam'):
            before = E.detach().clone()
            E.grad = gl;opt.step()
            dn = float((E.detach()[ex.N:]-before[ex.N:]).double().norm())
            beta = .1*dn/pn if pn else 0.
            with torch.no_grad():
                E.add_(gp,alpha=-beta)
        else:
            beta = None
            E.grad=gl+alpha*gp;opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(p) or not torch.isfinite(E).all():
            raise FloatingPointError('nonfinite search arm')
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalty':float(p),
                        'alpha':alpha,'gradient_ratio':alpha*pn/bn if bn else 0.,'postadam_beta':beta,
                        'projected_rows':projected_rows})
        if j==739 or tape['epochs'][j+1]!=tape['epochs'][j]:
            ex.progress('train',arm=name,epoch=int(tape['epochs'][j]),seconds=time.monotonic()-started)
    assert torch.equal(E[:ex.N].detach().cpu(),initial['E'][:ex.N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save({'E':E.detach().cpu().double(),'R':R.cpu().double()},out/'epoch-20.pt')
    ex.base.write(out/'steps.json',records)
    result={'seconds':time.monotonic()-started,'spec':spec,'steps':370,'frozen_rows_equal':True,
            'max_gradient_ratio':max(x['gradient_ratio'] for x in records)}
    if name=='absolute_control':
        baseline=torch.load(PRIOR/'models/uniform/epoch-20.pt',map_location='cpu',weights_only=True)
        result['historical_control_equal']=torch.equal(E.detach().cpu().double(),baseline['E'])
        assert result['historical_control_equal']
    ex.base.write(out/'training.json',result)
    return result


def dropstats(a,A):
    return {kind:{'eligible':int((A[:,col]<=10).sum()),'drops':int(((A[:,col]<=10)&(a[:,col]>10)).sum())}
            for col,kind in enumerate(['raw','filtered_train_truths'])}


@torch.no_grad()
def evaluate(old,new,arms=None):
    started=time.monotonic()
    arms=list(ARMS) if arms is None else list(arms)
    truth=collections.defaultdict(set)
    for h,r,t in np.concatenate([old,new]):truth[int(h),int(r)].add(int(t))
    valid=ex.base.load(ex.SRC/'entity_refresh43_20261012/artifacts/data/validation.npz')
    A=ex.base.load(ex.SRC/'replay_strength53_20261013/artifacts/evaluation/A.npz')
    prior=json.loads((PRIOR/'evaluation/metrics.json').read_text())
    replay=json.loads(Path('/data/lc/untzh/meor_replay57_20261013/evaluation/metrics.json').read_text())
    report={'new_only':prior['new_only'],'uniform':prior['uniform'],'historical_replay':replay['replay']}
    out=ROOT/'evaluation';out.mkdir(exist_ok=True)
    screened=[]
    for name in arms:
        model=torch.load(ROOT/'models'/name/'epoch-20.pt',map_location='cpu',weights_only=True)
        E,R=model['E'].cuda(),model['R'].cuda()
        ranks={split:ex.base.ranks(E,R,valid[key],truth,arm=name,split=split)
               for split,key in [('old_valid','old'),('new_valid','new')]}
        np.savez(out/f'{name}-valid.npz',**ranks)
        item={split:ex.base.metrics(ranks[split],valid[key]) for split,key in [('old_valid','old'),('new_valid','new')]}
        item['old_valid']['drops']=dropstats(ranks['old_valid'],A['old_valid'])
        ratios=[item['new_valid'][d]['filtered_train_truths']['mrr']/prior['new_only']['new_valid'][d]['filtered_train_truths']['mrr'] for d in DIRECTIONS]
        d=item['old_valid']['drops']['filtered_train_truths']['drops']
        oldm=item['old_valid']['all']['filtered_train_truths']['mrr']
        passed=min(ratios)>=.98 and d<=4 and oldm>=.99*prior['uniform']['old_valid']['all']['filtered_train_truths']['mrr']
        item['development_screen_pass']=passed
        item['directional_ratios']=ratios
        report[name]=item
        if passed and name!='absolute_control':screened.append(name)
        ex.progress('valid_screen',arm=name,new_mrr=item['new_valid']['all']['filtered_train_truths']['mrr'],
                    old_mrr=oldm,drops=d,ratios=ratios,passed=passed)
    valid_seconds=time.monotonic()-started
    # Keep non-dominated VALID tradeoffs; report every tested VALID configuration.
    def vector(name):
        a=report[name]
        return np.array([a['new_valid']['all']['filtered_train_truths']['mrr'],
                         a['old_valid']['all']['filtered_train_truths']['mrr'],
                         -a['old_valid']['drops']['filtered_train_truths']['drops']])
    chosen=[n for n in screened if not any(np.all(vector(m)>=vector(n)) and np.any(vector(m)>vector(n)) for m in screened if m!=n)]
    for name in chosen:
        model=torch.load(ROOT/'models'/name/'epoch-20.pt',map_location='cpu',weights_only=True)
        a=ex.base.ranks(model['E'].cuda(),model['R'].cuda(),old,truth,arm=name,split='old_train')
        np.savez(out/f'{name}-train.npz',old_train=a)
        report[name]['old_train']=ex.base.metrics(a,old)
        report[name]['old_train']['drops']=dropstats(a,A['old_train'])
        ex.progress('full_train_evaluation',arm=name,drops=report[name]['old_train']['drops']['filtered_train_truths']['drops'])
    ex.dump('evaluation/metrics.json',report)
    lines=['# Protection mechanism search','',
           '| Arm | New MRR | Old VALID MRR | VALID drops | Old-new ratio | New-old ratio | New-new ratio | TRAIN drops |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,item in report.items():
        ratios=[item['new_valid'][d]['filtered_train_truths']['mrr']/report['new_only']['new_valid'][d]['filtered_train_truths']['mrr'] for d in DIRECTIONS]
        td=item.get('old_train',{}).get('drops',{}).get('filtered_train_truths',{}).get('drops','uncomputed')
        lines.append(f"| {name} | {item['new_valid']['all']['filtered_train_truths']['mrr']:.9f} | {item['old_valid']['all']['filtered_train_truths']['mrr']:.9f} | {item['old_valid']['drops']['filtered_train_truths']['drops']} | {ratios[0]:.6f} | {ratios[1]:.6f} | {ratios[2]:.6f} | {td} |")
    (ROOT/'results.md').write_text('\n'.join(lines)+'\n')
    return report,{'seconds':time.monotonic()-started,'valid_seconds':valid_seconds,'full_train_arms':chosen}


def prepare_data(initial,old,new):
    data=ex.base.load(PRIOR/'cache/cohort.npz')
    truth=collections.defaultdict(set)
    for h,r,t in np.concatenate([old,new]):truth[int(h),int(r)].add(int(t))
    legal=np.ones((ex.SIZE,ex.TOTAL-ex.N),bool)
    for i,(h,r,t) in enumerate(data['queries']):
        ids=[c-ex.N for c in truth[int(h),int(r)] if c>=ex.N]
        legal[i,ids]=False
    data['legal_t']=torch.as_tensor(legal,device='cuda')
    for key in ['q','gold','tau','candidates']:
        data[key+'_t']=torch.as_tensor(data[key],device='cuda')
    return data


def main():
    torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(PRIOR/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    tick=time.monotonic();data=prepare_data(initial,old,new)
    calibration=calibrate(initial,prefix,new,tape,data)
    costs={'prepare_seconds':time.monotonic()-tick,'training':{}}
    tree=joblib.load(PRIOR/'cache/tree_current.joblib')
    ex.dump('environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'seed':17})
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,tree,calibration)
        ex.dump('costs-running.json',costs)
    _,costs['evaluation']=evaluate(old,new)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs)
    ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
