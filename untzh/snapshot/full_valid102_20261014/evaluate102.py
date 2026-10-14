import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/increment87_20261014')
import experiment87 as existing
core,base=existing.core,existing.base
ROOT=Path(__file__).resolve().parent
N,TOTAL=existing.N,existing.TOTAL


def configure(out):
    existing.ROOT=core.ROOT=base.ROOT=out


def root_status(stage,**extra):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**extra))


def specs(seed):
    if seed==29:
        return [
            ('none','risk_increment88_20261014','none'),
            ('absolute','risk_increment88_20261014','absolute'),
            ('absolute075','risk_increment88_20261014','absolute075'),
            ('fullabsolute','budget_action90_20261014','absolute_full_uniform'),
            ('anchor_uniform','anchor_space94_20261014','anchor_null'),
            ('answer_tree','anchor_route97_20261014','anchor_tree'),
            ('risk_actionable','risk_action100_20261014','risk_actionable'),
            ('geometry_only','risk_action100_20261014','actionable_only'),
            ('query_tree','query_budget101_20261014','query_tree'),
            ('query_random','query_budget101_20261014','query_random_pool'),
            ('query_uniform','query_budget101_20261014','query_uniform'),
            ('uniform_tree_norm','route_strength98_20261014','uniform_tree_norm')]
    return [
        ('none','increment87_20261014','none'),
        ('absolute','increment87_20261014','absolute'),
        ('absolute075','increment87_20261014','absolute075'),
        ('fullabsolute','rank_confirm91_20261014','absolute_full_uniform'),
        ('normalized_abs128','rank_confirm91_20261014','absolute128_uniform'),
        ('anchor_uniform','anchor_confirm95_20261014','anchor_null')]


def combine(existing_ranks, extra_ranks, ids, rest, size):
    merged=np.empty((size,2),dtype=existing_ranks.dtype)
    merged[ids]=existing_ranks;merged[rest]=extra_ranks
    return merged


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();root_status('prepare')
    oldvalid=np.concatenate([existing.triples(0,'valid'),existing.triples(1,'valid')])
    source_new=existing.triples(2,'valid');novel=(source_new[:,0]>=N)|(source_new[:,2]>=N)
    newvalid=source_new[novel]
    queries=dict(old_valid=oldvalid,new_valid=newvalid)
    rng=np.random.default_rng(124)
    ids=dict(old_valid=rng.permutation(len(oldvalid))[:1024],new_valid=rng.permutation(len(newvalid))[:1024])
    rest={name:np.setdiff1d(np.arange(len(q)),ids[name]) for name,q in queries.items()}
    z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')
    truths=core.truthmap(z['old'],np.concatenate([z['new'],z['extra']]))
    np.savez(ROOT/'queries.npz',**queries)
    np.savez(ROOT/'selection.npz',**{f'{k}_reused':v for k,v in ids.items()},**{f'{k}_new':v for k,v in rest.items()})
    protocol=dict(old_valid=len(oldvalid),new_entity_valid=len(newvalid),source_new_valid=len(source_new),
                  excluded_new_old_old=int((~novel).sum()),reused_per_split=1024,
                  new_scored={k:len(v) for k,v in rest.items()},selection_seed=124,
                  filtering='TRAIN truths only',status='expanded exposed development evaluation; no TEST, no training')
    base.write(ROOT/'data.json',protocol);print('PREPARE',protocol,flush=True)
    prepare_seconds=time.monotonic()-tick
    summaries={}
    for seed in [29,17]:
        out=ROOT/f'seed{seed}';out.mkdir();configure(out)
        source=Path('/data/lc/untzh')/('risk_increment88_20261014' if seed==29 else 'increment87_20261014')
        initial=torch.load(source/'cache/initial.pt',map_location='cpu',weights_only=True)
        previous_queries=base.load(source/'evaluation/queries.npz')
        for name,q in queries.items():assert np.array_equal(previous_queries[name],q[ids[name]])
        teacher_sample=base.load(source/'evaluation/teacher.npz')['old_valid']
        root_status('teacher',seed=seed);teacher_tick=time.monotonic()
        teacher_extra=base.ranks(initial['E'][:N].cuda().double(),initial['R'].cuda().double(),oldvalid[rest['old_valid']],truths,arm=f'seed{seed}_teacher',split='old_valid_extra')
        teacher=combine(teacher_sample,teacher_extra,ids['old_valid'],rest['old_valid'],len(oldvalid))
        np.savez(out/'teacher.npz',old_valid=teacher)
        costs=dict(teacher_extra_seconds=time.monotonic()-teacher_tick,models={},historical_training_and_cache_not_free=True)
        metrics={}
        for name,directory,arm in specs(seed):
            model_tick=time.monotonic();path=Path('/data/lc/untzh')/directory
            m=torch.load(path/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
            assert torch.equal(m['E'][:N],initial['E'][:N].double()) and torch.equal(m['R'],initial['R'].double())
            E,R=m['E'].cuda(),m['R'].cuda();saved=base.load(path/'evaluation'/f'{arm}.npz')
            states={}
            root_status('evaluate',seed=seed,arm=name)
            for split,q in queries.items():
                extra=base.ranks(E,R,q[rest[split]],truths,arm=f'seed{seed}_{name}',split=f'{split}_extra')
                states[split]=combine(saved[split],extra,ids[split],rest[split],len(q))
            np.savez(out/f'{name}.npz',**states)
            metric={split:base.metrics(states[split],q) for split,q in queries.items()}
            metric['old_valid']['drops']=core.drops(states['old_valid'],teacher)
            metric['directional_ratios']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/metrics['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS] if name!='none' else [1.,1.,1.]
            metrics[name]=metric
            costs['models'][name]=dict(extra_scoring_seconds=time.monotonic()-model_tick,source=str(path/'models'/arm/'epoch-20.pt'),frozen_equal=True)
            base.write(out/'metrics-running.json',metrics);base.write(out/'costs-running.json',costs)
            print('EVALUATED102',dict(seed=seed,arm=name,new_mrr=metric['new_valid']['all']['filtered_train_truths']['mrr'],old_mrr=metric['old_valid']['all']['filtered_train_truths']['mrr'],old_drops=metric['old_valid']['drops']['filtered_train_truths']['drops']),flush=True)
        summaries[str(seed)]=dict(metrics=metrics,costs=costs)
        base.write(out/'summary.json',summaries[str(seed)])
    base.write(ROOT/'summary.json',dict(data=protocol,seeds=summaries,preparation_seconds=prepare_seconds,total_seconds=time.monotonic()-tick))
    root_status('complete',seconds=time.monotonic()-tick);print('COMPLETE102',time.monotonic()-tick,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:root_status('failed',error=repr(exc));raise
