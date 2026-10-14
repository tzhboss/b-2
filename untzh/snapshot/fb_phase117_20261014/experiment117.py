import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/pair_constraint111_20261014')
import experiment111 as pair
base,core=pair.base,pair.core
ROOT=Path(__file__).resolve().parent

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def describe(states):
    report={}
    for col,name in enumerate(['raw','filtered_train_truths']):
        eligible=states['old_candidates'][:,col]<=10;drops={k:eligible&(v[:,col]>10) for k,v in states.items()}
        phases={k:dict(drops=int(v.sum()),eligible=int(eligible.sum()),mrr=float((1/states[k][:,col]).mean())) for k,v in drops.items()}
        transitions={}
        for before,after in [('old_candidates','expanded_initial'),('expanded_initial','epoch10'),('epoch10','epoch20')]:
            transitions[f'{before}_to_{after}']=dict(new_drops=int((eligible&~drops[before]&drops[after]).sum()),rescued=int((drops[before]&~drops[after]).sum()),persistent=int((drops[before]&drops[after]).sum()))
        report[name]=dict(phases=phases,transitions=transitions)
    return report

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    base.ROOT=ROOT;tick=time.monotonic();status('starting');z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz');truths=core.truthmap(z['old'],np.concatenate([z['new'],z['extra']]))
    reports={}
    for seed in [43,17,29]:
        standard=Path('/data/lc/untzh')/({43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[seed])
        initial=torch.load(standard/'cache/initial.pt',map_location='cpu',weights_only=True)
        fast=pair.fast43(initial,truths) if seed==43 else pair.fastmodule.FastOld(seed,truths);cases={}
        for mode,path in [('standard',standard),('low',Path(f'/data/lc/untzh/lowinit_confirm116_20261014/seed{seed}'))]:
            init=torch.load(path/'cache/initial.pt',map_location='cpu',weights_only=True);prefix=torch.load(path/'cache/prefix10.pt',map_location='cpu',weights_only=True)
            final=base.load(path/'evaluation/none.npz') if seed==43 or mode=='low' else dict(old_train=base.load(path/'evaluation/none.npz')['old_train'],old_valid=base.load(f'/data/lc/untzh/full_valid102_20261014/seed{seed}/none.npz')['old_valid'])
            split_reports={};out=ROOT/f'seed{seed}'/mode;out.mkdir(parents=True)
            for split in ['old_train','old_valid']:
                status('score',seed=seed,mode=mode,split=split)
                expanded,cost0=fast.ranks(init['E'].cuda().double(),init['R'].cuda().double(),split)
                halfway,cost10=fast.ranks(prefix['E'].cuda().double(),init['R'].cuda().double(),split)
                states=dict(old_candidates=fast.teacher[split],expanded_initial=expanded,epoch10=halfway,epoch20=final[split])
                np.savez(out/f'{split}-phases.npz',**states);split_reports[split]=dict(metrics=describe(states),costs=dict(expanded=cost0,epoch10=cost10))
                d=split_reports[split]['metrics']['filtered_train_truths'];print('PHASE117',seed,mode,split,[v['drops'] for v in d['phases'].values()],d['transitions'],flush=True)
            cases[mode]=split_reports
        reports[str(seed)]=cases;base.write(ROOT/'summary-running.json',dict(seeds=reports))
    base.write(ROOT/'summary.json',dict(seeds=reports,no_training=True,total_seconds=time.monotonic()-tick));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
