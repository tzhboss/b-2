import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/anchor_space94_20261014')
import experiment94 as absolute
sys.path.insert(0,'/data/lc/untzh/full_replay105_20261014')
import experiment105 as replay
sys.path.insert(0,'/data/lc/untzh/rank_confirm91_20261014')
import experiment91 as source17
prior,action,route,core,base=absolute.prior,absolute.action,absolute.route,absolute.core,absolute.base
ROOT=Path(__file__).resolve().parent
N=absolute.N


def status(stage,**extra):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**extra))


def configure(out,source):
    absolute.ROOT=absolute.previous.ROOT=prior.ROOT=source17.ROOT=action.ROOT=route.ROOT=route.prev.ROOT=core.ROOT=base.ROOT=out
    replay.ROOT=out;replay.previous.ROOT=ROOT
    route.SOURCE=source


class OnlineAnchor:
    def __init__(self,initial,prefix,new,tape):
        self.E,self.R,self.opt=core.newmodel(initial,prefix)
        self.new,self.tape=new,tape;self.step=220

    def __getitem__(self,k):
        assert k==self.step-220
        loss=base.learning(self.E,self.R,self.new,self.tape,self.step)
        self.opt.zero_grad(set_to_none=True);loss.backward();self.opt.step()
        self.step+=1
        return self.E.detach()[N:]


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();status('starting');cases={}
    for seed in [29,17]:
        source=Path('/data/lc/untzh')/('risk_increment88_20261014' if seed==29 else 'increment87_20261014')
        reference=Path('/data/lc/untzh')/('anchor_space94_20261014' if seed==29 else 'anchor_confirm95_20261014')
        for kind in ['absolute','full_replay']:
            out=ROOT/f'seed{seed}'/kind;out.mkdir(parents=True);configure(out,source)
            torch.manual_seed(seed);status('prepare',seed=seed,kind=kind);prep=time.monotonic()
            if seed==29:
                initial,prefix,old,new,truths,tape,data,model,coefficient,ps=route.prepare();action.extra_data(truths,data)
            else:initial,prefix,new,tape,truths,data,ps=source17.prepare()
            spaces=Path('/data/lc/untzh')/('query_space92_20261014' if seed==29 else 'anchor_confirm95_20261014')
            space=replay.previous.load_space(spaces/'cache/query-space.pt')
            if kind=='full_replay':
                z=base.load(f'/data/lc/untzh/full_replay105_20261014/seed{seed}/cache/full-replay.npz')
                data={k:torch.as_tensor(v,device='cuda') for k,v in z.items() if k!='positions'}
            setup=time.monotonic();online=OnlineAnchor(initial,prefix,new,tape);setup_seconds=time.monotonic()-setup
            prepare_seconds=time.monotonic()-prep;status('train',seed=seed,kind=kind)
            if kind=='absolute':
                training=absolute.train('anchor_null',initial,prefix,new,tape,data,space,online)
                result_path=out/'models/anchor_null/epoch-20.pt';reference_path=reference/'models/anchor_null/epoch-20.pt'
            else:
                training=replay.train('full_replay_anchor',initial,prefix,new,tape,data,space,online,out)
                result_path=out/'models/full_replay_anchor/epoch-20.pt';reference_path=Path(f'/data/lc/untzh/full_replay105_20261014/seed{seed}/models/full_replay_anchor/epoch-20.pt')
            assert online.step==440
            actual=torch.load(result_path,map_location='cpu',weights_only=True)
            original=torch.load(reference_path,map_location='cpu',weights_only=True)
            ordinary=torch.load(source/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
            equal=torch.equal(actual['E'],original['E']) and torch.equal(actual['R'],original['R'])
            shadow_equal=torch.equal(online.E.detach().cpu().double(),ordinary['E']) and torch.equal(online.R.cpu().double(),ordinary['R'])
            report=dict(seed=seed,kind=kind,preparation_seconds=prepare_seconds,online_setup_seconds_included=setup_seconds,
                        training=training,endpoint_equal=equal,shadow_ordinary_endpoint_equal=shadow_equal,
                        max_parameter_difference=float((actual['E']-original['E']).abs().max()),
                        required_per_step_anchor_cache_bytes=0,eliminated_anchor_tensor_bytes=330176000,
                        shadow_compute_included_in_training=True,historical_assets_not_free=True,
                        reference_model=str(reference_path))
            if equal and shadow_equal:
                source_summary=Path('/data/lc/untzh/full_valid102_20261014/summary.json') if kind=='absolute' else Path('/data/lc/untzh/full_replay105_20261014/summary.json')
                saved=json.loads(source_summary.read_text())['seeds'][str(seed)]['metrics']
                report['reused_metrics']=saved['anchor_uniform' if kind=='absolute' else 'full_replay_anchor']
                report['reuse_note']='parameter-identical implementation, not independent repeat; no re-scoring'
            else:
                report['reuse_note']='no metric reuse; endpoint mismatch requires new evaluation'
            cases[f'{seed}_{kind}']=report;base.write(out/'implementation.json',report)
            print('ONLINE106',dict(seed=seed,kind=kind,endpoint_equal=equal,shadow_equal=shadow_equal,seconds=training['seconds']),flush=True)
            del online
    base.write(ROOT/'summary.json',dict(cases=cases,total_seconds=time.monotonic()-tick))
    status('complete' if all(x['endpoint_equal'] and x['shadow_ordinary_endpoint_equal'] for x in cases.values()) else 'needs_evaluation',seconds=time.monotonic()-tick)


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
