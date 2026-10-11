"""Config-driven narrow/full-column scoring identity checks; no model updates."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch


def run(root,cfg):
    from shared_rank_cache_v2 import fixed_scores
    from adaptive_rank_scoring import adaptive_fixed_scores
    torch.set_num_threads(cfg['cpu_threads']);device=torch.device(cfg['device'])
    gen=torch.Generator(device=device).manual_seed(cfg['fixture_seed'])
    checks=[];started=time.monotonic()
    for dimension,queries,candidates in cfg['shapes']:
        Q=torch.randn((queries,dimension*2),generator=gen,dtype=torch.float64,device=device)[:,::2]
        E=torch.randn((candidates,dimension*2),generator=gen,dtype=torch.float64,device=device)[:,::2]
        # Include exact zero/tie inputs; noncontiguous views exercise the same
        # input-copy policy as the original implementation.
        if len(Q):Q[0]=0
        if len(Q)>1:Q[1]=-0.0
        if len(E):E[0]=E[0].abs()
        if len(E)>1:E[1]=E[0]
        expected=fixed_scores(Q,E);actual=adaptive_fixed_scores(Q,E)
        assert torch.equal(actual.view(torch.int64),expected.view(torch.int64)),(dimension,queries,candidates)
        assert torch.isfinite(actual).all()
        if len(Q):assert torch.equal(actual[0],torch.zeros_like(actual[0]))
        if len(E)>1:assert torch.equal(actual[:,0],actual[:,1])
        # Independent NumPy binary-tree reference for every output cell.
        q=Q.cpu().numpy();e=E.cpu().numpy();padded=1 << (dimension-1).bit_length()
        for off in range(0,len(q),cfg['numpy_query_batch']):
            part=q[off:off+cfg['numpy_query_batch']]
            for begin in range(0,len(e),256):
                rows=e[begin:begin+256];tree=np.zeros((len(part),len(rows),padded))
                tree[:,:,:dimension]=part[:,None,:]*rows[None,:,:]
                while tree.shape[-1]>1:tree=tree[...,0::2]+tree[...,1::2]
                assert np.array_equal(tree[...,0].copy().view(np.uint64),actual[off:off+len(part),begin:begin+len(rows)].cpu().numpy().copy().view(np.uint64))
        checks.append(dict(dimension=dimension,queries=queries,candidates=candidates,output_cells=queries*candidates,exact_original_and_numpy=True,bit_patterns_checked=True))
        print('SCORING_IDENTITY',dimension,queries,candidates,flush=True)
    result=dict(status='PASS_FIXED_TREE_SCORING_SHAPE_CONTROLS',device=str(device),torch_version=torch.__version__,test_read=False,valid_read=False,model_updates=0,
                shapes=checks,total_cells=sum(c['output_cells'] for c in checks),elapsed_s=time.monotonic()-started,
                code_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['src/adaptive_rank_scoring.py','src/shared_rank_cache_v2.py','scripts/check_rank_scoring.py']},
                limits=['Scoring identity only, no complete graph trajectory, method utility or runtime speedup claim.',
                        'CPU controls cannot certify CUDA launch-shape identity; CUDA qualification remains required before graph use.',
                        'No current registered scorer or source/config is modified.'])
    out=root/cfg['output_directory'];out.mkdir(parents=True,exist_ok=False);(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));run(root,json.loads((root/a.config).read_text()))
