"""Registered real-graph shared-update preflight and independent patch audit."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch


def audit(root,cfg,job):
    from common_training import sha
    from shared_rank_graph import load_scope
    torch.set_num_threads(2);started=time.monotonic();out=root/'artifacts/shared-preflight'/job['id']
    receipt=json.loads((out/'training-receipt.json').read_text());assert receipt['preflight'] and not receipt['test_read'] and receipt['validation'] is None
    for name,digest in receipt['bound_input_sha256'].items():assert sha(Path(name))==digest
    for name,digest in receipt['output_sha256'].items():assert sha(out/name)==digest
    assert sha(out/'best.pt')==receipt['checkpoint_sha256']
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=True);E=initial['E'].numpy().copy();R=initial['R'].numpy();centers=E.copy()
    keys,refs,_=load_scope(root,cfg,job['seed'],job['protection_scope']);steps=json.loads((out/'steps.json').read_text())
    with np.load(out/'shared-replay.npz') as z:
        order=z['entity_order'];ids=z['mutable_ids'];before=z['before'];after=z['after'];assert len(order)==len(steps)==cfg['preflight_updates']
        old_ids=set(receipt['selected_old_rows']);checked=0
        for i,row in enumerate(steps):
            selected=ids[i][ids[i]>=0];assert selected.tolist()==row['mutable_ids'] and selected[-1]==order[i]==row['entity']
            assert np.array_equal(E[selected],before[i,:len(selected)])
            if len(selected)==2:assert selected[0] in old_ids
            if not row['receipt']['accepted']:assert np.array_equal(before[i,:len(selected)],after[i,:len(selected)])
            E[selected]=after[i,:len(selected)]
            move=np.linalg.norm(E[selected]-centers[selected],axis=1);assert np.all(move<=np.asarray(row['radii'])+cfg['solver']['movement_tolerance'])
            assert hashlib.sha256(E.tobytes()).hexdigest()==row['committed_state_sha256']
            assert row['full_scope_max_rank']<=cfg['k'];checked+=len(selected)
        final=torch.load(out/'best.pt',map_location='cpu',weights_only=True);assert np.array_equal(E,final['E'].numpy()) and np.array_equal(R,final['R'].numpy())
        unused=np.setdiff1d(np.arange(7505),list(old_ids));assert np.array_equal(E[unused],centers[unused])
        # Independent CPU tree across every reference query at the terminal model.
        by_query={}
        for q,g in refs:by_query.setdefault(int(q),[]).append(int(g))
        counts=[];padded=1<<(E.shape[1]-1).bit_length()
        for qi,(h,r) in enumerate(keys):
            tree=np.zeros((len(E),padded));tree[:,:E.shape[1]]=E*(E[h]*R[r])
            while tree.shape[1]>1:tree=tree[:,0::2]+tree[:,1::2]
            scores=tree[:,0];cut=min(scores[g] for g in by_query[qi]);count=int((scores>cut).sum());assert count<cfg['k'],(qi,count)
            counts.append(count)
            if (qi+1)%8192==0:print('SHARED_PREFLIGHT_CPU_SCOPE',job['id'],qi+1,len(keys),round(time.monotonic()-started,2),flush=True)
        assert np.array_equal(counts,z['terminal_counts'])
    result=dict(status='PASS_PREFLIGHT_PATCH_REPLAY_AND_ALL_QUERY_CPU_TERMINAL_RANKS',failures=[],preflight=True,
                patches_replayed=len(steps),modified_row_steps_replayed=checked,terminal_query_groups_replayed=len(keys),reference_members_retained=len(refs),
                terminal_cpu_counts_exact=True,absolute_budgets_checked=True,unselected_old_rows_unchanged=True,test_read=False,valid_read=False,
                elapsed_s=time.monotonic()-started,receipt_sha256=sha(out/'training-receipt.json'),
                limits=['Preflight only; not all3753 graph updates or scientific utility.',
                        'All-step GPU guards logged and terminal entire CPU scope replayed; intermediate CPU full scopes not repeated.'])
    summary=dict(result=receipt,audit=result);dest=root/cfg['output_directory']/job['id'];dest.mkdir(parents=True,exist_ok=False)
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2));(dest/'audit.json').write_text(json.dumps(result,indent=2));(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);p.add_argument('--stage',choices=['prepare','audit'],required=True)
    a=p.parse_args();root=a.root.resolve();cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job);sys.path.insert(0,str(root/'src'))
    if a.stage=='prepare':
        from shared_rank_graph import train_rank
        train_rank(root,cfg,job,preflight=True)
    else:audit(root,cfg,job)
