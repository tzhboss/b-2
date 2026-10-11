"""TRAIN-only scoped protection conflict diagnosis and constructive toy check."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch


def run(root,cfg):
    sys.path.insert(0,str(root/'src'))
    from common_incremental import Data
    from common_training import sha
    started=time.monotonic();torch.set_num_threads(2)
    source_cfg=json.loads((root/cfg['task_config']).read_text());data=Data(root,source_cfg)
    q=data.triples(1,'train').numpy();q=q[(q[:,0]<7505)&(q[:,2]>=7505)]
    rows=[];bound={cfg['task_config']:sha(root/cfg['task_config'])}
    for seed in cfg['seeds']:
        feature=root/source_cfg['feature_directory']/f's{seed}';replay=root/'artifacts/incremental'/f'proposal-qppred-s{seed}'/'projection-replay.npz'
        receipt=json.loads((feature/'receipt.json').read_text());assert receipt['status']=='PASS_TRAIN_ONLY_GUARDS' and not receipt['test_read']
        assert sha(feature/'guards.npz')==receipt['output_sha256']
        with np.load(feature/'guards.npz') as f,np.load(replay) as z:
            predmask=(f['pred_old_rank']==10)&(z['initial_pred_counts']==9)
            pred={tuple(k):float(cut) for k,cut in zip(f['pred_queries'][predmask],f['pred_cut'][predmask])}
            truemask=(f['true_old_rank']==10)&(z['initial_true_counts']==9)
            true={tuple(k[:2]):float(cut) for k,cut in zip(f['true_queries'][truemask],f['true_gold'][truemask])}
        mp=np.asarray([tuple(row[:2]) in pred for row in q]);mt=np.asarray([tuple(row[:2]) in true for row in q]);locked=mp|mt
        cutoff=np.asarray([pred.get(tuple(row[:2]),true.get(tuple(row[:2]),np.nan)) for row in q[locked]])
        metrics=[];oldE=None;R=None
        for template in cfg['protected_models']:
            path=root/'artifacts/incremental'/template.format(seed=seed)
            audit=json.loads((path/'audit.json').read_text());assert audit['status'].startswith('PASS') and not audit['failures']
            tr=json.loads((path/'training-receipt.json').read_text());assert sha(path/'best.pt')==tr['checkpoint_sha256']
            c=torch.load(path/'best.pt',map_location='cpu',weights_only=True);E=c['E'].numpy();rr=c['R'].numpy()
            if oldE is None:oldE=E[:7505].copy();R=rr.copy()
            else:assert np.array_equal(oldE,E[:7505]) and np.array_equal(R,rr)
            part=q[locked];Q=E[part[:,0]]*R[part[:,1]];gold=np.sum(Q*E[part[:,2]],axis=1)
            # Independent scoring of all old candidates gives a full-model rank
            # lower bound, irrespective of other new candidates.
            counts=[]
            for off in range(0,len(part),cfg['batch_size']):
                scores=Q[off:off+cfg['batch_size']]@oldE.T
                counts.extend((scores>gold[off:off+cfg['batch_size'],None]).sum(1).tolist())
            counts=np.asarray(counts)
            assert np.all(gold<cutoff),'Strict-margin locked query unexpectedly permits a boundary tie'
            assert np.all(counts>=10),'Independent old-candidate scoring contradicts claimed TRAIN barrier'
            metrics.append(dict(model=template.format(seed=seed),facts_checked=len(part),all_new_gold_strictly_below_boundary=True,
                                minimum_old_competitor_rank_lower_bound=int((1+counts).min()),new_gold_top10_impossible_under_this_fixed_old_model=True))
            for name in ('best.pt','audit.json','training-receipt.json'):bound[str((path/name).relative_to(root))]=sha(path/name)
        rows.append(dict(seed=seed,new_train_old_head_new_gold_facts=len(q),unique_queries=len(np.unique(q[:,:2],axis=0)),
                         pred_boundary_locked_facts=int(mp.sum()),true_rank10_locked_facts=int(mt.sum()),union_locked_facts=int(locked.sum()),
                         pred_only_locked_facts=int((mp&~mt).sum()),union_fraction=float(locked.mean()),model_checks=metrics))
        for p in (feature/'guards.npz',feature/'receipt.json',replay):bound[str(p.relative_to(root))]=sha(p)
    # Constructive example: one old competitor moves, with real shared head
    # scoring and unchanged prediction membership for the unaffected relation.
    old=np.column_stack([np.arange(12,0,-1),np.r_[np.arange(1,10),0,-1,-2]]).astype(float)
    E=np.vstack([old,[-1.,-1.]]);before=E.copy();relations=np.array([[1/12,0.],[0.,1.]])
    def rank(model,h,r,t):
        score=(model[h]*relations[r])@model.T;return int(1+(score>score[t]).sum())
    prediction_before=np.argsort(-((E[0]*relations[1])@E.T))[:10]
    assert rank(E,0,0,9)==rank(E,0,1,9)==rank(E,8,0,9)==10
    E[8,0]=2.5;E[12,0]=5.
    assert rank(E,0,0,9)==rank(E,0,1,9)==rank(E,8,0,9)==10
    assert rank(E,0,0,12)<=10 and rank(before,0,0,12)>10
    prediction_after=np.argsort(-((E[0]*relations[1])@E.T))[:10]
    assert set(prediction_before)==set(prediction_after)
    assert not np.array_equal((before[8]*relations[0])@before.T,(E[8]*relations[0])@E.T)
    assert not np.array_equal((before[12]*relations[0])@before.T,(E[12]*relations[0])@E.T)
    assert rank(E,0,0,8)>10 # Affected prediction set changes, explicitly.
    bound.update(data.bound)
    for name,digest in bound.items():assert sha(root/name)==digest,name
    result=dict(status='PASS_TRAIN_SCOPE_DIAGNOSTIC_ONLY',test_read=False,seeds=rows,elapsed_s=time.monotonic()-started,input_sha256=bound,
                constructive_synthetic_example=dict(old_true_ranks_retained=True,new_answer_enters_top10=True,unaffected_relation_prediction_set_unchanged=True,
                                                    changed_old_entity_as_head_replayed=True,affected_old_prediction_set_changes=True,old_vector_movement=1.5),
                limits=['Descriptive diagnosis after exposed019/020; not a new preregistered scientific discovery.',
                        'Locked-query argument requires frozen oldE/R and strict boundary separation; ties are checked, not silently ruled out.',
                        'TRAIN sufficient barrier subset only; not full causal attribution for VALID/TEST loss.',
                        'Toy demonstrates possibility of competitor redistribution, not a trained graph method or all-query guarantee.',
                        'Unlabelled old competitor is not certified false knowledge; whole real protection scope remains required.'])
    out=root/cfg['output_directory'];out.mkdir(parents=True,exist_ok=False);(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args()
    run(a.root.resolve(),json.loads((a.root/a.config).read_text()))
