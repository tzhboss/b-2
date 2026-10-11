"""Immutable foundation replay with an explicitly registered CPU precision."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from common_training import sha
from lm_common_base import cached_npz,base_validate
from lm_editor_common_v2 import load_model,setup_seed


def audit(root,cfg,job):
    started=time.monotonic();setup_seed(job['seed'])
    source=root/cfg['source_base_directory']/job['id'];out=root/cfg['output_base_directory']/job['id']
    original=json.loads((source/'audit.json').read_text());term=json.loads((source/'terminal.json').read_text())
    assert not term['test_read'] and not term['new_snapshot_facts_read'] and not term['meta_train_facts_fitted_in_base']
    for name,digest in term['input_sha256'].items():assert sha(name)==digest,name
    assert sha(source/'best.pt')==term['checkpoint_sha256']
    original_hash=sha(source/'audit.json');fixture=cached_npz(source/'valid-fixture.npz')
    model=load_model(root,cfg,source/'best.pt','cpu')
    new=model.bert.embeddings.word_embeddings.weight[30522+7505:30522+11258].detach().numpy()
    assert hashlib.sha256(new.tobytes()).hexdigest()==term['new_entity_initial_sha256']
    checkpoint=torch.load(str(source/'best.pt'),map_location='cpu');records=json.loads((source/'training.json').read_text())
    chosen=max(records,key=lambda r:r['old_validation_mrr']);assert chosen['epoch']==checkpoint['epoch']==term['best_epoch']
    model=model.to(dtype=getattr(torch,cfg['cpu_precision']))
    _,scores,ranks=base_validate(model,fixture,torch.device('cpu'),batch_size=cfg['cpu_batch_size'])
    error=float(np.abs(scores-fixture['scores']).max());mismatch=int((ranks!=fixture['ranks']).sum())
    gpu_model=load_model(root,cfg,source/'best.pt','cuda:0')
    _,gpu_scores,gpu_ranks=base_validate(gpu_model,fixture,torch.device('cuda:0'),batch_size=cfg['gpu_batch_size'])
    gpu_error=float(np.abs(gpu_scores-fixture['scores']).max());gpu_mismatch=int((gpu_ranks!=fixture['ranks']).sum())
    failures=[]
    if error>cfg['audit_logit_tolerance']:failures.append('Registered CPU precision score tolerance')
    if mismatch:failures.append('Registered CPU precision exact ranking mismatch')
    if gpu_error!=0 or gpu_mismatch:failures.append('GPU float32 exact score/rank restoration')
    assert sha(source/'audit.json')==original_hash
    out.mkdir(parents=True,exist_ok=False)
    # Hard links preserve exact checkpoint bytes; no scientific weights are changed.
    for name in ['best.pt','terminal.json','training.json','valid-fixture.npz']:os.link(source/name,out/name)
    result=dict(status='PASS_CLEAN_BASE_CPU_GPU_FIXTURE' if not failures else 'FAIL_RETAINED',failures=failures,
                queries=len(ranks),candidate_score_checks=int(scores.size),max_abs_logit_error=error,rank_mismatches=mismatch,
                cpu_precision=cfg['cpu_precision'],cpu_batch_size=cfg['cpu_batch_size'],gpu_precision='float32',gpu_max_abs_logit_error=gpu_error,gpu_rank_mismatches=gpu_mismatch,
                original_source_audit_status=original['status'],original_source_audit_sha256=original_hash,original_source_audit_unchanged=True,
                source_checkpoint_sha256=sha(source/'best.pt'),source_fixture_sha256=sha(source/'valid-fixture.npz'),elapsed_s=time.monotonic()-started,
                limits=['Explicitly registered float64 CPU replay; original float32 CPU audit FAIL retained.',
                        'Native GPU float32 scores and all fixture strict ranks must reproduce exactly.',
                        'Foundation-only replay, not independent optimizer retraining or full KGEditor completion.'])
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    assert not failures,result
    return result
