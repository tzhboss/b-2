"""Native GPU delta replay plus unchanged CPU common-inference checks."""
import argparse,json,sys,gc
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import lm_editor_common_v2 as api
from lm_editor_common_v2 import *

class NativeReplay:
    def __init__(self,editor):object.__setattr__(self,'editor',editor)
    def __getattr__(self,name):return getattr(self.editor,name)
    def __setattr__(self,name,value):setattr(self.editor,name,value)
    def eval(self):self.editor.eval();return self
    def get_logits_orig_params_dict(self,batch):
        orig,delta=self.editor.get_logits_orig_params_dict({k:v.to('cuda:0') for k,v in batch.items()})
        return orig,{k:v.detach().cpu() for k,v in delta.items()}

def prepare(root,source_cfg,source_job,out,base,meta,dest):
    api.setup_seed(source_job['seed']);original=api.construct_editor
    def factory(root,cfg,job,base,device,restore=None):
        editor,include=original(root,cfg,job,base,'cuda:0',restore)
        return NativeReplay(editor),include
    source_audit_hash=sha(out/'audit.json')
    api.construct_editor=factory
    try:result=api.audit_online_scope(root,source_cfg,source_job,out,base,meta)
    finally:api.construct_editor=original
    result['limits']=[v.replace('Two native CPU step replays','Two native GPU step replays') for v in result['limits']]
    result.update(native_delta_replay_device='GPU float32',original_native_delta_tolerance=source_cfg['audit_delta_tolerance'],
                  source_original_audit_sha256=source_audit_hash,source_original_audit=json.loads((out/'audit.json').read_text()),
                  source_checkpoint_sha256=sha(out/'best.pt'),source_training_receipt_sha256=sha(out/'training-receipt.json'),
                  retrained=False,limits=result['limits']+['Native steps replayed on GPU; original CPU delta failures remain unchanged.'])
    assert sha(out/'audit.json')==source_audit_hash
    dest.mkdir(parents=True,exist_ok=False);(dest/'receipt.json').write_text(json.dumps(result,indent=2))
    print('NATIVE_GPU_DELTA_SCOPE_PASS',source_job['id'],result['max_native_delta_error'],flush=True)

def audit(root,cfg,job):
    source_cfg=json.loads((root/job['source_config']).read_text());source_job=next(j for j in source_cfg['jobs'] if j['id']==job['id'])
    setup_seed(job['seed']);out=root/'artifacts/incremental'/job['id'];base=root/source_cfg['base_directory']/f"lm-base-s{job['seed']}"/'best.pt';meta=root/'artifacts/lm-editor-meta'/job['id']
    dest=root/cfg['output_directory']/job['id'];result=json.loads((dest/'receipt.json').read_text())
    assert sha(out/'audit.json')==result['source_original_audit_sha256'] and sha(out/'best.pt')==result['source_checkpoint_sha256']
    assert sha(out/'training-receipt.json')==result['source_training_receipt_sha256']
    cfg=source_cfg
    receipt=json.loads((out/'evaluation-receipt.json').read_text())
    for name,digest in receipt['test_input_sha256'].items():assert sha(name)==digest
    for name,digest in receipt['output_sha256'].items():assert sha(out/name)==digest
    ranks=cached_npz(out/'ranks.npz');data=Data(root,cfg)
    assert np.array_equal(ranks['old_queries'],data.triples(0,'test').numpy()) and np.array_equal(ranks['new_queries'],data.triples(1,'test').numpy())
    row=json.loads((out/'evaluation.json').read_text());assert row['candidates']==11258
    for key in ['old_before','old_after','new_after']:
        assert ranks[key].min()>=1 and ranks[key].max()<=(7505 if key=='old_before' else 11258)
        assert row[key]==metrics(ranks[key])
    p=ranks['old_before']<=10;e=ranks['old_expanded']<=10
    assert row['old_top10_protected']==int(p.sum()) and row['old_top10_loss']==float((ranks['old_after'][p]>10).mean())
    assert row['same_pool_top10_loss']==float((ranks['old_after'][e]>10).mean())
    assert row['mean_rank_displacement']==float((ranks['old_after']-ranks['old_before']).mean())
    fixture=cached_npz(out/'audit-fixture.npz');expected=np.concatenate([ranks['old_queries'][:32],ranks['new_queries'][:32]])
    assert np.array_equal(fixture['queries'],expected)
    encoded,_=encode_queries(root,cfg,data,expected)
    for key in ['input_ids','attention_mask','mask_positions']:assert np.array_equal(encoded[key],fixture[key])
    before_table={k:v[:32] for k,v in fixture.items() if k in ['input_ids','attention_mask','mask_positions','queries']}
    bm=load_model(root,cfg,base,'cpu');br,bfs=rank_stream(bm,before_table,'cpu',cfg['cpu_audit_batch_size'],pools=(7505,11258),fixture_ids=range(32))
    before_error=float(np.abs(np.stack([bfs[i] for i in range(32)])-fixture['old_before_scores']).max())
    assert before_error<=cfg['audit_logit_tolerance'],'CPU/GPU initial logits mismatch'
    assert np.array_equal(br[7505],ranks['old_before'][:32]) and np.array_equal(br[11258],ranks['old_expanded'][:32])
    del bm
    model=load_model(root,cfg,out/'best.pt','cpu');rr,fs=rank_stream(model,fixture,'cpu',cfg['cpu_audit_batch_size'],fixture_ids=range(64))
    replay=np.stack([fs[i] for i in range(64)]);error=float(np.abs(replay-fixture['scores']).max())
    assert error<=cfg['audit_logit_tolerance'],'CPU/GPU final logits mismatch'
    saved=1+(fixture['scores']>fixture['scores'][np.arange(64),expected[:,2]][:,None]).sum(1)
    assert np.array_equal(saved,rr[11258]) and np.array_equal(saved,np.r_[ranks['old_after'][:32],ranks['new_after'][:32]])
    result.update(status='PASS_COMMON_LM_NATIVE_GPU_DELTA_CPU_RANK_AUDIT',failures=[],candidate_score_checks=int(replay.size)+32*11258,max_abs_logit_error=error,max_initial_logit_error=before_error,rank_mismatches=0,old_queries=74428,new_queries=12406)
    (dest/'audit.json').write_text(json.dumps(result,indent=2));assert sha(out/'audit.json')==result['source_original_audit_sha256']
    print('QUALIFIED_NATIVE_GPU_DELTA_CPU_RANK_PASS',job['id'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);p.add_argument('--stage',choices=['prepare','audit'],required=True);a=p.parse_args()
    root=a.root.resolve();cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
    source_cfg=json.loads((root/job['source_config']).read_text());source_job=next(j for j in source_cfg['jobs'] if j['id']==a.job)
    out=root/'artifacts/incremental'/job['id'];base=root/source_cfg['base_directory']/f"lm-base-s{job['seed']}"/'best.pt';meta=root/'artifacts/lm-editor-meta'/job['id'];dest=root/cfg['output_directory']/job['id']
    try:
        if a.stage=='prepare':prepare(root,source_cfg,source_job,out,base,meta,dest)
        else:audit(root,cfg,job)
    except Exception as exc:
        dest.mkdir(parents=True,exist_ok=True);(dest/'audit.json').write_text(json.dumps(dict(status='FAIL_RETAINED',failures=[repr(exc)]),indent=2));raise
