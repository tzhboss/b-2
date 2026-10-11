"""Cumulative native KGEditor adaptation and common novel-entity rank audit.

The native hypernetwork remains unchanged. Applying its independent-edit
proposals cumulatively is an explicit task adaptation, not the author protocol.
"""
import hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from common_incremental import Data,metrics
from common_training import sha
from lm_common_base import cached_npz,tensor_batch,encoder_init,encode_chunk
from lm_editor_bridge_v2 import (construct_editor,descriptions,condition_features,
                                 meta_tables,model_hash)


def setup_seed(seed):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def encode_queries(root,cfg,data,queries):
    E,R,bound=descriptions(root,cfg,data)
    encoder_init(str(root/cfg['pretraining_assets']),E,R,cfg['max_sequence_length'],11258)
    parts=[encode_chunk(queries[i:i+cfg['encoding_chunk_rows']])
           for i in range(0,len(queries),cfg['encoding_chunk_rows'])]
    return dict(input_ids=np.concatenate([p[0] for p in parts]),
                attention_mask=np.concatenate([p[1] for p in parts]),
                mask_positions=np.concatenate([p[2] for p in parts]),queries=queries),bound


def prepare_online(root,cfg):
    started=time.monotonic();data=Data(root,cfg);queries=data.triples(1,'train').numpy()
    assert len(queries)==37214 and ((queries[:,0]>=7505)|(queries[:,2]>=7505)).all()
    assert set(queries[:,[0,2]].reshape(-1)[queries[:,[0,2]].reshape(-1)>=7505])==set(range(7505,11258))
    table,bound=encode_queries(root,cfg,data,queries)
    from transformers import BertTokenizer
    tokenizer=BertTokenizer.from_pretrained(str(root/cfg['pretraining_assets']))
    E,R,text_bound=descriptions(root,cfg,data)
    rng=np.random.default_rng(cfg['online_corruption_seed'])
    wrong=(queries[:,2]+rng.integers(1,11258,size=len(queries)))%11258
    cond,mask=condition_features(tokenizer,queries,wrong,E,R,cfg['max_sequence_length'])
    out=root/cfg['online_data_cache'];out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'new-train.npz',**table,wrong=wrong,cond_input_ids=cond,cond_attention_mask=mask)
    receipt=dict(status='PASS_NOVEL_TRAIN_ENCODING',test_read=False,queries=len(queries),new_entities=3753,
                 wrong_equals_gold=int((wrong==queries[:,2]).sum()),input_sha256=dict(data.bound,**bound,**text_bound),
                 output_sha256=sha(out/'new-train.npz'),elapsed_s=time.monotonic()-started,
                 conditions='Uniform different full-pool label, fixed TRAIN-only seed; native condition text/tokenizer')
    assert receipt['wrong_equals_gold']==0
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('KGEDITOR_NOVEL_TRAIN_READY',flush=True)


def online_tables(root,cfg):
    directory=root/cfg['online_data_cache'];receipt=json.loads((directory/'receipt.json').read_text())
    assert receipt['status']=='PASS_NOVEL_TRAIN_ENCODING' and not receipt['test_read']
    assert sha(directory/'new-train.npz')==receipt['output_sha256']
    for name,digest in receipt['input_sha256'].items():assert sha(name)==digest
    return cached_npz(directory/'new-train.npz'),receipt


def online_batch(new,old,edit_ids,stable_ids,device):
    b={key:torch.as_tensor(np.concatenate([old[key][stable_ids],new[key][edit_ids]]).astype(np.int64),device=device)
       for key in ['input_ids','attention_mask']}
    b['label']=torch.as_tensor(np.r_[old['queries'][stable_ids,2],new['queries'][edit_ids,2]],dtype=torch.long,device=device)
    for source,target in [('cond_input_ids','cond_input_ids'),('cond_attention_mask','cond_attention_mask')]:
        b[target]=torch.as_tensor(np.concatenate([np.zeros((len(stable_ids),new[source].shape[1]),dtype=np.int64),new[source][edit_ids]]).astype(np.int64),device=device)
    return b


def commit_delta(editor,include,batch):
    with torch.no_grad():_,delta=editor.get_logits_orig_params_dict(batch)
    assert set(delta)==include,'Native proposal changed editable parameter scope'
    named=dict(editor.ex_model.named_parameters());norms={}
    with torch.no_grad():
        for name,value in delta.items():
            assert value.shape==named[name].shape and torch.isfinite(value).all(),'Invalid native delta'
            candidate=named[name]+value;assert torch.isfinite(candidate).all(),'Nonfinite cumulative model'
            named[name].copy_(candidate);norms[name]=float(torch.linalg.vector_norm(value).cpu())
    return {name:value.detach().cpu().clone() for name,value in delta.items()},norms


def train_online(root,cfg,job,preflight=False,cpu_smoke=False):
    setup_seed(job['seed']);started=time.monotonic();device='cpu' if cpu_smoke else 'cuda:0'
    base=root/cfg['base_directory']/f"lm-base-s{job['seed']}"/'best.pt'
    meta=root/'artifacts'/('lm-editor-preflight' if preflight else 'lm-editor-meta')/job['id']
    if cpu_smoke:base=root/cfg['smoke_base_path'];meta=root/cfg['smoke_meta_directory']
    else:assert json.loads(base.with_name('audit.json').read_text())['status']=='PASS_CLEAN_BASE_CPU_GPU_FIXTURE'
    mt=json.loads((meta/'terminal.json').read_text());assert mt['status']=='PASS_NATIVE_META_AND_RESTORE_ONLY'
    assert mt['base_sha256']==sha(base) and mt['meta_checkpoint_sha256']==sha(meta/'best.pt')
    if not cpu_smoke:assert mt['preflight']==preflight and mt['epochs_executed']==(1 if preflight else cfg['meta_epochs'])
    restore=torch.load(str(meta/'best.pt'),map_location='cpu')
    editor,include=construct_editor(root,cfg,job,base,device,restore);editor.eval();editor.entity_id_ed=30522+11258
    learner_hash=model_hash(editor.ex_learner)
    clean=torch.load(str(base),map_location='cpu');initial_edit={n:p.detach().cpu().clone() for n,p in editor.ex_model.named_parameters() if n in include}
    new,receipt=online_tables(root,cfg);old,split,_,_,_=meta_tables(root,cfg)
    gen=torch.Generator().manual_seed(job['seed']+71);order=torch.randperm(len(new['queries']),generator=gen).numpy()
    stable_order=np.random.default_rng(job['seed']+171).permutation(split['base_ids'])
    stage='lm-editor-online-smoke' if cpu_smoke else ('lm-editor-online-preflight' if preflight else 'incremental')
    out=root/'artifacts'/stage/job['id'];out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'online-order.npz',query_ids=order,stable_ids=stable_order)
    records=[];witnesses=[];processed=[];limit=cfg['cpu_smoke_steps'] if cpu_smoke else (cfg['online_preflight_steps'] if preflight else len(order)//cfg['edit_num'])
    for step,off in enumerate(range(0,len(order),cfg['edit_num'])):
        if step>=limit:break
        assert time.monotonic()-started<cfg['online_budget_seconds'],'Registered online budget exceeded'
        edits=order[off:off+cfg['edit_num']];assert len(edits)==cfg['edit_num']
        sid=step*cfg['stable_batch_size'];stable=stable_order[sid:sid+cfg['stable_batch_size']];assert len(stable)==cfg['stable_batch_size']
        batch=online_batch(new,old,edits,stable,device)
        before={n:p.detach().cpu().clone() for n,p in editor.ex_model.named_parameters() if n in include} if step<cfg['audit_online_step_witnesses'] else None
        delta,norms=commit_delta(editor,include,batch);processed.extend(edits.tolist())
        if before is not None:witnesses.append(dict(step=step,edit_ids=edits,stable_ids=stable,batch={k:v.cpu() for k,v in batch.items()},before=before,delta=delta))
        records.append(dict(step=step,edit_query_ids=edits.tolist(),delta_norms=norms,elapsed_s=time.monotonic()-started))
        if step%cfg['online_progress_every_steps']==0:
            (out/'online-progress.json').write_text(json.dumps(records[-1],indent=2));print('KGEDITOR_ONLINE_STEP',job['id'],step,len(order)//cfg['edit_num'],round(time.monotonic()-started,2),flush=True)
    assert model_hash(editor.ex_learner)==learner_hash,'Online stage trained or modified the learner'
    unchanged=0
    for n,p in editor.ex_model.state_dict().items():
        if n not in include:assert torch.equal(p.detach().cpu(),clean['state_dict'][n]),n;unchanged+=1
    processed=np.asarray(processed);assert len(np.unique(processed))==len(processed)
    if not preflight and not cpu_smoke:assert np.array_equal(np.sort(processed),np.arange(37214))
    validation=[]
    if not cpu_smoke:
        vd=Data(root,cfg);vg=torch.Generator().manual_seed(cfg['validation_seed'])
        for snapshot in (0,1):
            q=vd.triples(snapshot,'valid').numpy();ids=torch.randperm(len(q),generator=vg)[:cfg['validation_queries_per_split']].numpy()
            table,_=encode_queries(root,cfg,vd,q[ids]);vr,_=rank_stream(editor.ex_model,table,device,cfg['evaluation_batch_size'])
            validation.append(dict(snapshot=snapshot,query_ids=ids.tolist(),**metrics(vr[11258])))
    torch.save(dict(state_dict={n:v.detach().cpu() for n,v in editor.ex_model.state_dict().items()},config=clean['config'],seed=job['seed']),out/'best.pt')
    torch.save(witnesses,out/'online-step-witnesses.pt')
    (out/'online-steps.json').write_text(json.dumps(records))
    result=dict(status='TERMINAL_NATIVE_CUMULATIVE_ONLINE',id=job['id'],method='KGEditor',kind='EXBert',dimension=768,seed=job['seed'],
                test_read=False,preflight=preflight,cpu_smoke=cpu_smoke,elapsed_s=time.monotonic()-started,steps=len(records),directed_train_queries=len(processed),
                base_sha256=sha(base),meta_checkpoint_sha256=sha(meta/'best.pt'),checkpoint_sha256=sha(out/'best.pt'),editable_names=sorted(include),
                noneditable_state_tensors_unchanged=unchanged,learner_unchanged=True,meta_elapsed_s=mt['elapsed_s'],
                base_elapsed_s=json.loads(base.with_name('terminal.json').read_text())['elapsed_s'],
                validation=validation,checkpoint_selection='Fixed single-sweep endpoint; VALID reported without online selection',
                input_sha256=receipt['input_sha256'],online_data_receipt_sha256=sha(root/cfg['online_data_cache']/'receipt.json'),
                output_sha256={n:sha(out/n) for n in ['online-order.npz','online-step-witnesses.pt','online-steps.json']},
                limits=['Native independent edit proposals accumulated into one shared model; explicit task adaptation.',
                        'No online learner optimization, clipping, rescaling or rollback; numerical failures retained.',
                        'EXBert language pretraining/text and architecture differ from embedding baselines.',
                        'One full new TRAIN sweep; not matched to baseline SGD epochs/cost.'])
    (out/'training-receipt.json').write_text(json.dumps(result,indent=2));print('KGEDITOR_ONLINE_TERMINAL',job['id'],flush=True)
    if cpu_smoke:audit_online_scope(root,cfg,job,out,base,meta,cpu_smoke=True)
    return result


def load_model(root,cfg,path,device):
    import sys
    sys.path.insert(0,str((root/cfg['source_root']/cfg['native_source']).resolve()))
    from transformers import BertConfig
    from src.models.modeling_bert import EXBertForMaskedLM
    c=torch.load(str(path),map_location='cpu');model=EXBertForMaskedLM(BertConfig.from_dict(c['config']))
    model.load_state_dict(c['state_dict']);return model.to(device).eval()


@torch.no_grad()
def rank_stream(model,table,device,batch_size,pools=(11258,),fixture_ids=()):
    ranks={p:[] for p in pools};fixtures={};fixture_ids=set(fixture_ids)
    for off in range(0,len(table['queries']),batch_size):
        ids=np.arange(off,min(off+batch_size,len(table['queries'])));b=tensor_batch(table,ids,device)
        output=model(**b).logits;pos=torch.as_tensor(table['mask_positions'][ids].astype(np.int64),device=device)
        s=output[torch.arange(len(ids),device=device),pos,30522:30522+max(pools)]
        assert torch.isfinite(s).all(),'Nonfinite evaluation logits'
        target=torch.as_tensor(table['queries'][ids,2],device=device)
        gold=s[torch.arange(len(ids),device=device),target]
        for pool in pools:ranks[pool].append((1+(s[:,:pool]>gold[:,None]).sum(1)).cpu().numpy())
        for j,qid in enumerate(ids):
            if int(qid) in fixture_ids:fixtures[int(qid)]=s[j].cpu().numpy()
    return {p:np.concatenate(v) for p,v in ranks.items()},fixtures


def evaluate(root,cfg,job):
    setup_seed(job['seed']);started=time.monotonic();device='cuda:0';out=root/'artifacts/incremental'/job['id']
    tr=json.loads((out/'training-receipt.json').read_text());assert not tr['test_read'] and not tr['preflight'] and not tr['cpu_smoke']
    assert sha(out/'best.pt')==tr['checkpoint_sha256']
    data=Data(root,cfg);queries=[data.triples(s,'test').numpy() for s in (0,1)]
    tables=[encode_queries(root,cfg,data,q)[0] for q in queries]
    base=root/cfg['base_directory']/f"lm-base-s{job['seed']}"/'best.pt';assert sha(base)==tr['base_sha256']
    model=load_model(root,cfg,base,device)
    before,before_fixture=rank_stream(model,tables[0],device,cfg['evaluation_batch_size'],pools=(7505,11258),fixture_ids=range(32));del model;torch.cuda.empty_cache()
    model=load_model(root,cfg,out/'best.pt',device);after=[];fixtures=[]
    for table in tables:
        ranks,fs=rank_stream(model,table,device,cfg['evaluation_batch_size'],fixture_ids=range(32))
        after.append(ranks[11258]);fixtures.append(np.stack([fs[i] for i in range(32)]))
    old_before=before[7505];old_expanded=before[11258];old_after,new_after=after
    np.savez_compressed(out/'ranks.npz',old_queries=queries[0],new_queries=queries[1],old_before=old_before,old_expanded=old_expanded,old_after=old_after,new_after=new_after)
    fixture={k:np.concatenate([table[k][:32] for table in tables]) for k in ['input_ids','attention_mask','mask_positions','queries']}
    np.savez_compressed(out/'audit-fixture.npz',**fixture,scores=np.concatenate(fixtures),old_before_scores=np.stack([before_fixture[i] for i in range(32)]))
    protected=old_before<=10;expanded_good=old_expanded<=10
    summary=dict(id=job['id'],method=job['method'],kind='EXBert',dimension=768,seed=job['seed'],candidates=11258,
                 old_before=metrics(old_before),old_after=metrics(old_after),new_after=metrics(new_after),
                 old_top10_loss=float((old_after[protected]>10).mean()),old_top10_protected=int(protected.sum()),
                 same_pool_top10_loss=float((old_after[expanded_good]>10).mean()),mean_rank_displacement=float((old_after-old_before).mean()),
                 evaluation_elapsed_s=time.monotonic()-started,parameter_scope='Four extension weights; base E/R and all other state tensors unchanged')
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2))
    (out/'evaluation-receipt.json').write_text(json.dumps(dict(test_input_sha256=data.bound,output_sha256={n:sha(out/n) for n in ['ranks.npz','audit-fixture.npz','evaluation.json']}),indent=2))
    print('KGEDITOR_COMMON_EVALUATION_TERMINAL',job['id'],flush=True)


def audit_online_scope(root,cfg,job,out,base,meta,cpu_smoke=False):
    tr=json.loads((out/'training-receipt.json').read_text());assert not tr['test_read']
    for name,digest in tr['input_sha256'].items():assert sha(name)==digest
    assert sha(base)==tr['base_sha256'] and sha(meta/'best.pt')==tr['meta_checkpoint_sha256']
    assert sha(out/'best.pt')==tr['checkpoint_sha256']
    for name,digest in tr['output_sha256'].items():assert sha(out/name)==digest
    c=torch.load(str(out/'best.pt'),map_location='cpu');b=torch.load(str(base),map_location='cpu');include=set(tr['editable_names']);scope_checks=0
    for n,p in c['state_dict'].items():
        assert torch.isfinite(p).all(),n
        if n not in include:assert torch.equal(p,b['state_dict'][n]),n;scope_checks+=1
    new,_=online_tables(root,cfg);old,split,_,_,_=meta_tables(root,cfg)
    order=cached_npz(out/'online-order.npz');gen=torch.Generator().manual_seed(job['seed']+71)
    assert np.array_equal(order['query_ids'],torch.randperm(len(new['queries']),generator=gen).numpy())
    assert np.array_equal(order['stable_ids'],np.random.default_rng(job['seed']+171).permutation(split['base_ids']))
    records=json.loads((out/'online-steps.json').read_text());seen=np.asarray([v for row in records for v in row['edit_query_ids']])
    assert len(records)==tr['steps'] and len(seen)==tr['directed_train_queries'] and np.array_equal(seen,order['query_ids'][:len(seen)])
    if not cpu_smoke:assert len(seen)==37214 and len(np.unique(seen))==37214
    restore=torch.load(str(meta/'best.pt'),map_location='cpu');editor,names=construct_editor(root,cfg,job,base,'cpu',restore)
    assert names==include;editor.eval();editor.entity_id_ed=30522+11258
    witnesses=torch.load(str(out/'online-step-witnesses.pt'),map_location='cpu');assert len(witnesses)==cfg['audit_online_step_witnesses']
    max_error=0.;named=dict(editor.ex_model.named_parameters());previous=None
    for w in witnesses:
        off=w['step']*cfg['edit_num'];sid=w['step']*cfg['stable_batch_size']
        assert np.array_equal(w['edit_ids'],order['query_ids'][off:off+cfg['edit_num']])
        assert np.array_equal(w['stable_ids'],order['stable_ids'][sid:sid+cfg['stable_batch_size']])
        batch=online_batch(new,old,w['edit_ids'],w['stable_ids'],'cpu')
        assert all(torch.equal(v,w['batch'][k]) for k,v in batch.items())
        for n in include:
            assert torch.equal(w['before'][n],b['state_dict'][n] if previous is None else previous[n])
            with torch.no_grad():named[n].copy_(w['before'][n])
        with torch.no_grad():_,delta=editor.get_logits_orig_params_dict(batch)
        assert set(delta)==include
        for n in include:
            error=float((delta[n]-w['delta'][n]).abs().max());max_error=max(max_error,error)
            assert error<=cfg['audit_delta_tolerance'],'Native delta CPU replay mismatch'
        previous={n:w['before'][n]+w['delta'][n] for n in include}
    result=dict(scope_checks=scope_checks,full_train_query_coverage=len(seen),native_steps_replayed=len(witnesses),max_native_delta_error=max_error,
                alias_receipt=editor.alias_receipt,limits=['Two native CPU step replays, full TRAIN coverage, parameter scope; not independent full optimizer retraining.'])
    if cpu_smoke:
        assert all(torch.equal(c['state_dict'][n],previous[n]) for n in include)
        result['status']='PASS_NATIVE_CUMULATIVE_CPU_SMOKE_ONLY';(out/'audit.json').write_text(json.dumps(result,indent=2));print('KGEDITOR_CUMULATIVE_CPU_SMOKE_PASS',flush=True)
    return result


def audit(root,cfg,job):
    setup_seed(job['seed']);out=root/'artifacts/incremental'/job['id'];base=root/cfg['base_directory']/f"lm-base-s{job['seed']}"/'best.pt';meta=root/'artifacts/lm-editor-meta'/job['id']
    failures=[]
    try:
        result=audit_online_scope(root,cfg,job,out,base,meta)
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
        result.update(status='PASS_COMMON_LM_ONLINE_AND_RANK_AUDIT',failures=[],candidate_score_checks=int(replay.size)+32*11258,max_abs_logit_error=error,max_initial_logit_error=before_error,rank_mismatches=0,old_queries=74428,new_queries=12406)
    except Exception as exc:
        result=dict(status='FAIL_RETAINED',failures=[repr(exc)])
        (out/'audit.json').write_text(json.dumps(result,indent=2));raise
    (out/'audit.json').write_text(json.dumps(result,indent=2));print('KGEDITOR_COMMON_AUDIT_PASS',job['id'],flush=True)

