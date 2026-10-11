"""Clean old-graph EXBert preparation and training for the native KGEditor bridge.

Native text pairs and placeholder replacement are retained. Full entity metadata
is public, but no snapshot1 facts are read and new entity parameters stay at init.
"""
import hashlib,json,os,random,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
import torch
from common_incremental import Data
from common_training import sha

_encoder=None;_ent_text=None;_rel_text=None;_length=None;_offset=None
def encoder_init(assets,entity_text,relation_text,length,offset):
    global _encoder,_ent_text,_rel_text,_length,_offset
    from transformers import BertTokenizer
    _encoder=BertTokenizer.from_pretrained(assets)
    _ent_text=entity_text;_rel_text=relation_text;_length=length;_offset=offset

def encode_one(h,r):
    # Original solve_kgc uses two PAD placeholders. Relations retain original
    # even IDs; inverse directions are represented by the author's head mask.
    if r%2:
        text_a='[MASK]';text_b=_rel_text[r-1]+'[PAD]';text_c='[PAD] '+_ent_text[h]
        first=_encoder.sep_token.join([text_a,text_b]);second=text_c
        replacements=[_offset+(r-1),h]
    else:
        text_a='[PAD] ';text_b=_rel_text[r]+'[PAD]';text_c='[MASK] '+_ent_text[h]
        first=text_a;second=_encoder.sep_token.join([text_b,text_c]);replacements=[h,_offset+r]
    enc=_encoder(first,second,truncation='longest_first',max_length=_length,add_special_tokens=True)
    ids=enc['input_ids'];mask=enc['attention_mask'];used=0
    for i,value in enumerate(ids):
        if value==_encoder.pad_token_id:
            assert used<2,'Unexpected internal PAD in author text pair'
            ids[i]=30522+int(replacements[used]);used+=1
        if used==2:break
    assert used==2 and ids.count(103)==1 and all(value!=0 for value in ids)
    n=len(ids);assert n<=_length
    return ids+[0]*(_length-n),mask+[0]*(_length-n),ids.index(103)

def encode_chunk(queries):
    cache={};features=[]
    for h,r,t in queries:
        key=(int(h),int(r))
        if key not in cache:cache[key]=encode_one(*key)
        features.append(cache[key])
    return (np.asarray([v[0] for v in features],dtype=np.uint16),
            np.asarray([v[1] for v in features],dtype=np.uint8),
            np.asarray([v[2] for v in features],dtype=np.uint8))

def prepare(root,cfg):
    started=time.monotonic();data=Data(root,cfg)
    # Only old TRAIN and old VALID fact contents are loaded for this base stage.
    train=data.triples(0,'train').numpy();valid=data.triples(0,'valid').numpy()
    metadata=data.root/cfg['text_metadata_directory']
    names={v:k for k,v in data.entities[1].items()};relations={v:k for k,v in data.relations[1].items()}
    ent_raw=dict(x.split('\t',1) for x in (metadata/'entity2textlong.txt').read_text().splitlines())
    rel_raw=dict(x.split('\t',1) for x in (metadata/'relation2text.txt').read_text().splitlines())
    ent_text={i:ent_raw[name].replace('\\n',' ').replace('\\','') for i,name in names.items()}
    rel_text={i:rel_raw[name] for i,name in relations.items() if i%2==0}
    assert len(ent_text)==11258 and len(rel_text)==237
    rng=np.random.default_rng(cfg['meta_split_seed']);order=rng.permutation(len(train)//2)
    meta_train=order[:cfg['meta_train_original_facts']]
    meta_valid=order[cfg['meta_train_original_facts']:cfg['meta_train_original_facts']+cfg['meta_valid_original_facts']]
    excluded=set(np.concatenate([meta_train,meta_valid]).tolist())
    base_ids=np.asarray([i for i in range(len(train)) if i//2 not in excluded])
    meta_train_ids=np.asarray([j for i in meta_train for j in [2*i,2*i+1]])
    meta_valid_ids=np.asarray([j for i in meta_valid for j in [2*i,2*i+1]])
    g=torch.Generator().manual_seed(cfg['validation_seed']);valid_ids=torch.randperm(len(valid),generator=g)[:cfg['validation_queries']].numpy()
    out=root/cfg['data_cache'];out.mkdir(parents=True,exist_ok=False)
    args=(str(root/cfg['pretraining_assets']),ent_text,rel_text,cfg['max_sequence_length'],11258)
    with ProcessPoolExecutor(max_workers=cfg['encoding_workers'],mp_context=multiprocessing.get_context('spawn'),initializer=encoder_init,initargs=args) as pool:
        for name,queries,ids in [('old-train',train,np.arange(len(train))),('old-valid',valid[valid_ids],valid_ids)]:
            parts=list(pool.map(encode_chunk,[queries[i:i+cfg['encoding_chunk_rows']] for i in range(0,len(queries),cfg['encoding_chunk_rows'])]))
            x=np.concatenate([v[0] for v in parts]);m=np.concatenate([v[1] for v in parts]);pos=np.concatenate([v[2] for v in parts])
            np.savez_compressed(out/(name+'.npz'),input_ids=x,attention_mask=m,mask_positions=pos,queries=queries,query_ids=ids)
    np.savez_compressed(out/'old-train-meta-split.npz',base_ids=base_ids,meta_train_ids=meta_train_ids,meta_valid_ids=meta_valid_ids)
    bound=dict(data.bound)
    for name in ['entity2textlong.txt','relation2text.txt']:bound[str(metadata/name)]=sha(metadata/name)
    receipt=dict(status='PASS_CLEAN_OLD_TRAIN_ENCODING',test_read=False,new_snapshot_facts_read=False,
                 base_directed_queries=len(base_ids),meta_train_directed_queries=len(meta_train_ids),meta_valid_directed_queries=len(meta_valid_ids),
                 validation_queries=len(valid_ids),entities_text_coverage=len(ent_text),relations_text_coverage=len(rel_text),
                 elapsed_s=time.monotonic()-started,input_sha256=bound,output_sha256={f.name:sha(f) for f in out.iterdir()},
                 implementation_sha256=sha(Path(__file__)),limits=['Public entity descriptions and language-model pretraining are extra architecture information; not strict world-knowledge novelty.'])
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('LM_DATA_READY',json.dumps(receipt),flush=True)

def native_model(root,cfg):
    source=(root/cfg['source_root']/cfg['native_source']).resolve();sys.path.insert(0,str(source))
    from transformers import BertConfig
    from src.models.modeling_bert import EXBertForMaskedLM
    config=BertConfig.from_pretrained(str(root/cfg['pretraining_assets']));config.ex_size=cfg['ex_size'];config.kb_layer=cfg['kb_layers']
    model,info=EXBertForMaskedLM.from_pretrained(str(root/cfg['pretraining_assets']),config=config,output_loading_info=True)
    assert not info['error_msgs'] and not info['mismatched_keys']
    assert all('dense_in_ex' in k or 'dense_out_ex' in k for k in info['missing_keys'])
    model.resize_token_embeddings(30522+11258+474)
    return model,info

def tensor_batch(table,ids,device):
    return dict(input_ids=torch.as_tensor(table['input_ids'][ids].astype(np.int64),device=device),
                attention_mask=torch.as_tensor(table['attention_mask'][ids].astype(np.int64),device=device))

def cached_npz(path):
    # NpzFile decompresses a member on every indexing operation; materialize once.
    with np.load(path) as table:return {k:table[k] for k in table.files}

@torch.no_grad()
def base_validate(model,table,device,batch_size=32,candidates=7505):
    scores=[]
    for off in range(0,len(table['queries']),batch_size):
        ids=np.arange(off,min(off+batch_size,len(table['queries'])));b=tensor_batch(table,ids,device)
        output=model(**b).logits;pos=torch.as_tensor(table['mask_positions'][ids].astype(np.int64),device=device)
        value=output[torch.arange(len(ids),device=device),pos,30522:30522+candidates]
        scores.append(value.detach().cpu().numpy())
    s=np.concatenate(scores);gold=s[np.arange(len(s)),table['queries'][:,2]]
    rank=1+(s>gold[:,None]).sum(1)
    return float((1/rank).mean()),s,rank

def train_base(root,cfg,job,preflight=False):
    torch.set_num_threads(2);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed']);random.seed(job['seed']);np.random.seed(job['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();cache=root/cfg['data_cache'];receipt=json.loads((cache/'receipt.json').read_text())
    assert receipt['status']=='PASS_CLEAN_OLD_TRAIN_ENCODING' and not receipt['test_read'] and not receipt['new_snapshot_facts_read']
    for name,digest in receipt['output_sha256'].items():assert sha(cache/name)==digest
    data=cached_npz(cache/'old-train.npz');valid=cached_npz(cache/'old-valid.npz');split=cached_npz(cache/'old-train-meta-split.npz')
    out=root/'artifacts'/('lm-base-preflight' if preflight else 'lm-base')/job['id'];out.mkdir(parents=True,exist_ok=False)
    model,loading=native_model(root,cfg);model=model.to('cuda:0');device=torch.device('cuda:0')
    w=model.bert.embeddings.word_embeddings.weight;bias=model.cls.predictions.bias
    new_start=30522+7505;new_end=30522+11258
    new_initial=w[new_start:new_end].detach().clone();bias_initial=bias[new_start:new_end].detach().clone()
    initial_hash=hashlib.sha256(new_initial.cpu().numpy().tobytes()).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['base_learning_rate'],weight_decay=cfg['base_weight_decay'])
    epochs=1 if preflight else cfg['base_epochs'];rng=np.random.default_rng(job['seed']);best=-1.;records=[];best_epoch=0
    for epoch in range(1,epochs+1):
        model.train();permutation=rng.permutation(split['base_ids']);losses=[]
        for off in range(0,len(permutation),cfg['base_batch_size']):
            assert time.monotonic()-started<cfg['base_budget_seconds'],'Registered foundation training budget exceeded'
            ids=permutation[off:off+cfg['base_batch_size']];batch=tensor_batch(data,ids,device)
            output=model(**batch).logits;pos=torch.as_tensor(data['mask_positions'][ids].astype(np.int64),device=device)
            target=torch.as_tensor(data['queries'][ids,2].astype(np.int64),device=device)
            logits=output[torch.arange(len(ids),device=device),pos,30522:30522+7505]
            loss=torch.nn.functional.cross_entropy(logits,target);assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['base_gradient_clip']);optimizer.step()
            # The full vocabulary is metadata only. Even optimizer weight decay
            # must not modify new entity rows during old-graph foundation fitting.
            with torch.no_grad():w[new_start:new_end].copy_(new_initial);bias[new_start:new_end].copy_(bias_initial)
            losses.append(float(loss.detach()))
            if (off//cfg['base_batch_size'])%cfg['progress_every_steps']==0:print('LM_BASE_STEP',job['id'],epoch,off,len(permutation),round(time.monotonic()-started,2),flush=True)
            if preflight:break
        model.eval();mrr,scores,rank=base_validate(model,valid,device)
        if mrr>best:
            best=mrr;best_epoch=epoch
            torch.save(dict(state_dict={n:v.detach().cpu() for n,v in model.state_dict().items()},config=model.config.to_dict(),seed=job['seed'],epoch=epoch),out/'best.pt')
        records.append(dict(epoch=epoch,loss=float(np.mean(losses)),old_validation_mrr=mrr,elapsed_s=time.monotonic()-started))
        (out/'training.json').write_text(json.dumps(records,indent=2));print('LM_BASE_EPOCH',job['id'],epoch,mrr,flush=True)
    check=torch.load(str(out/'best.pt'),map_location=device);model.load_state_dict(check['state_dict']);model.eval()
    fixture={k:valid[k][:64] for k in ['input_ids','attention_mask','mask_positions','queries']}
    mrr,s,rank=base_validate(model,fixture,device,batch_size=16)
    np.savez_compressed(out/'valid-fixture.npz',**fixture,scores=s,ranks=rank)
    assert torch.equal(w[new_start:new_end],new_initial) and torch.equal(bias[new_start:new_end],bias_initial)
    term=dict(status='TERMINAL_CLEAN_OLD_GRAPH_BASE_ONLY',id=job['id'],seed=job['seed'],test_read=False,new_snapshot_facts_read=False,
              meta_train_facts_fitted_in_base=False,best_epoch=best_epoch,best_validation_mrr=best,epochs_executed=epochs,
              new_entity_rows_unchanged=True,new_entity_initial_sha256=initial_hash,elapsed_s=time.monotonic()-started,
              parameters=sum(v.numel() for v in model.parameters()),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
              loading_diagnostics=loading,data_receipt_sha256=sha(cache/'receipt.json'),input_sha256=receipt['input_sha256'],
              checkpoint_sha256=sha(out/'best.pt'),implementation_sha256=sha(Path(__file__)),limits=['Native EXBert clean foundation training; no completed KGEditor meta/online comparison yet.'])
    (out/'terminal.json').write_text(json.dumps(term,indent=2));print('LM_BASE_TERMINAL',job['id'],flush=True)

def audit_base(root,cfg,job):
    torch.set_num_threads(2);out=root/'artifacts/lm-base'/job['id'];term=json.loads((out/'terminal.json').read_text());failures=[]
    assert not term['test_read'] and not term['new_snapshot_facts_read'] and not term['meta_train_facts_fitted_in_base']
    for name,digest in term['input_sha256'].items():assert sha(name)==digest
    assert sha(out/'best.pt')==term['checkpoint_sha256']
    source=(root/cfg['source_root']/cfg['native_source']).resolve();sys.path.insert(0,str(source))
    from transformers import BertConfig
    from src.models.modeling_bert import EXBertForMaskedLM
    c=torch.load(str(out/'best.pt'),map_location='cpu');model=EXBertForMaskedLM(BertConfig.from_dict(c['config']));model.load_state_dict(c['state_dict']);model.eval()
    f=np.load(out/'valid-fixture.npz');mrr,s,rank=base_validate(model,f,torch.device('cpu'),batch_size=8)
    error=float(np.abs(s-f['scores']).max());mismatch=int((rank!=f['ranks']).sum())
    if error>cfg['audit_logit_tolerance']:failures.append('CPU/GPU fixture logit tolerance')
    if mismatch:failures.append('CPU/GPU fixture exact ranking mismatch')
    new=model.bert.embeddings.word_embeddings.weight[30522+7505:30522+11258].detach().numpy()
    if hashlib.sha256(new.tobytes()).hexdigest()!=term['new_entity_initial_sha256']:failures.append('New entity rows changed during base fitting')
    records=json.loads((out/'training.json').read_text());chosen=max(records,key=lambda r:r['old_validation_mrr'])
    if chosen['epoch']!=c['epoch'] or chosen['epoch']!=term['best_epoch']:failures.append('Foundation validation selection mismatch')
    result=dict(status='PASS_CLEAN_BASE_CPU_GPU_FIXTURE' if not failures else 'FAIL_RETAINED',failures=failures,queries=len(rank),candidate_score_checks=int(s.size),max_abs_logit_error=error,rank_mismatches=mismatch,
                limits=['CPU/GPU native model forward replay; not independent architecture implementation or optimizer retraining.','Common old foundation only; editor and novel-entity online stages remain required.'])
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert not failures,result
