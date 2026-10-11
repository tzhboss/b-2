"""Native KGEditor meta loss/hypernetwork with clean common-task data.

The native module's dataset setup is bypassed to enforce TRAIN-only proxy edits.
CPU/GPU parameter aliasing is explicit; restored learner embeddings never
overwrite the separately stored scientific model embeddings.
"""
import hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from common_incremental import Data,metrics
from common_training import sha
from lm_common_base import cached_npz,tensor_batch,base_validate,encoder_init,encode_chunk


def descriptions(root,cfg,data):
    directory=data.root/cfg['text_metadata_directory']
    en=dict(x.split('\t',1) for x in (directory/'entity2textlong.txt').read_text().splitlines())
    re=dict(x.split('\t',1) for x in (directory/'relation2text.txt').read_text().splitlines())
    E={v:en[k].replace('\\n',' ').replace('\\','') for k,v in data.entities[1].items()}
    R={v:re[k] for k,v in data.relations[1].items() if v%2==0}
    return E,R,{str(directory/f):sha(directory/f) for f in ['entity2textlong.txt','relation2text.txt']}


def pair_text(h,r,E,R,tokenizer):
    if r%2:return tokenizer.sep_token.join(['[MASK]',R[r-1]+'[PAD]']),'[PAD] '+E[h]
    return '[PAD] ',tokenizer.sep_token.join([R[r]+'[PAD]','[MASK] '+E[h]])


def condition_features(tokenizer,queries,wrong,E,R,length):
    ids=[];mask=[]
    for (h,r,t),w in zip(queries,wrong):
        a,b=pair_text(int(h),int(r),E,R,tokenizer)
        text=f'[ENTITY_{int(w)}] >> [ENTITY_{int(t)}] || '+a+b
        enc=tokenizer(text,truncation=True,max_length=length,padding='max_length',add_special_tokens=True)
        ids.append(enc['input_ids']);mask.append(enc['attention_mask'])
    return np.asarray(ids,dtype=np.uint16),np.asarray(mask,dtype=np.uint8)


def prepare_meta(root,cfg):
    from transformers import BertTokenizer,logging
    logging.set_verbosity_error();data=Data(root,cfg);E,R,bound=descriptions(root,cfg,data)
    foundation=root/cfg['base_data_cache'];old=cached_npz(foundation/'old-train.npz');split=cached_npz(foundation/'old-train-meta-split.npz')
    source=json.loads((foundation/'receipt.json').read_text());assert not source['test_read'] and not source['new_snapshot_facts_read']
    for name,digest in source['output_sha256'].items():assert sha(foundation/name)==digest
    assert np.array_equal(old['queries'],data.triples(0,'train').numpy())
    sets=[set(split[k].tolist()) for k in ['base_ids','meta_train_ids','meta_valid_ids']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    assert set.union(*sets)==set(range(len(old['queries'])))
    tokenizer=BertTokenizer.from_pretrained(str(root/cfg['pretraining_assets']))
    selected=np.r_[split['meta_train_ids'],split['meta_valid_ids']];q=old['queries'][selected]
    rng=np.random.default_rng(cfg['proxy_corruption_seed']);wrong=(q[:,2]+rng.integers(1,7505,size=len(q)))%7505
    cond,mask=condition_features(tokenizer,q,wrong,E,R,cfg['max_sequence_length'])
    out=root/cfg['meta_data_cache'];out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'proxy-conditions.npz',query_indices=selected,wrong=wrong,input_ids=cond,attention_mask=mask)
    bound.update(source['input_sha256']);bound[str(foundation/'receipt.json')]=sha(foundation/'receipt.json')
    receipt=dict(status='PASS_OLD_TRAIN_PROXY_CONDITIONS',test_read=False,new_train_facts_read=False,
                 meta_train_directed=len(split['meta_train_ids']),meta_valid_directed=len(split['meta_valid_ids']),
                 wrong_label_equals_gold=int((wrong==q[:,2]).sum()),input_sha256=bound,output_sha256=sha(out/'proxy-conditions.npz'),
                 condition_tokenizer='Plain author BERT tokenizer; ENTITY strings in conditions are not added special tokens')
    assert receipt['wrong_label_equals_gold']==0
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('KGEDITOR_PROXY_READY',flush=True)


def model_hash(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def construct_editor(root,cfg,job,base_path,device,restore=None):
    source=(root/cfg['source_root']/cfg['native_source']).resolve();sys.path.insert(0,str(source))
    from transformers import BertConfig,BertTokenizer
    from src.models.models import BaseKEModule,OneShotLearner
    from src.models.modeling_bert import EXBertForMaskedLM
    checkpoint=torch.load(str(base_path),map_location='cpu')
    class Bridge(BaseKEModule):
        def __init__(self,options):
            super().__init__();self.save_hyperparameters(options)
        def log(self,name,value,*args,**kwargs):self.last_logs[name]=float(value.detach())
    editor=Bridge(cfg['native_options']);editor.last_logs={}
    editor.ex_model=EXBertForMaskedLM(BertConfig.from_dict(checkpoint['config']));editor.ex_model.load_state_dict(checkpoint['state_dict']);editor.ex_model.eval()
    editor.tokenizer=BertTokenizer.from_pretrained(str(root/cfg['pretraining_assets']));editor.edit_num=cfg['edit_num'];editor.entity_id_st=30522;editor.entity_id_ed=30522+7505
    include={n for n,_ in editor.ex_model.named_parameters() if ('dense_in_ex' in n or 'dense_out_ex' in n) and 'bias' not in n}
    assert len(include)==4
    editor.ex_learner=OneShotLearner(editor.ex_model,vocab_dim=42254,embedding_dim=768,hidden_dim=128,condition_dim=1024,include_set=include,
                                   max_scale=cfg['native_options']['max_scale'],embedding_init=editor.ex_model.bert.embeddings.word_embeddings.weight.data)
    editor.alpha_kl=torch.nn.Parameter(torch.ones(()));editor.alpha_kl.register_hook(lambda g:-g)
    editor.alpha_lp=torch.nn.Parameter(torch.ones(()));editor.alpha_lp.register_hook(lambda g:-g)
    editor.register_buffer('margin_kl',torch.tensor(cfg['native_options']['margin_kl_max']));editor.register_buffer('margin_lp',torch.tensor(cfg['native_options']['margin_lp_max']))
    before_alias=editor.ex_model.bert.embeddings.word_embeddings.weight.data_ptr()==editor.ex_learner.condition.embedding.weight.data_ptr();assert before_alias
    if str(device)=='cpu':
        # Reconstruct the disjoint storage produced by the original .cuda()
        # traversal before independently loading a meta-trained learner.
        editor.ex_learner.condition.embedding.weight=torch.nn.Parameter(editor.ex_learner.condition.embedding.weight.detach().clone())
    editor=editor.to(device)
    after_alias=editor.ex_model.bert.embeddings.word_embeddings.weight.data_ptr()==editor.ex_learner.condition.embedding.weight.data_ptr();assert not after_alias
    if restore is not None:
        editor.ex_learner.load_state_dict(restore['learner_state']);editor.alpha_kl.data.copy_(restore['alpha_kl']);editor.alpha_lp.data.copy_(restore['alpha_lp'])
    assert torch.equal(editor.ex_model.bert.embeddings.word_embeddings.weight.detach().cpu(),checkpoint['state_dict']['bert.embeddings.word_embeddings.weight'])
    editor.alias_receipt=dict(cpu_constructor_shared_storage=before_alias,operational_storage_disjoint=not after_alias,
                              restored_model_embedding_equals_clean_base=True,cpu_restore_rule='Clone conditioner embedding to reproduce native GPU disjoint storage before loading learner')
    return editor,include


def meta_tables(root,cfg):
    source=json.loads((root/cfg['base_data_cache']/'receipt.json').read_text())
    for name,digest in source['output_sha256'].items():assert sha(root/cfg['base_data_cache']/name)==digest
    old=cached_npz(root/cfg['base_data_cache']/'old-train.npz');split=cached_npz(root/cfg['base_data_cache']/'old-train-meta-split.npz')
    path=root/cfg['meta_data_cache'];r=json.loads((path/'receipt.json').read_text());assert r['status']=='PASS_OLD_TRAIN_PROXY_CONDITIONS'
    assert sha(path/'proxy-conditions.npz')==r['output_sha256'];cond=cached_npz(path/'proxy-conditions.npz')
    lookup={int(index):j for j,index in enumerate(cond['query_indices'])}
    return old,split,cond,lookup,r


def batch_from_tables(table,edit_ids,stable_ids,conditions,device):
    ids=np.r_[stable_ids,edit_ids];b=tensor_batch(table,ids,device);b['label']=torch.as_tensor(table['queries'][ids,2],dtype=torch.long,device=device)
    cond,mask=conditions
    b['cond_input_ids']=torch.as_tensor(np.concatenate([np.zeros((len(stable_ids),cond.shape[1]),dtype=np.int64),cond]),dtype=torch.long,device=device)
    b['cond_attention_mask']=torch.as_tensor(np.concatenate([np.zeros((len(stable_ids),mask.shape[1]),dtype=np.int64),mask]),dtype=torch.long,device=device)
    return b


def masked_ranks(editor,batch):
    with torch.no_grad():orig,logits,params=editor.forward(batch)
    positions=(batch['input_ids']==103).nonzero(as_tuple=True);value=logits[positions[0],positions[1]]
    gold=value[torch.arange(len(value),device=value.device),batch['label']]
    return (1+(value>gold[:,None]).sum(1)).cpu().numpy(),value.detach().cpu().numpy()


def train_meta(root,cfg,job,preflight=False,cpu_smoke=False):
    torch.set_num_threads(2);torch.manual_seed(job['seed']);np.random.seed(job['seed']);random.seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();device='cpu' if cpu_smoke else 'cuda:0';base=root/cfg['base_directory']/f"lm-base-s{job['seed']}"/'best.pt'
    if cpu_smoke:base=root/cfg['smoke_base_path']
    else:assert json.loads(base.with_name('audit.json').read_text())['status']=='PASS_CLEAN_BASE_CPU_GPU_FIXTURE'
    editor,include=construct_editor(root,cfg,job,base,device);initial_model_hash=model_hash(editor.ex_model)
    old,split,cond,lookup,receipt=meta_tables(root,cfg);rng=np.random.default_rng(job['seed']+101)
    stage='lm-editor-smoke' if cpu_smoke else ('lm-editor-preflight' if preflight else 'lm-editor-meta')
    out=root/'artifacts'/stage/job['id'];out.mkdir(parents=True,exist_ok=False)
    opts,schedules=editor.configure_optimizers();opt=opts[0];scheduler=schedules[0]['scheduler'];records=[];best=-1.;best_epoch=0
    gradient_checks=0
    epochs=1 if preflight or cpu_smoke else cfg['meta_epochs']
    for epoch in range(1,epochs+1):
        order=rng.permutation(split['meta_train_ids']);stables=rng.permutation(split['base_ids']);losses=[];editor.train()
        for off in range(0,len(order)-cfg['edit_num']+1,cfg['edit_num']):
            assert time.monotonic()-started<cfg['meta_budget_seconds'],'Registered meta budget exceeded'
            edits=order[off:off+cfg['edit_num']];sid=(off//cfg['edit_num'])*cfg['stable_batch_size'];stable=stables[sid:sid+cfg['stable_batch_size']];assert len(stable)==cfg['stable_batch_size']
            ci=np.asarray([lookup[int(i)] for i in edits]);batch=batch_from_tables(old,edits,stable,(cond['input_ids'][ci],cond['attention_mask'][ci]),device)
            editor.zero_grad(set_to_none=True);loss=editor.training_step(batch)['loss'];assert torch.isfinite(loss);loss.backward()
            gradients=[p.grad for p in editor.ex_learner.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients),'Nonfinite native learner gradient'
            gradient_checks+=len(gradients);opt.step();editor.on_before_zero_grad(opt);scheduler.step();losses.append(float(loss.detach()))
            if (off//cfg['edit_num'])%cfg['meta_progress_every_steps']==0:print('KGEDITOR_META_STEP',job['id'],epoch,off,len(order),round(time.monotonic()-started,2),flush=True)
            if preflight or cpu_smoke:break
        editor.eval();val=[];valid_ids=split['meta_valid_ids'];stable=rng.permutation(split['base_ids'])
        for off in range(0,len(valid_ids)-cfg['edit_num']+1,cfg['edit_num']):
            edits=valid_ids[off:off+cfg['edit_num']];ci=np.asarray([lookup[int(i)] for i in edits]);sid=(off//cfg['edit_num'])*cfg['stable_batch_size']
            batch=batch_from_tables(old,edits,stable[sid:sid+cfg['stable_batch_size']],(cond['input_ids'][ci],cond['attention_mask'][ci]),device)
            vr,s=masked_ranks(editor,batch);val.append(vr)
            if preflight or cpu_smoke:break
        ar=np.asarray(val);criterion=.5*((1/ar[:,-cfg['edit_num']:]).mean()+(1/ar[:,:-cfg['edit_num']]).mean())
        record=dict(epoch=epoch,loss=float(np.mean(losses)),selection_score=float(criterion),proxy_edit_mrr=float((1/ar[:,-cfg['edit_num']:]).mean()),stable_mrr=float((1/ar[:,:-cfg['edit_num']]).mean()),elapsed_s=time.monotonic()-started)
        records.append(record)
        if criterion>best:
            best=float(criterion);best_epoch=epoch
            torch.save(dict(learner_state={n:v.detach().cpu() for n,v in editor.ex_learner.state_dict().items()},alpha_kl=editor.alpha_kl.detach().cpu(),alpha_lp=editor.alpha_lp.detach().cpu(),epoch=epoch,seed=job['seed'],base_sha256=sha(base)),out/'best.pt')
        (out/'training.json').write_text(json.dumps(records,indent=2));print('KGEDITOR_META_EPOCH',job['id'],epoch,criterion,flush=True)
    assert model_hash(editor.ex_model)==initial_model_hash,'Meta learner training changed scientific model parameters'
    restore=torch.load(str(out/'best.pt'),map_location='cpu');restored,_=construct_editor(root,cfg,job,base,device,restore)
    rr,rs=masked_ranks(restored,batch);original,_=construct_editor(root,cfg,job,base,'cpu',restore)
    cb={k:v.detach().cpu() for k,v in batch.items()};cr,cs=masked_ranks(original,cb)
    error=float(np.abs(rs-cs).max());assert error<=cfg['audit_logit_tolerance'] and np.array_equal(rr,cr),'Meta CPU/GPU restore replay'
    np.savez_compressed(out/'restore-fixture.npz',**{k:v.detach().cpu().numpy() for k,v in batch.items()},scores=rs,ranks=rr)
    terminal=dict(status='PASS_NATIVE_META_AND_RESTORE_ONLY',id=job['id'],seed=job['seed'],device=device,cpu_smoke=cpu_smoke,preflight=preflight,
                  restore_scope='CPU-to-CPU' if cpu_smoke else 'GPU-to-CPU',test_read=False,new_train_facts_read=False,
                  best_epoch=best_epoch,best_validation_score=best,epochs_executed=epochs,elapsed_s=time.monotonic()-started,
                  base_sha256=sha(base),meta_checkpoint_sha256=sha(out/'best.pt'),base_parameters_unchanged=True,finite_gradient_tensor_checks=gradient_checks,
                  alias_receipt=editor.alias_receipt,restore_max_logit_error=error,restore_rank_mismatches=int((rr!=cr).sum()),
                  meta_input_receipt_sha256=sha(root/cfg['meta_data_cache']/'receipt.json'),native_options=cfg['native_options'],
                  limits=['Common old-TRAIN proxy edits, not original author edit split.','Native BaseKEModule forward/loss/optimizer retained; data/selection/trainer wrapper adapted.','Meta result only; complete novel online and common TEST remain required.'])
    (out/'terminal.json').write_text(json.dumps(terminal,indent=2));print('KGEDITOR_META_TERMINAL',job['id'],flush=True)
    return terminal
