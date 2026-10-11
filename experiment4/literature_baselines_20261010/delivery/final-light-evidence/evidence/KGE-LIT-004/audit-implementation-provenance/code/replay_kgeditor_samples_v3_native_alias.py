#!/usr/bin/env python3
"""Fixed 16-batch independent KGEditor checkpoint replay. Main explicitly launches GPU work."""
import argparse
import collections
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True

MASTER_SHA="314b14fa671d140b579ba5e728afd6efb36349e34404d2f04a8ef33d2442a58b"
AUDITOR_SHA="ee004305ad0f01d985622833ba89a1fecaa339d752c3f1a4451af6d77b792b37"
LOGIT_ATOL=1e-4
N_BATCHES=16
N_PER_BATCH=10
N_EDIT=2
N_STABLE=8

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def inside(path,parent):
    try:Path(path).resolve().relative_to(Path(parent).resolve());return True
    except ValueError:return False

def read_rows(path):
    with Path(path).open() as f:return [json.loads(s) for s in f if s.strip()]

def tensor_hash(tensor):
    t=tensor.detach().contiguous().cpu()
    h=hashlib.sha256()
    h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()

def state_hash(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(tensor_hash(value).encode())
    return h.hexdigest()

def versions(module):
    return {name:int(value._version) for name,value in module.named_parameters()}

def input_hash(batch):
    return {name:tensor_hash(value) for name,value in batch.items()}

def texts(example,tokenizer):
    require(all(k in example for k in ["text_a","text_b","text_c","label","real_label","en"]),
            "incomplete textual example")
    if example["text_a"]=="[MASK]":
        return tokenizer.sep_token.join([example["text_a"],example["text_b"]]),example["text_c"]
    return example["text_a"],tokenizer.sep_token.join([example["text_b"],example["text_c"]])

def source_feature(example,tokenizer):
    """Independent pair tokenization/placeholder replacement; no author processor import."""
    first,second=texts(example,tokenizer)
    encoded=tokenizer(first,second,truncation="longest_first",max_length=64,
                      padding="longest",add_special_tokens=True)
    ids=list(encoded["input_ids"]);attention=list(encoded["attention_mask"])
    replacements=example["en"];count=0
    for i,token in enumerate(ids):
        if token==tokenizer.pad_token_id:
            require(count<len(replacements),"extra placeholder")
            ids[i]=30522+int(replacements[count]);count+=1
        if count==len(replacements):break
    require(count==2,"expected exactly two native entity/relation placeholders")
    require(ids.count(tokenizer.mask_token_id)==1,"source mask absent/truncated/duplicated")
    require(attention==[1]*len(ids),"unexpected unpadded feature attention")
    # Attention is derived from source tokenization length, never guessed from the export shape.
    require(all(t!=0 for t in ids),"internal PAD0 remains after placeholder replacement; contract not clean")
    return ids,attention

def condition(example,tokenizer,entities):
    """Exact independent processor.py679-700 condition formula, plain base tokenizer."""
    labels=example["label"];target=example["real_label"]
    require(isinstance(labels,list) and len(labels)==1,"condition corruption label must be singleton")
    require(isinstance(labels[0],int) and isinstance(target,int) and
            0<=labels[0]<entities and 0<=target<entities,"condition entity label range")
    first,second=texts(example,tokenizer)
    text="[ENTITY_%d] >> [ENTITY_%d] || %s"%(labels[0],target,first+second)
    encoded=tokenizer(text,truncation=True,max_length=64,padding="max_length",add_special_tokens=True)
    ids=encoded["input_ids"];mask=encoded["attention_mask"]
    require(len(ids)==64 and len(mask)==64,"condition length must be64")
    require(all(0<=t<30522 for t in ids),"condition must use plain30522-token base vocabulary")
    require(all(v in [0,1] for v in mask),"invalid condition attention mask")
    return text,ids,mask

def independent_edit_forward(model,learner,batch,entity_start,entity_end,make_functional):
    """Base forward -> edit CE gradients -> learner delta -> functional edited forward."""
    import torch
    named=list(model.named_parameters())
    with torch.enable_grad():
        before=model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).logits
        positions=(batch["input_ids"]==103).nonzero(as_tuple=True)
        require(len(positions[0])==10 and positions[0].tolist()==list(range(10)),
                "one ordered mask per replay row required")
        score=before[positions[0],positions[1],entity_start:entity_end]
        loss=torch.nn.functional.cross_entropy(score[-2:],batch["targets"][-2:],reduction="none").mean(-1)
        gradients=torch.autograd.grad(loss,tuple(p for n,p in named),allow_unused=False)
    gradient_map={name:gradient for (name,param),gradient in zip(named,gradients)}
    with torch.no_grad():
        delta=learner(batch["cond_input_ids"],batch["cond_attention_mask"],grads=gradient_map)
        require(set(delta)==set(learner.param2conditioner_map),"learner delta key set")
        require(all(value.shape==dict(named)[name].shape for name,value in delta.items()),"learner delta shape")
        functional=make_functional(model).eval()
        after=functional(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],
            params=[param+delta.get(name,0) for name,param in named]).logits
    del gradient_map,gradients,loss,delta,functional
    return before.detach(),after.detach(),positions

def selected_rows(path,indices,total_batches):
    chosen={i:[] for i in indices};seen=collections.Counter()
    with path.open() as f:
        for line_no,line in enumerate(f,1):
            require(bool(line.strip()),"blank export row")
            row=json.loads(line)
            index=row.get("batch_idx")
            require(isinstance(index,int) and 0<=index<total_batches,"export batch index bounds")
            seen[index]+=1
            if index in chosen:chosen[index].append(row)
    require(set(seen)==set(range(total_batches)) and all(v==10 for v in seen.values()),
            "complete source export must contain exactly10 rows per native batch")
    for i,batch in chosen.items():
        require([r.get("position") for r in batch]==list(range(10)),"selected row order")
        require([r.get("role") for r in batch]==["locality"]*8+["edit"]*2,"selected role order")
    return chosen

def prepare_contract(args):
    import numpy as np
    root=args.root.resolve();run=args.run.resolve()
    manifest=root/"configs/experiments/KGE-LIT-004-jobs.json"
    require(sha(manifest)==MASTER_SHA,"registered master manifest hash mismatch")
    jobs=json.loads(manifest.read_text())
    job=next((j for j in jobs if (root/j["run_directory"]).resolve()==run),None)
    require(job is not None,"run must be one of the registered KGE-LIT-004 jobs")
    require(json.loads((run/"registered-job.json").read_text())==job,"registered/master job mismatch")
    require(sha(root/"code/audit_kgeditor_native_v2.py")==AUDITOR_SHA,"source auditor executable pin mismatch")
    audit=json.loads(args.source_audit.read_text())
    require(audit.get("status")=="PASS_WITH_LIMITATIONS","source audit must have terminal PASS_WITH_LIMITATIONS")
    require(audit.get("read_only") is True,"source-audit format")
    require(audit.get("master",{}).get("failure_count")==0 and
            audit["master"].get("evidence",{}).get("master_manifest_sha256")==MASTER_SHA,
            "source audit master binding")
    matches=[r for r in audit.get("runs",[]) if r.get("run")==job["id"]]
    require(len(matches)==1,"exactly one matching audited job")
    reviewed=matches[0]
    require(reviewed.get("status")=="PASS_WITH_LIMITATIONS" and reviewed.get("failure_count")==0
            and not reviewed.get("missing_evidence"),"selected source audit must be complete and passed")
    evidence=reviewed["evidence"]
    require("token_contract" in evidence and "recalculated_metrics" in evidence,
            "source audit token reconstruction/metric evidence absent")
    export=run/"native-exposed-examples.jsonl"
    require(sha(export)==evidence.get("export_sha256"),"export no longer matches source-audit PASS")
    complete=json.loads((run/"training-complete.json").read_text())
    terminal=json.loads((run/"terminal.json").read_text())
    require(terminal.get("exit_code")==0 and terminal.get("source_inputs_unchanged") is True,"terminal success required")
    checkpoint=Path(complete["best_model_path"]).resolve()
    require(inside(checkpoint,run/"logger"),"checkpoint outside registered logger")
    require(sha(checkpoint)==evidence.get("training_checkpoint",{}).get("checkpoint_sha256"),
            "selected checkpoint hash differs from source-audit PASS")
    require(evidence["training_checkpoint"].get("best_model_path")==str(checkpoint),"audited selected checkpoint path")
    copied=json.loads((run/"copied-source-sha256.json").read_text())
    require(len(copied)==14,"registered copied-source scope must remain14")
    for relative,digest in copied.items():
        require(sha(run/relative)==digest,"copied source changed: "+relative)
    for relative,digest in job["input_sha256"].items():
        require(sha(root/relative)==digest,"registered raw/current input changed: "+relative)
    data=root/"data/KGEditor-HF"/job["dataset"]
    staged=json.loads((run/"source-preflight.json").read_text())["input_sha256"]
    for relative,digest in staged.items():require(sha(run/"data_view"/relative)==digest,"staged raw file changed")
    for name in ["dev.jsonl","test.jsonl"]:
        require(sha(run/"data_view"/name)==sha(data/"edit_test.jsonl"),"source exposed split alias changed")
    dev=read_rows(run/"data_view/examples_dev.txt")
    stable=read_rows(run/"data_view/examples_memory.txt")
    expected_edits=3086 if job["dataset"]=="E-FB15k237" else 830
    total_batches=expected_edits//2
    require(len(dev)==(3087 if job["dataset"]=="E-FB15k237" else 830),"textual dev row count")
    require(evidence["recalculated_metrics"]["edit"]["count"]==expected_edits,"audited edit drop_last count")
    # Fixed before seeing scores: NumPy linspace integer truncation, including both endpoints.
    indices=np.linspace(0,total_batches-1,num=16,dtype=int).tolist()
    require(len(set(indices))==16,"must select sixteen unique evenly spaced batches")
    selected=selected_rows(export,indices,total_batches)
    verify_sample_text_lineage(data,dev,stable,selected)
    bound_files={str(args.source_audit.resolve()):sha(args.source_audit),
                 str(export):sha(export),str(checkpoint):sha(checkpoint),
                 str(run/"data_view/examples_dev.txt"):sha(run/"data_view/examples_dev.txt"),
                 str(run/"data_view/examples_memory.txt"):sha(run/"data_view/examples_memory.txt"),
                 str(run/"training-complete.json"):sha(run/"training-complete.json")}
    bound_files.update({str(root/relative):digest for relative,digest in job["input_sha256"].items()})
    bound_files.update({str(run/relative):digest for relative,digest in copied.items()})
    return root,run,job,evidence,checkpoint,copied,dev,stable,indices,selected,bound_files

def verify_sample_text_lineage(data,dev,stable,selected):
    """Bind condition-critical example labels/texts to registered raw HF rows, not just input tokens."""
    ep=data/("entity2textlong.txt" if (data/"entity2textlong.txt").exists() else "entity2text.txt")
    descriptions={}
    for line in ep.read_text().splitlines():
        bits=line.strip().split("\t")
        require(len(bits)>=2,"invalid source entity text")
        descriptions[bits[0]]=bits[1].replace("\\n"," ").replace("\\","")
    entity={name:i for i,name in enumerate(descriptions)}
    relation={name.strip():i+len(entity) for i,name in enumerate((data/"relations.txt").read_text().splitlines())}
    reltext={}
    for line in (data/"relation2text.txt").read_text().splitlines():
        bits=line.strip().split("\t");require(len(bits)>=2,"invalid source relation text")
        reltext[bits[0]]=bits[1]
    rawdev=read_rows(data/"edit_test.jsonl");rawstable=read_rows(data/"stable.jsonl")
    checked=0
    for index,batch in selected.items():
        for pos,row in enumerate(batch):
            is_stable=pos<8
            source=row["source_row_candidates"][0] if is_stable else index*2+pos-8
            raw=rawstable[source] if is_stable else rawdev[source]
            example=stable[source] if is_stable else dev[source]
            triple=raw["triple"] if is_stable else raw["cor"]
            head=triple[0]==raw["label"] if is_stable else raw["ori"][0]==raw["label"]
            rel=relation[triple[1]]
            expected=dict(guid=("memory" if is_stable else "dev")+"-0",
              text_a="[MASK]" if head else "[PAD] ",text_b=reltext[triple[1]]+"[PAD]",
              text_c="[PAD] "+descriptions[triple[2]] if head else "[MASK] "+descriptions[triple[0]],
              label=[entity[triple[0] if head else triple[2]]],real_label=entity[raw["label"]],
              en=[rel,entity[triple[2]]] if head else [entity[triple[0]],rel],rel=rel)
            require(example==expected,"raw source/text/condition-label lineage mismatch batch%d position%d"%(index,pos))
            checked+=1
    require(checked==160,"sample text lineage must bind160source examples")
    return checked

def build_inputs(rows,dev,stable,tokenizer,entities,torch,device):
    inputs=[];attention=[];targets=[];cond_ids=[];cond_attention=[];conditions=[]
    for position,row in enumerate(rows):
        if position<8:
            ids=row["source_row_candidates"]
            require(len(ids)==1,"locality source identity must be unique")
            example=stable[ids[0]]
        else:
            source=row["batch_idx"]*2+position-8
            require(source in row["source_row_candidates"],"native edit row absent from source-equivalence candidates")
            example=dev[source]
        require(example["real_label"]==row["target_entity_index"],"textual target/export mismatch")
        feature,mask=source_feature(example,tokenizer)
        padding=len(row["masked_input_ids"])-len(feature)
        require(padding>=0 and row["masked_input_ids"]==feature+[0]*padding,
                "source-derived feature disagrees with exported masked input")
        inputs.append(row["masked_input_ids"]);attention.append(mask+[0]*padding)
        targets.append(row["target_entity_index"])
        if position>=8:
            text,ids,masks=condition(example,tokenizer,entities)
            conditions.append(dict(position=position,source_row=source,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
              cond_input_ids=ids,cond_attention_mask=masks,feature_length=len(feature)))
            cond_ids.append(ids);cond_attention.append(masks)
    require(len(set(map(len,inputs)))==1,"unequal padded input lengths")
    batch={name:torch.tensor(value,dtype=torch.long,device=device) for name,value in
      dict(input_ids=inputs,attention_mask=attention,targets=targets,
           cond_input_ids=cond_ids,cond_attention_mask=cond_attention).items()}
    return batch,conditions

def isolate(args,run,copied):
    work=args.work_dir.resolve()
    require(not work.exists(),"fresh isolated work-dir required; preserve all previous replay evidence")
    require(not inside(work,run),"replay work cannot be inside native run")
    for name in ["runs","repos","models","data","environment","experiments","configs","code","plan"]:
        require(not inside(work,args.root.resolve()/name),"choose work-dir outside protected baseline "+name)
    require(not inside(args.root.resolve(),work),"work-dir cannot contain baseline root")
    work.mkdir(parents=True,exist_ok=False)
    snapshot=work/"frozen-source"
    hashes={}
    for relative,digest in copied.items():
        if not relative.startswith("worktree/deltaKG/src/"):continue
        suffix=Path(relative).relative_to("worktree/deltaKG")
        target=snapshot/suffix;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(str(run/relative),str(target))
        require(sha(target)==digest,"isolated source copy mismatch")
        hashes[str(target.relative_to(work))]=digest
    require(len(hashes)==9,"nine registered author source modules required")
    (work/"copied-author-source-sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    return work,snapshot,hashes

def instantiate(snapshot,checkpoint,job,device):
    """Only EXBert/OneShotLearner constructors, strict prefix weights, no KGEditor constructor."""
    import torch
    from transformers import BertConfig
    sys.path.insert(0,str(snapshot))
    require(not any(k=="src" or k.startswith("src.") for k in sys.modules),"author src namespace already loaded")
    arch=importlib.import_module("src.models.modeling_bert")
    model_classes=importlib.import_module("src.models.models")
    require(inside(arch.__file__,snapshot) and inside(model_classes.__file__,snapshot),"author import provenance mismatch")
    ckpt=torch.load(str(checkpoint),map_location="cpu")
    hp=ckpt["hyper_parameters"]
    require(hp.get("edit_num")==2 and hp.get("max_seq_length")==64,"checkpoint replay edit/token config")
    require(Path(hp["ex_model_checkpoint"]).name==job["dataset"],"checkpoint base-model dataset")
    config_path=Path(hp["ex_model_checkpoint"])/"config.json"
    config=BertConfig.from_dict(json.loads(config_path.read_text()))
    config.ex_size=int(hp["ex_size"]);config.kb_layer=[int(i) for i in hp["kb_layer"].split(",") if i]
    model=arch.EXBertForMaskedLM(config)
    state=ckpt["state_dict"]
    model_weights={name[len("ex_model."):]:value for name,value in state.items() if name.startswith("ex_model.")}
    learner_weights={name[len("ex_learner."):]:value for name,value in state.items() if name.startswith("ex_learner.")}
    require(bool(model_weights) and bool(learner_weights),"checkpoint prefix groups missing")
    selected={name for name,param in model.named_parameters()
              if ("dense_in_ex" in name.lower() or "dense_out_ex" in name.lower()) and "bias" not in name.lower()}
    require(len(selected)==(4 if job["dataset"]=="E-FB15k237" else 2),"expected editable matrix count")
    # Preserve native constructor's CPU shared storage before Lightning root load_state_dict.
    # Native ex_learner is registered after ex_model; its embedding load overwrites base embedding.
    embedding=model.bert.embeddings.word_embeddings.weight.data
    learner=model_classes.OneShotLearner(model,vocab_dim=config.vocab_size,embedding_dim=config.hidden_size,
       hidden_dim=128,condition_dim=1024,include_set=selected,max_scale=hp["max_scale"],embedding_init=embedding)
    base_key='bert.embeddings.word_embeddings.weight'
    decoder_key='cls.predictions.decoder.weight'
    learner_key='condition.embedding.weight'
    require(model.bert.embeddings.word_embeddings.weight.data_ptr()==learner.condition.embedding.weight.data_ptr(),
            'native CPU embedding storage alias must exist before root restore')
    require(model.bert.embeddings.word_embeddings.weight is model.cls.predictions.decoder.weight,
            'native tied model embedding/decoder parameter identity')
    require(torch.equal(model_weights[base_key],model_weights[decoder_key]),'serialized tied base weights differ')
    holder=torch.nn.Module()
    holder.register_parameter('alpha_kl',torch.nn.Parameter(torch.zeros_like(state['alpha_kl'])))
    holder.register_parameter('alpha_lp',torch.nn.Parameter(torch.zeros_like(state['alpha_lp'])))
    holder.register_buffer('margin_kl',torch.zeros_like(state['margin_kl']))
    holder.register_buffer('margin_lp',torch.zeros_like(state['margin_lp']))
    holder.add_module('ex_model',model);holder.add_module('ex_learner',learner)
    # Same registered child order as KGEditor.__init__: one strict root-prefix load.
    holder.load_state_dict(state,strict=True)
    require(list(holder._modules)==['ex_model','ex_learner'],'native restoration child order')
    require(model.bert.embeddings.word_embeddings.weight.data_ptr()==learner.condition.embedding.weight.data_ptr(),
            'native CPU embedding storage alias must persist through restore')
    effective_model=dict(model_weights)
    effective_model[base_key]=learner_weights[learner_key]
    effective_model[decoder_key]=learner_weights[learner_key]
    alias_evidence=dict(cpu_storage_shared=True,strict_root_load=True,child_order=list(holder._modules),
      base_embedding_checkpoint_sha256=tensor_hash(model_weights[base_key]),
      learner_embedding_checkpoint_sha256=tensor_hash(learner_weights[learner_key]),
      effective_base_embedding_sha256=tensor_hash(model.bert.embeddings.word_embeddings.weight),
      checkpoint_embedding_groups_equal=torch.equal(model_weights[base_key],learner_weights[learner_key]),
      max_checkpoint_embedding_difference=float((model_weights[base_key]-learner_weights[learner_key]).abs().max()),
      effective_model_keys_overwritten_by_native_alias=[base_key,decoder_key],
      basis='Frozen KGEditor constructor shares embedding_init=.weight.data, registers ex_model before ex_learner; Lightning root load_state_dict loads child modules in registration order on CPU.')
    for module,weights in [(model,effective_model),(learner,learner_weights)]:
        actual=module.state_dict()
        require(set(actual)==set(weights),"strict prefix key set mismatch")
        for name,value in weights.items():
            require(torch.equal(actual[name],value),"strict-loaded parameter mismatch: "+name)
    metadata=dict(model_prefix_keys=len(model_weights),learner_prefix_keys=len(learner_weights),
      unused_checkpoint_keys=sorted(name for name in state if not name.startswith(("ex_model.","ex_learner."))),
      extension_matrices=sorted(selected),strict_load=True,native_alias_restore=alias_evidence,
      max_scale=hp["max_scale"],ex_size=config.ex_size,kb_layer=config.kb_layer,
      author_module_paths=[arch.__file__,model_classes.__file__],checkpoint_epoch=ckpt["epoch"],
      checkpoint_global_step=ckpt["global_step"])
    del ckpt,state,model_weights,learner_weights,embedding,effective_model,holder
    model.to(device).eval();learner.to(device).eval()
    require(all(param.dtype==torch.float32 for param in model.parameters()),"replay expects native float32 weights")
    require(all(param.dtype==torch.float32 for param in learner.parameters()),"replay expects float32 learner weights")
    return model,learner,metadata

def replay(args):
    root,run,job,evidence,checkpoint,copied,dev,stable,indices,selected,bound=prepare_contract(args)
    work,snapshot,snapshot_hashes=isolate(args,run,copied)
    output=dict(status="NOT_RUN",dataset=job["dataset"],registered_job=job["id"],
       batch_selection="np.linspace(0,total_eval_batches-1,num=16,dtype=int), fixed before score inspection",
       selected_batches=indices,expected_examples=160,logit_absolute_tolerance=1e-4,
       checkpoint=str(checkpoint),bound_input_sha256=bound,source_audit=str(args.source_audit.resolve()),
       independent=True,protocol="native exposed evaluation; numerical sample replay, not heldout or metric reselection")
    try:
        import torch
        from transformers import BertTokenizer
        from higher.patch import monkeypatch as make_functional
        device=torch.device(args.device)
        require(device.type=="cuda","full replay must be explicitly launched on main-assigned CUDA device")
        require(torch.cuda.is_available(),"main-assigned CUDA runtime unavailable")
        tokenizer=BertTokenizer(vocab_file=str(run/"base_tokenizer/vocab.txt"),do_lower_case=True)
        tokenizer.sanitize_special_tokens()
        require(len(tokenizer)==30522 and tokenizer.mask_token_id==103 and tokenizer.pad_token_id==0,
                "plain base tokenizer contract")
        entities=14951 if job["dataset"]=="E-FB15k237" else 40943
        config=json.loads((root/"models/KGEditor-HF"/job["dataset"]/"config.json").read_text())
        require(config["vocab_size"] in [45710,71476],"HF base architecture vocabulary")
        model,learner,metadata=instantiate(snapshot,checkpoint,job,device)
        output["architecture"]=metadata
        initial_hash={"ex_model":state_hash(model),"ex_learner":state_hash(learner)}
        initial_versions={"ex_model":versions(model),"ex_learner":versions(learner)}
        output["parameter_hash_before"]=initial_hash
        failures=[];records=[];conditions=[];max_error=0;checked=0;input_unchanged=True
        for batch_index in indices:
            rows=selected[batch_index]
            batch,condition_records=build_inputs(rows,dev,stable,tokenizer,entities,torch,device)
            for item in condition_records:item["batch_idx"]=batch_index
            conditions.extend(condition_records)
            before_input=input_hash(batch)
            before,after,mask=independent_edit_forward(model,learner,batch,30522,30522+entities,make_functional)
            require(versions(model)==initial_versions["ex_model"] and versions(learner)==initial_versions["ex_learner"],
                    "original parameter versions changed during replay batch "+str(batch_index))
            after_input=input_hash(batch)
            input_unchanged=input_unchanged and before_input==after_input
            for when,logits in [("before",before),("after",after)]:
                scores=logits[mask[0],mask[1],30522:30522+entities]
                require(bool(torch.isfinite(scores).all()),"nonfinite candidate scores")
                order=torch.sort(scores,dim=1,descending=True)[1]
                inverse=torch.sort(order,dim=1)[1]
                ranks=(inverse[torch.arange(10,device=device),batch["targets"]]+1).cpu().tolist()
                top=order[:,:3].cpu().tolist()
                target_scores=scores[torch.arange(10,device=device),batch["targets"]].cpu().tolist()
                for position,row in enumerate(rows):
                    delta=abs(target_scores[position]-row["target_logit_"+when]);max_error=max(max_error,delta)
                    rank_equal=ranks[position]==row["rank_"+when]
                    top_equal=top[position]==row["top3_"+when]
                    logit_equal=math.isfinite(delta) and delta<=1e-4
                    record=dict(batch_idx=batch_index,position=position,when=when,
                      role=row["role"],source_row_candidates=row["source_row_candidates"],
                      target_entity_index=row["target_entity_index"],
                      exported_rank=row["rank_"+when],replayed_rank=ranks[position],
                      exported_top3=row["top3_"+when],replayed_top3=top[position],
                      exported_target_logit=row["target_logit_"+when],replayed_target_logit=target_scores[position],
                      abs_logit_error=delta,rank_exact=rank_equal,top3_exact=top_equal,logit_within_fixed_atol=logit_equal,
                      input_sha256_before=before_input,input_sha256_after=after_input)
                    records.append(record)
                    if not (rank_equal and top_equal and logit_equal):
                        failures.append(dict(batch_idx=batch_index,position=position,when=when,
                          rank_exact=rank_equal,top3_exact=top_equal,abs_logit_error=delta))
            require(all(param.grad is None for param in model.parameters()) and
                    all(param.grad is None for param in learner.parameters()),"unexpected populated .grad field")
            checked+=10
            del batch,before,after,mask,scores,order,inverse
        final_hash={"ex_model":state_hash(model),"ex_learner":state_hash(learner)}
        final_versions={"ex_model":versions(model),"ex_learner":versions(learner)}
        parameters_unchanged=initial_hash==final_hash and initial_versions==final_versions
        require(parameters_unchanged,"model/editor original weights or versions mutated")
        require(input_unchanged,"replay inputs mutated")
        require(all(sha(path)==digest for path,digest in bound.items()),"bound source evidence changed during replay")
        require(all(sha(work/path)==digest for path,digest in snapshot_hashes.items()),"isolated author source changed")
        require(checked==160 and len(records)==320,"exact16batch/160example/320phase-check contract")
        output.update(status="PASS_SAMPLE_REPLAY_WITH_LIMITATIONS" if not failures else "FAIL",
          checked_examples=checked,before_after_records=len(records),max_absolute_logit_error=max_error,
          mismatch_count=len(failures),mismatches=failures,parameter_hash_after=final_hash,
          parameter_versions_before=initial_versions,parameter_versions_after=final_versions,
          parameters_unchanged=parameters_unchanged,inputs_unchanged=input_unchanged,
          runtime=dict(torch_version=torch.__version__,device=str(device),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
          limits=["160 fixed evenly spaced examples only; no full-score claim for unselected batches.",
           "edit_test used for model selection: native exposed evaluation, not pristine heldout.",
           "Condition tokens reconstructed from frozen textual examples/formula; original condition tensors were not exported.",
           "No optimizer, backward accumulation, training, selection changes or metric/protocol replacement.",
           "Replays native CPU shared-embedding restoration: effective base embedding can differ from serialized ex_model prefix. Not certification of artifact-free checkpoint restoration.",
           "Earlier clone-separated replay FAIL retained; same samples and exact ranking/logit tolerance unchanged."])
        (work/"replayed-observations.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records))
        (work/"reconstructed-edit-conditions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in conditions))
    except Exception as err:
        output.update(status="BLOCKED_OR_FAILED",error=type(err).__name__+": "+str(err),
                      limitation="No PASS: source/condition reconstruction or replay execution could not be established cleanly.")
    (work/"replay-report.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(output,indent=2,allow_nan=False))
    return 0 if output["status"]=="PASS_SAMPLE_REPLAY_WITH_LIMITATIONS" else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--run",type=Path,required=True,help="Registered native run; read only")
    p.add_argument("--source-audit",type=Path,required=True,help="Saved terminal PASS_WITH_LIMITATIONS auditor stdout")
    p.add_argument("--work-dir",type=Path,required=True,help="Fresh isolated directory outside protected baseline subtrees")
    p.add_argument("--device",default="cuda:0",help="Main selects a GPU through CUDA_VISIBLE_DEVICES")
    args=p.parse_args()
    try:return replay(args)
    except Exception as err:
        print(json.dumps(dict(status="BLOCKED_OR_FAILED",error=type(err).__name__+": "+str(err),
          sampling_contract="fixed16linspace batches /160examples /logit abs<=1e-4",
          limitation="No PASS; prerequisite audit/provenance/isolation failed."),indent=2))
        return 1

if __name__=="__main__":
    raise SystemExit(main())
