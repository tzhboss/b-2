#!/usr/bin/env python3
"""Read-only CPU audit of registered KGEditor native exposed evaluation; JSON stdout only."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True

MANIFEST_SHA = "314b14fa671d140b579ba5e728afd6efb36349e34404d2f04a8ef33d2442a58b"
SOURCE_COMMIT = "3035e7f898b0eaa805542a120aba080f04fd5ba4"
MODEL_REVISION = "a103204def5465ac090935c925451eef6692f2cc"
DATA_REVISION = "18a560856826fdcfab5b328829f550ce0f9e5a7a"
SPECS = {
    "E-FB15k237": dict(train=3087,test=3087,stable=7051,entities=14951,relations=237,
                       evaluated=3086,steps=30860,ex_size=64,kb_layer="10,11",
                       article=[.866,.986,.999,.874,.635]),
    "E-WN18RR": dict(train=1491,test=830,stable=5003,entities=40943,relations=11,
                     evaluated=830,steps=14900,ex_size=16,kb_layer="11",
                     article=[.833,.844,.991,.956,.256]),
}
COMMON = dict(lr=.0005,gpus=1,max_seq_length=64,accelerator="ddp",num_workers=4,
              edit_num=2,stable_batch_size=8,max_epochs=20,divergences="kl",
              task_name="edit",seed=0,save_top_k=10)
LABEL = "native exposed evaluation (checkpoint-selection edit_test)"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def number(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def integer(x):
    return isinstance(x,int) and not isinstance(x,bool)

def close(a,b,tol=1e-10):
    return number(a) and number(b) and math.isclose(a,b,rel_tol=tol,abs_tol=tol)

def rows(path):
    result=[]
    with Path(path).open() as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try:
                    value=json.loads(line)
                    if not isinstance(value,dict):raise ValueError("row is not an object")
                    result.append(value)
                except Exception as err:
                    raise ValueError("%s line %d: %s" % (path,n,err))
    return result

def within(path,root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False

class Audit:
    def __init__(self,name):
        self.name=name
        self.checks=0
        self.failures=[]
        self.failure_count=0
        self.missing=[]
        self.evidence={}
    def ck(self,ok,message):
        self.checks+=1
        if not ok:
            self.failure_count+=1
            if len(self.failures)<40:self.failures.append(message)
        return bool(ok)
    def need(self,path):
        if not Path(path).is_file():
            self.missing.append(str(path))
            return False
        return True
    def load(self,path):
        if not self.need(path):return None
        try:return json.loads(Path(path).read_text())
        except Exception as err:
            self.ck(False,"Invalid JSON "+str(path)+": "+str(err))
            return None
    def result(self):
        return dict(run=self.name,
          status="FAIL" if self.failure_count else "INCOMPLETE" if self.missing else "PASS_WITH_LIMITATIONS",
          checks=self.checks,failure_count=self.failure_count,failures=self.failures,
          missing_evidence=self.missing,evidence=self.evidence)

def independent_features(data,model,run,a,spec):
    # No author imports, no model/checkpoint inference. Original base WordPiece only.
    from transformers import BertTokenizer
    vocab=model/"vocab.txt"
    tokenizer=BertTokenizer(vocab_file=str(vocab),do_lower_case=True)
    tokenizer.sanitize_special_tokens() # Match from_pretrained special-token registration; adds no new vocab IDs.
    a.ck(len(tokenizer)==30522,"base WordPiece vocabulary length")
    a.ck((tokenizer.pad_token_id,tokenizer.mask_token_id,tokenizer.cls_token_id,
          tokenizer.sep_token_id,tokenizer.unk_token_id)==(0,103,101,102,100),"base special token IDs")
    a.ck(sha(run/"base_tokenizer/vocab.txt")==sha(vocab),"staged base vocab lineage")
    basecfg=a.load(run/"base_tokenizer/tokenizer_config.json")
    basemap=a.load(run/"base_tokenizer/special_tokens_map.json")
    a.ck(basecfg==dict(tokenizer_class="BertTokenizer",do_lower_case=True,model_max_length=512),
         "base tokenizer configuration")
    a.ck(basemap==dict(cls_token="[CLS]",sep_token="[SEP]",unk_token="[UNK]",
                       pad_token="[PAD]",mask_token="[MASK]"),"base tokenizer special map")
    a.ck(not (run/"base_tokenizer/added_tokens.json").exists(),"base tokenizer must not carry added tokens")
    ep=data/("entity2textlong.txt" if (data/"entity2textlong.txt").exists() else "entity2text.txt")
    names=[];descriptions={}
    for line in ep.read_text().splitlines():
        bits=line.strip().split("\t")
        if len(bits)<2:raise ValueError("Invalid entity text row")
        names.append(bits[0])
        descriptions[bits[0]]=bits[1].replace("\\n"," ").replace("\\","")
    a.ck(len(names)==len(set(names))==spec["entities"],"entity text order/count/uniqueness")
    entity={n:i for i,n in enumerate(descriptions)}
    relnames=[s.strip() for s in (data/"relations.txt").read_text().splitlines()]
    relation={n:i+len(names) for i,n in enumerate(relnames)}
    reltext={}
    for line in (data/"relation2text.txt").read_text().splitlines():
        bits=line.strip().split("\t")
        if len(bits)<2:raise ValueError("Invalid relation description row")
        reltext[bits[0]]=bits[1]
    a.ck(len(relnames)==len(set(relnames))==spec["relations"],"relation token order/count/uniqueness")
    added=json.loads((model/"added_tokens.json").read_text())
    a.ck(all(added.get("[ENTITY_%d]"%i)==30522+i for i in range(len(names))),"HF entity token lineage")
    a.ck(all(added.get("[RELATION_%d]"%i)==30522+len(names)+i for i in range(len(relnames))),
         "HF relation token lineage uses relations.txt, not all description rows")
    a.ck(json.loads((model/"config.json").read_text())["vocab_size"]==30522+len(names)+len(relnames),
         "HF model vocab equals base plus entity/relation tokens")
    def convert(raw,stable,mode):
        retained=[];examples=[];features=[]
        for index,row in enumerate(raw):
            triple=row["triple"] if stable else row["cor"]
            if triple[0] not in entity or triple[2] not in entity or triple[1] not in reltext:
                continue
            target=row["label"]
            head=(triple[0]==target) if stable else (row["ori"][0]==target)
            label=entity[triple[0] if head else triple[2]]
            real=entity[target]
            rel=relation[triple[1]]
            text_a="[MASK]" if head else "[PAD] "
            text_b=reltext[triple[1]]+"[PAD]"
            text_c="[PAD] "+descriptions[triple[2]] if head else "[MASK] "+descriptions[triple[0]]
            en=[rel,entity[triple[2]]] if head else [entity[triple[0]],rel]
            ex=dict(guid=mode+"-0",text_a=text_a,text_b=text_b,text_c=text_c,
                    label=[label],real_label=real,en=en,rel=rel)
            text1=tokenizer.sep_token.join([text_a,text_b]) if head else text_a
            text2=text_c if head else tokenizer.sep_token.join([text_b,text_c])
            encoded=tokenizer(text1,text2,truncation="longest_first",max_length=64,
                              padding="longest",add_special_tokens=True)
            ids=list(encoded["input_ids"])
            pos=0;cnt=0
            for j,token in enumerate(ids):
                if token==tokenizer.pad_token_id:
                    ids[j]=en[cnt]+30522
                    cnt+=1
                if ids[j]==real+30522:pos=j
                if cnt==len(en):break
            a.ck(cnt==len(en),"entity/relation placeholder count %s row %d"%(mode,index))
            a.ck(ids.count(103)==1,"single retained mask %s row %d"%(mode,index))
            a.ck(all(integer(t) and 0<=t<30522+len(names)+len(relnames) for t in ids),
                 "token range %s row %d"%(mode,index))
            # Condition text is base-tokenized in native code; observation exports omit it.
            retained.append(index);examples.append(ex)
            features.append(dict(input_ids=ids,label=real,labels=[label],pos=pos))
        return retained,examples,features
    rawtest=rows(data/"edit_test.jsonl")
    rawstable=rows(data/"stable.jsonl")
    rawtrain=rows(data/"edit_train.jsonl")
    a.ck((len(rawtrain),len(rawtest),len(rawstable))==(spec["train"],spec["test"],spec["stable"]),
         "primary Table1/raw split counts")
    train_ids,train_examples,train_features=convert(rawtrain,False,"train")
    dev_ids,dev_examples,dev_features=convert(rawtest,False,"dev")
    mem_ids,mem_examples,mem_features=convert(rawstable,True,"memory")
    a.ck(len(train_ids)==spec["train"] and len(dev_ids)==spec["test"] and len(mem_ids)==spec["stable"],
         "no unexpected processor filtering")
    for mode,expected in [("train",train_examples),("dev",dev_examples),("memory",mem_examples)]:
        path=run/("data_view/examples_"+mode+".txt")
        if a.need(path):
            a.ck(rows(path)==expected,"independent native textual example lineage "+mode)
    groups=collections.defaultdict(list)
    for i,f in zip(dev_ids,dev_features):
        groups[(tuple(f["input_ids"]),f["label"],tuple(f["labels"]))].append(i)
    a.evidence["token_contract"]={
        "entities":len(names),"relations":len(relnames),"base_vocab":30522,
        "entity_text_source":ep.name,"relation_text_count":len(reltext),
        "independent_reconstruction":"WordPiece input/label/mask/placeholder contract only; no model scores",
        "condition_tokens":"not present in observation JSONL; frozen source binding only"}
    return dev_ids,dev_features,mem_features,groups

def recalculate(export,a,features,spec,workers):
    import numpy as np
    dev_ids,dev,mem,duplicates=features
    permutation=np.random.default_rng(0).permutation(len(mem)).tolist()
    expected_batches=spec["evaluated"]//2
    totals={role:dict(n=0,before=0,after=0,mrr=0,hits={1:0,3:0,5:0},baseline_hits3=0,
                      sources=collections.Counter()) for role in ["edit","locality"]}
    batch=[];batch_no=0;line_count=0
    def check_batch(batch,no):
        a.ck(len(batch)==10,"batch %d must contain edit2 + stable8"%no)
        if len(batch)!=10:return
        lengths=[len(r.get("masked_input_ids",[])) for r in batch]
        a.ck(len(set(lengths))==1 and lengths[0]%8==0 and 0<lengths[0]<=64,
             "batch %d padding length"%no)
        selected=permutation[(no//workers)*8:(no//workers)*8+8]
        expected_fs=[]
        for pos,row in enumerate(batch):
            role="locality" if pos<8 else "edit"
            a.ck(row.get("batch_idx")==no and row.get("position")==pos,
                 "contiguous export batch/position %d:%d"%(no,pos))
            a.ck(row.get("role")==role,"native role order %d:%d"%(no,pos))
            a.ck(row.get("evaluation_label")==LABEL,"exposed evaluation label %d:%d"%(no,pos))
            ids=row.get("source_row_candidates")
            if role=="locality":
                a.ck(row.get("source_split")=="stable","stable source split")
                if pos>=len(selected):
                    a.ck(False,"worker locality permutation exhaustion");continue
                source=selected[pos];f=mem[source]
                a.ck(ids==[source],"worker-local seeded source exposure %d:%d"%(no,pos))
            else:
                a.ck(row.get("source_split")=="edit_test","edit source split")
                index=no*2+pos-8
                if index>=len(dev):
                    a.ck(False,"unexpected edit index");continue
                f=dev[index]
                expected=duplicates[(tuple(f["input_ids"]),f["label"],tuple(f["labels"]))]
                a.ck(ids==expected,"source candidates exactly match duplicate feature class %d:%d"%(no,pos))
            expected_fs.append(f)
            target=row.get("target_entity_index")
            a.ck(target==f["label"],"source-derived target entity %d:%d"%(no,pos))
            input_ids=row.get("masked_input_ids")
            valid_tokens=isinstance(input_ids,list) and all(integer(t) and 0<=t<30522+spec["entities"]+spec["relations"] for t in input_ids)
            a.ck(valid_tokens,"export token types/ranges")
            if valid_tokens:
                a.ck(input_ids.count(103)==1,"exactly one mask in exported input")
                expected_padded=f["input_ids"]+[0]*(lengths[pos]-len(f["input_ids"]))
                a.ck(input_ids==expected_padded,"independent source WordPiece feature/padding %d:%d"%(no,pos))
            valid_metrics=True
            for when in ["before","after"]:
                rank=row.get("rank_"+when);top=row.get("top3_"+when);logit=row.get("target_logit_"+when)
                a.ck(integer(rank) and 1<=rank<=spec["entities"],"rank range")
                a.ck(number(logit),"finite target logit")
                valid_top=isinstance(top,list) and len(top)==3 and len(set(top))==3 and all(integer(t) and 0<=t<spec["entities"] for t in top)
                a.ck(valid_top,"top3 unique entity-index range")
                if integer(rank) and valid_top:
                    a.ck((target in top)==(rank<=3),"rank/top3 membership consistency")
                    if rank<=3:a.ck(top[rank-1]==target,"rank/top3 exact position")
                else:valid_metrics=False
            if valid_metrics:
                t=totals[role];t["n"]+=1;t["before"]+=row["rank_before"];t["after"]+=row["rank_after"]
                t["mrr"]+=1/row["rank_after"]
                t["baseline_hits3"]+=row["rank_before"]<=3
                for k in [1,3,5]:t["hits"][k]+=row["rank_after"]<=k
                if isinstance(ids,list):
                    for i in ids:t["sources"][i]+=1
        if len(expected_fs)==10:
            padded=((max(len(f["input_ids"]) for f in expected_fs)+7)//8)*8
            a.ck(lengths[0]==padded,"exact native dynamic pad-to-multiple8 batch size")
    with export.open() as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():
                a.ck(False,"blank observation line "+str(line_no));continue
            try:row=json.loads(line)
            except Exception as err:
                a.ck(False,"malformed export line %d: %s"%(line_no,err));continue
            if not isinstance(row,dict):
                a.ck(False,"observation row not an object");continue
            batch.append(row);line_count+=1
            if len(batch)==10:
                check_batch(batch,batch_no);batch=[];batch_no+=1
    if batch:check_batch(batch,batch_no)
    a.ck(batch_no==expected_batches and not batch,"expected complete native exposure batches")
    a.ck(line_count==expected_batches*10,"exact export row count")
    output=dict(row_count=line_count)
    for role,t in totals.items():
        if not t["n"]:a.ck(False,"no valid observations for "+role);continue
        b=t["before"]/t["n"];c=t["after"]/t["n"]
        m=dict(count=t["n"],mean_rank_before=b,mean_rank_after=c,mrr_after=t["mrr"]/t["n"],
               unique_source_rows=len(t["sources"]))
        for k in [1,3,5]:m["hits%d_after"%k]=t["hits"][k]/t["n"]
        if role=="edit":m["article_ER_roc"]=abs(c-b)/b
        else:
            m["baseline_hits3_count"]=t["baseline_hits3"]
            m["article_RK_at3"]=t["hits"][3]/t["baseline_hits3"] if t["baseline_hits3"] else None
            m["article_RK_roc"]=abs(c-b)/c
        output[role]=m
        a.evidence[role+"_exposure_multiplicity"]={
            "unique_rows":len(t["sources"]),"exposures":t["n"],
            "max_source_multiplicity":max(t["sources"].values(),default=0),
            "identity_note":"edit duplicate candidates count possible identities, not unique attribution"}
    a.ck(totals["edit"]["n"]==spec["evaluated"],"FB3086/WN830 edit drop_last contract")
    a.ck(totals["locality"]["n"]==expected_batches*8,"8 stable probes per native edit batch")
    a.evidence["recalculated_metrics"]=output
    a.evidence["drop_last"]={"source_test":spec["test"],"evaluated_edits":spec["evaluated"],
                            "omitted_zero_based_rows":dev_ids[spec["evaluated"]:]}
    return output

def audit_training(run,a,job,spec,config):
    complete=a.load(run/"training-complete.json")
    summary=a.load(run/"native-exposed-metrics.json")
    terminal=a.load(run/"terminal.json")
    if terminal is not None:
        a.ck(terminal.get("id")==job["id"],"terminal job identity")
        a.ck(terminal.get("exit_code")==0,"terminal exit code zero")
        a.ck(terminal.get("source_inputs_unchanged") is True,"producer input immutability record")
        if a.need(run/"run.log"):a.ck(sha(run/"run.log")==terminal.get("log_sha256"),"terminal log binding")
    if complete is None or summary is None:return summary
    a.ck(complete.get("validation_exposure")=="native exposed evaluation","training exposure label")
    a.ck(complete.get("global_step")==spec["steps"],"exact 20-epoch optimizer step count")
    a.ck(complete.get("current_epoch") in [19,20],"training complete epoch index consistent with20epochs")
    checkpoint=Path(complete.get("best_model_path",""))
    if not checkpoint.is_absolute():checkpoint=run/checkpoint
    a.ck(within(checkpoint,run/"logger"),"selected checkpoint inside registered run logger")
    a.ck(summary.get("evaluation_label")==LABEL,"summary exposed evaluation label")
    a.ck(Path(summary.get("checkpoint","")).resolve()==checkpoint.resolve(),"summary selected checkpoint identity")
    a.ck(Path(summary.get("export","")).resolve()==(run/"native-exposed-examples.jsonl").resolve(),"summary export identity")
    a.ck(Path(summary.get("source_preflight","")).resolve()==(run/"source-preflight.json").resolve(),"summary preflight identity")
    effective=dict(COMMON,ex_size=spec["ex_size"],kb_layer=spec["kb_layer"],
                   data_type=job["dataset"][2:],ex_model_checkpoint=str(run.parents[1]/"models/KGEditor-HF"/job["dataset"]),
                   model_name_or_path=str(run/"base_tokenizer"),data_dir=str(run/"data_view"),dirpath=str(run/"logger"))
    a.ck(summary.get("config")==effective,"summary exact frozen effective configuration")
    if not a.need(checkpoint):return summary
    a.ck(sha(checkpoint)==summary.get("checkpoint_sha256"),"selected checkpoint SHA256")
    import torch
    ckpt=torch.load(str(checkpoint),map_location="cpu")
    a.ck(isinstance(ckpt,dict),"Lightning checkpoint dictionary")
    hp=ckpt.get("hyper_parameters",{})
    for k,v in effective.items():a.ck(hp.get(k)==v,"checkpoint hyperparameter "+k)
    ep=ckpt.get("epoch");step=ckpt.get("global_step")
    a.ck(integer(ep) and 0<=ep<20,"selected checkpoint epoch0..19")
    a.ck(integer(step) and 0<step<=spec["steps"],"selected checkpoint optimizer step bounds")
    if integer(ep) and integer(step):
        per_epoch=spec["steps"]//20
        a.ck(step in [(ep+1)*per_epoch-1,(ep+1)*per_epoch],
             "selected checkpoint epoch/step link (Lightning save step convention)")
    state=ckpt.get("state_dict",{})
    a.ck(bool(state) and any(k.startswith("ex_model.") for k in state) and any(k.startswith("ex_learner.") for k in state),
         "selected checkpoint contains base model and trained editor")
    a.ck(all(not torch.is_tensor(v) or v.device.type=="cpu" for v in state.values()),"checkpoint metadata loaded CPU only")
    callbacks=ckpt.get("callbacks",{})
    choices=[v for v in callbacks.values() if isinstance(v,dict) and "best_model_score" in v]
    a.ck(len(choices)==1,"one checkpoint selection callback state")
    if len(choices)==1:
        c=choices[0]
        score=c.get("best_model_score")
        score=float(score) if score is not None else None
        a.ck(close(score,complete.get("best_model_score"),1e-7),"checkpoint callback/complete selection score")
        a.ck(Path(c.get("best_model_path","")).resolve()==checkpoint.resolve(),"checkpoint callback selected path")
    a.evidence["training_checkpoint"]={k:complete.get(k) for k in
         ["best_model_path","best_model_score","global_step","current_epoch"]}
    a.evidence["training_checkpoint"].update(checkpoint_epoch=ep,checkpoint_global_step=step,
                                            checkpoint_sha256=summary.get("checkpoint_sha256"))
    del ckpt,state
    # Read-only logged scalar history independently verifies the monitor and all completed epochs.
    event_paths=list((run/"logger").glob("**/events.out.tfevents.*"))
    if not event_paths:
        a.missing.append(str(run/"logger/**/events.out.tfevents.*"));return summary
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    monitors=[];epochs=[]
    for event in event_paths:
        acc=EventAccumulator(str(event),size_guidance={"scalars":0})
        acc.Reload()
        tags=acc.Tags().get("scalars",[])
        if "Eval/hits3" in tags:
            monitors.extend((x.step,x.value) for x in acc.Scalars("Eval/hits3"))
        if "epoch" in tags:epochs.extend(x.value for x in acc.Scalars("epoch"))
    a.ck(bool(monitors),"logged Eval/hits3 selection monitor")
    if monitors:a.ck(close(max(v for s,v in monitors),complete.get("best_model_score"),1e-6),
                       "selected score equals maximum logged validation hits3")
    a.ck(set(range(20)).issubset({int(e) for e in epochs if number(e) and e==int(e)}),
         "logged epochs cover0..19")
    a.evidence["logged_selection"]={"monitor_samples":len(monitors),
         "maximum_logged_hits3":max((v for s,v in monitors),default=None),
         "logged_epoch_indices":sorted({int(e) for e in epochs if number(e) and e==int(e)})}
    return summary

def audit_one(root,run,job):
    a=Audit(job["id"]);dataset=job["dataset"];spec=SPECS[dataset]
    try:
        registered=a.load(run/"registered-job.json")
        a.ck(registered==job,"registered job exactly matches pinned master manifest")
        a.ck(job.get("source_commit")==SOURCE_COMMIT and job.get("model_revision")==MODEL_REVISION
             and job.get("dataset_revision")==DATA_REVISION and job.get("seed")==0,"source/HF revision and seed binding")
        a.ck(run.resolve()==(root/job["run_directory"]).resolve(),"run path binding")
        source=(root/job["copy_repository"]).resolve()
        expected_copy={}
        for name,digest in job["input_sha256"].items():
            path=root/name
            if a.need(path):a.ck(sha(path)==digest,"current registered input SHA256 "+name)
            if within(path,source):
                key="worktree/"+str(path.resolve().relative_to(source))
                expected_copy[key]=digest
        copied=a.load(run/"copied-source-sha256.json")
        a.ck(copied==expected_copy,"copied source manifest exact registered subset; no sidecar widening")
        for name,digest in expected_copy.items():
            if a.need(run/name):a.ck(sha(run/name)==digest,"copied executable source SHA256 "+name)
        expected_command=[str(run/"worktree")+v[len(str(source)):] if v.startswith(str(source)+"/") else v for v in job["command"]]
        expected_cwd=str(run/"worktree"/Path(job["cwd"]).resolve().relative_to(source))
        runtime=a.load(run/"runtime-command.json")
        a.ck(runtime==dict(command=expected_command,cwd=expected_cwd,gpu=job["gpu"]),"copied runtime command binding")
        cfg=a.load(run/"worktree/deltaKG/native/config.json")
        if cfg is None:return a.result()
        a.ck(cfg.get("common_args")==COMMON,"frozen twenty epoch/edit2/stable8/workers4 native config")
        a.ck(cfg.get("source_commit")==SOURCE_COMMIT and cfg.get("model_revision")==MODEL_REVISION
             and cfg.get("dataset_revision")==DATA_REVISION,"config provenance pins")
        ds=cfg["datasets"][dataset]
        a.ck(ds.get("ex_size")==spec["ex_size"] and ds.get("kb_layer")==spec["kb_layer"],
             "dataset extension size/layers")
        a.ck(job.get("completion_contract")==dict(epochs=20,selected_checkpoint=1,exposed_example_export=1,exposed_metrics=1),
             "registered completion contract")
        data=root/"data/KGEditor-HF"/dataset;model=root/"models/KGEditor-HF"/dataset
        pre=a.load(run/"source-preflight.json")
        if pre is not None:
            a.ck(pre.get("dataset")==dataset and pre.get("source_commit")==SOURCE_COMMIT
                 and pre.get("model_revision")==MODEL_REVISION and pre.get("dataset_revision")==DATA_REVISION,
                 "source-preflight provenance")
            a.ck(pre.get("edit_num")==2 and pre.get("drop_last") is True,"preflight native batching")
            a.ck(pre.get("model_file_bytes")== (model/"pytorch_model.bin").stat().st_size,"preflight weight size")
            expected_staged={p.name:job["input_sha256"].get("data/KGEditor-HF/"+dataset+"/"+p.name)
                             for p in data.iterdir() if p.is_file() and
                             (p.suffix==".txt" or p.name in ["stable.jsonl","edit_train.jsonl","edit_test.jsonl"])}
            a.ck(pre.get("input_sha256")==expected_staged,"preflight staged manifest exact registered raw subset")
            for name,digest in pre.get("input_sha256",{}).items():
                a.ck(digest==job["input_sha256"].get("data/KGEditor-HF/"+dataset+"/"+name),
                     "preflight/staged raw input pin "+name)
                if a.need(run/"data_view"/name):a.ck(sha(run/"data_view"/name)==digest,"staged raw input unchanged "+name)
            aliases={"train.jsonl":"edit_train.jsonl","dev.jsonl":"edit_test.jsonl","test.jsonl":"edit_test.jsonl"}
            a.ck(pre.get("split_aliases")==aliases and pre.get("validation_exposure")=="edit_test used for checkpoint selection",
                 "native exposed split aliases")
            for dest,raw in aliases.items():
                if a.need(run/"data_view"/dest):a.ck(sha(run/"data_view"/dest)==sha(data/raw),"staged split source "+dest)
            a.ck(pre.get("filtered_train_row_ids")==list(range(spec["train"]))
                 and pre.get("filtered_validation_row_ids")==list(range(spec["test"])),"preflight row retention identity")
            a.ck(pre.get("eval_dropped_rows")==list(range(spec["evaluated"],spec["test"])),"preflight fixed omitted eval rows")
            a.ck(pre.get("train_dropped_count")==spec["train"]%2,"native shuffled train drop_last count")
        a.evidence["registered_inputs"]={"current_input_count":len(job["input_sha256"]),
               "copied_source_count":len(expected_copy),"model_revision":MODEL_REVISION,"dataset_revision":DATA_REVISION}
        if not (run/"terminal.json").is_file():
            # Active exports/checkpoints can be partial. Never diagnose their transient contents as final failure.
            for name in ["terminal.json","training-complete.json","native-exposed-metrics.json",
                         "native-exposed-examples.jsonl"]:
                a.need(run/name)
            a.evidence["final_artifact_checks"]="deferred until producer terminal.json exists"
            return a.result()
        summary=audit_training(run,a,job,spec,cfg)
        export=run/"native-exposed-examples.jsonl"
        if a.need(export):
            export_digest_before=sha(export)
            features=independent_features(data,model,run,a,spec)
            calculated=recalculate(export,a,features,spec,4)
            if summary is not None:
                a.ck(summary.get("row_count")==calculated["row_count"],"logged summary export row_count")
                for role in ["edit","locality"]:
                    for key,value in calculated.get(role,{}).items():
                        observed=summary.get(role,{}).get(key)
                        a.ck(observed==value if value is None or integer(value) else close(observed,value),
                             "independent summary agreement "+role+"/"+key)
            edit=calculated.get("edit",{});loc=calculated.get("locality",{})
            actual=[edit.get("hits1_after"),edit.get("hits3_after"),edit.get("article_ER_roc"),
                    loc.get("article_RK_at3"),loc.get("article_RK_roc")]
            a.evidence["article_comparison"]={
                "primary":"https://arxiv.org/html/2301.10405v8 Table2 / Eqs3,4",
                "metric_order":["Succ@1","Succ@3","ER_roc","RK@3","RK_roc"],
                "article":spec["article"],"exposed_run":actual,
                "difference":[v-w if number(v) else None for v,w in zip(actual,spec["article"])],
                "claim":"descriptive comparison only; different locality exposure weighting, reused validation and rank tie backend"}
            a.evidence["export_sha256"]=sha(export)
            a.ck(export_digest_before==a.evidence["export_sha256"],"terminal export immutable during audit")
    except Exception as err:
        a.ck(False,"Auditor could not complete section: "+type(err).__name__+": "+str(err))
    return a.result()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--canonical",type=Path)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--runs",nargs="*",help="Registered job IDs; default both. Paths derived from pinned manifest.")
    args=p.parse_args()
    root=args.root.resolve()
    canonical=args.canonical or root/"experiments/KGE-LIT-004/experiment.yaml"
    manifest=args.manifest or root/"configs/experiments/KGE-LIT-004-jobs.json"
    master=Audit("KGE-LIT-004-master");results=[];snapshot=None
    try:
        import yaml
        if master.need(canonical) and master.need(manifest):
            snapshot=canonical.read_bytes()
            state=yaml.safe_load(snapshot)
            digest=sha(manifest)
            master.ck(digest==MANIFEST_SHA,"fixed registered master manifest SHA256")
            master.ck(state.get("id")=="KGE-LIT-004" and state.get("registration_status")=="REGISTERED","canonical registration identity")
            master.ck(state.get("jobs_manifest_sha256")==digest,"current canonical mastermanifest binding")
            master.ck((root/state["jobs_manifest"]).resolve()==manifest.resolve(),"canonical manifest path")
            master.evidence.update(current_canonical_sha256=hashlib.sha256(snapshot).hexdigest(),
                 canonical_status=state.get("status"),master_manifest_sha256=digest,
                 source="exact canonical byte snapshot at audit start; canonical remains mutable runtime record")
            jobs=json.loads(manifest.read_text())
            master.ck(isinstance(jobs,list) and len(jobs)==2,"two registered KGEditor jobs")
            if not master.failure_count:
                selected=[j for j in jobs if not args.runs or j["id"] in args.runs]
                master.ck(len(selected)==(len(args.runs) if args.runs else 2),"requested jobs exactly found")
                for job in selected:
                    results.append(audit_one(root,root/job["run_directory"],job))
                master.evidence["canonical_job_states"]={j.get("id"):j.get("status") for j in state.get("jobs",[])}
                end=canonical.read_bytes()
                master.evidence["canonical_changed_during_audit"]=snapshot!=end
                master.evidence["canonical_end_sha256"]=hashlib.sha256(end).hexdigest()
                active={j.get("id") for j in state.get("active_processes",[])}
                for result in results:
                    if result["run"] in active and result["status"]=="PASS_WITH_LIMITATIONS":
                        result["status"]="INCOMPLETE"
                        result["missing_evidence"].append("canonical snapshot still records active job")
    except Exception as err:master.ck(False,type(err).__name__+": "+str(err))
    mr=master.result()
    failure=mr["failure_count"] or any(r["failure_count"] for r in results)
    incomplete=bool(mr["missing_evidence"]) or not results or any(r["status"]=="INCOMPLETE" for r in results)
    status="FAIL" if failure else "INCOMPLETE" if incomplete else "PASS_WITH_LIMITATIONS"
    report=dict(status=status,master=mr,runs=results,
       verification_scope="CPU hashes/checkpoint metadata, independent token-source contract and exported rank aggregate arithmetic",
       unverified=["No model forward/GPU inference; full candidate logits unavailable, so exported ranks/top3 cannot be proven from all scores.",
          "Finite target logits checked for range/serialization only; score recomputation is outside this auditor.",
          "Source features/targets/masks verified; condition-token exports are unavailable.",
          "20 epochs and checkpoint selection checked from steps, metadata and logged scalar evidence, not external execution observation."],
       protocol_limits=["edit_test reused to select checkpoint: native exposed evaluation, not untouched heldout.",
          "8 locality probes per edit2 batch, repeated worker-local exposures, not a full unique L-Test evaluation.",
          "workers4 replaces32; locality weighting and rank tie backend may differ from article.",
          "ER_roc/RK_roc are mean-rank rate changes, not ROC AUC; no SOTA/novelty claim."],
       read_only=True)
    print(json.dumps(report,indent=2,allow_nan=False))
    return 1 if failure else 2 if incomplete else 0

if __name__=="__main__":
    raise SystemExit(main())
