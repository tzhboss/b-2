#!/usr/bin/env python3
"""Independent read-only CPU audit for registered LKGE clean FACT seed55.
JSON stdout only: exit0 PASS, exit1 FAIL, exit2 INCOMPLETE. No native imports,
training, CUDA APIs, filesystem writes or environment changes.
"""
import argparse,ast,difflib,hashlib,json,re,subprocess,sys,time,types,warnings
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
MASTER_SHA="6e0db48e2b684f575694f893b488bf4493f728ae021b7a0fbd1435f77f9d79f9"
PROTOCOL_SHA="c4084e4f00292869be8b3dbe087b920b1f7fe9dde63d04ef095caa753eeacbf8"
COMMIT="5a74daefb72cf9718f604b0f29d277dbc638a20d"
CONTRACT="author double-argsort; historical query-snapshot candidates/filter"
KEYS=["mr","mrr","hits1","hits3","hits5","hits10"]
EXPECTED={"snapshots":5,"best_checkpoints":5,"test_rank_files":19,"best_selection_sidecars":5}
REPLAY_METHODS=["embedding","predict","norm_ent","norm_rel"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def check(r,name,ok,detail=None):
    r["checks"]+=1
    if not ok:r["failures"].append({"check":name,"detail":detail})

def missing(r,name):
    if name not in r["missing"]:r["missing"].append(name)

def git(repo,*args):
    return subprocess.check_output(["git",*args],cwd=repo,text=True)

def rebuild(data,r):
    ent={};rel={};filters={};snaps=[];hashes={};seen=set()
    for i in range(5):
        rows={}
        for split in ["train","valid","test"]:
            path=data/str(i)/(split+".txt");hashes[str(path.relative_to(data))]=sha(path)
            rows[split]=[tuple(x.split()) for x in path.read_text().splitlines() if x.strip()]
            check(r,f"triple_columns_{i}_{split}",all(len(x)==3 for x in rows[split]))
            for h,p,t in rows[split]:
                for e in [h,t]:
                    if e not in ent:ent[e]=len(ent)
                if p not in rel:q=len(rel);rel[p]=q;rel[p+"_inv"]=q+1
        sets=[set(rows[k]) for k in ["train","valid","test"]]
        check(r,f"split_disjoint_{i}",not any(sets[a]&sets[b] for a in range(3) for b in range(a)))
        union=set.union(*sets);check(r,f"snapshot_new_triples_{i}",not seen&union);seen|=union
        ids={}
        for split,rs in rows.items():
            facts=[(ent[h],rel[p],ent[t]) for h,p,t in rs]
            if split!="train":facts.sort(key=lambda x:x[1])
            ids[split]=facts
            for h,p,t in facts:
                filters.setdefault((h,p),set()).add(t);filters.setdefault((t,p^1),set()).add(h)
        frozen={k:sorted(v) for k,v in filters.items()}
        query=np.asarray(ids["test"]+[(t,p+1,h) for h,p,t in ids["test"]],dtype=np.int64)
        counts=np.asarray([len(frozen[int(h),int(p)]) for h,p,t in query],dtype=np.int64)
        check(r,f"filter_universe_{i}",all(max(v)<len(ent) for v in frozen.values()))
        check(r,f"test_targets_in_filter_{i}",all(int(t) in filters[int(h),int(p)] for h,p,t in query))
        snaps.append({"num_ent":len(ent),"num_rel":len(rel),"queries":query,"filtered_count":counts,"filters":frozen,
                      "test_count":len(ids["test"]),"valid_count":len(ids["valid"]),"train_count":len(ids["train"]),
                      "filter_sha256":digest([[h,p,ts] for (h,p),ts in sorted(frozen.items())])})
    r["rebuilt_data"]={"raw_sha256":hashes,"id_mapping_sha256":digest([ent,rel]),
       "snapshots":[{k:v for k,v in s.items() if k not in ["queries","filtered_count","filters"]}|
                    {"query_order_sha256":hashlib.sha256(s["queries"].tobytes()).hexdigest(),
                     "filter_count_sha256":hashlib.sha256(s["filtered_count"].tobytes()).hexdigest()} for s in snaps]}
    return snaps

def metrics(rank):
    x=rank.astype(np.float64)
    return {"count":len(x),"mr":float(x.mean()),"mrr":float(np.mean(1/x)),
            **{f"hits{k}":float(np.mean(x<=k)) for k in [1,3,5,10]}}

def native_metrics(rank):
    x=torch.as_tensor(rank).float()
    m={"count":1.0,"mr":round(sum(float(x[i:i+100].sum()) for i in range(0,len(x),100))/len(x),4),
       "mrr":round(sum(float((1/x[i:i+100]).sum()) for i in range(0,len(x),100))/len(x),4)}
    m.update({f"hits{k}":round(float((x<=k).sum())/len(x),4) for k in [1,3,5,10]})
    return m

def overall(out,snaps,i,phase,raw=False):
    w=[s["test_count"] for s in snaps[:i+1]]
    return {"whole_"+k:(sum(out[i,j,phase]["raw" if raw else "rounded"][k]*w[j] for j in range(i+1))/sum(w)
                         if raw else round(sum(out[i,j,phase]["rounded"][k]*w[j] for j in range(i+1))/sum(w),3))
            for k in ["mrr","hits1","hits3","hits10"]}

def bind(root,run,r):
    master=root/"configs/experiments/KGE-LIT-006-jobs.json"
    protocol=root/"configs/protocols/KGE-LIT-006-clean-lkge.json"
    canonical=root/"experiments/KGE-LIT-006/experiment.yaml"
    for p,h in [(master,MASTER_SHA),(protocol,PROTOCOL_SHA)]:
        check(r,"frozen_sha_"+p.name,sha(p)==h)
    jobs=json.loads(master.read_text())
    check(r,"single_master_job",len(jobs)==1)
    job=next(j for j in jobs if j["id"]=="lkge-clean-fact-s55")
    doc=json.loads(protocol.read_text());can=yaml.safe_load(canonical.read_text())
    check(r,"canonical_master_binding",can["jobs_manifest"]==str(master.relative_to(root)) and can["jobs_manifest_sha256"]==MASTER_SHA)
    check(r,"canonical_protocol_binding",can["protocol"]==str(protocol.relative_to(root)) and can["protocol_sha256"]==PROTOCOL_SHA)
    check(r,"canonical_identity",can["id"]=="KGE-LIT-006" and can["registration_status"]=="REGISTERED")
    check(r,"protocol_identity",doc["id"]=="KGE-LIT-006" and doc["source_commit"]==COMMIT and doc["dataset"]=="FACT" and doc["seed"]==55)
    check(r,"protocol_full_budget",doc["no_parameter_grid"] is True and doc["no_shortened_training"] is True)
    check(r,"completion_contract",job["completion_contract"]==doc["completion_contract"]==EXPECTED)
    check(r,"job_identity",job["source_commit"]==COMMIT and job["method"]=="LKGE_clean_split" and job["dataset"]=="FACT" and job["seed"]==55 and job["gpu"]==6)
    check(r,"registered_equals_master",json.loads((run/"registered-job.json").read_text())==job)
    source="repos/LKGE_clean_fact/source";work=run/"worktree";repo=root/"repos/LKGE"
    check(r,"isolated_copy_repository",job["copy_repository"]==source)
    copies={str(Path("worktree")/Path(k).relative_to(source)):v for k,v in job["input_sha256"].items() if k.startswith(source+"/")}
    check(r,"copied_inventory_exact",json.loads((run/"copied-source-sha256.json").read_text())==copies)
    actual={str(p.relative_to(run)) for p in work.rglob("*") if p.is_file()}
    check(r,"actual_worktree_inventory_exact",actual==set(copies),{"extra":sorted(actual-set(copies)),"missing":sorted(set(copies)-actual)})
    check(r,"source_inventory_no_generated_metadata",len(copies)==37 and all(k.endswith(".py") or re.fullmatch(r"worktree/data/FACT/[0-4]/(train|valid|test)\.txt",k) for k in copies))
    changed=[]
    for k,h in job["input_sha256"].items():
        original=root/k
        clone=work/Path(k).relative_to(source) if k.startswith(source+"/") else original
        current=sha(original) if original.is_file() else None
        frozen=sha(clone) if clone.is_file() else None
        check(r,"current_input_"+k,current==h,{"expected":h,"actual":current} if current!=h else None)
        check(r,"executed_input_"+k,frozen==h,{"expected":h,"actual":frozen} if frozen!=h else None)
        if current!=h or frozen!=h:changed.append(k)
    check(r,"input_count",len(job["input_sha256"])==71)
    prep=root/"repos/LKGE_clean_fact/preparation"
    check(r,"protocol_preparation_job",sha(prep/"job-draft.json")==doc["preparation_job_sha256"])
    check(r,"protocol_preparation_manifest",sha(prep/"source-input-sha256.json")==doc["source_manifest_sha256"])
    check(r,"manifest_entries_exact",json.loads((prep/"source-input-sha256.json").read_text())==job["input_sha256"])
    check(r,"upstream_commit",git(repo,"rev-parse","HEAD").strip()==COMMIT)
    runtime=json.loads((run/"runtime-command.json").read_text())
    expected=list(job["command"]);expected[expected.index("-data_path")+1]=str(work/"data")+"/"
    check(r,"runtime_exact_frozen_remap",runtime["command"]==expected)
    check(r,"runtime_cwd_gpu",Path(runtime["cwd"]).resolve()==work.resolve() and runtime["gpu"]==6)
    receipt=run/"terminal.json"
    if receipt.exists():
        term=json.loads(receipt.read_text())
        r["producer_receipt"]={"sha256":sha(receipt),"receipt":term}
        check(r,"producer_exit0",term.get("exit_code")==0)
        check(r,"producer_integrity",term.get("status")=="terminal_unreviewed" and term.get("source_inputs_unchanged") is True,term)
        check(r,"producer_log_binding",term.get("log_sha256")==sha(run/"run.log"))
    else:missing(r,"terminal.json")
    r["binding"]={"master_sha256":sha(master),"protocol_sha256":sha(protocol),"canonical_read_status":can.get("status"),
       "registered_sha256":sha(run/"registered-job.json"),"runtime_sha256":sha(run/"runtime-command.json"),
       "copied_inventory_sha256":sha(run/"copied-source-sha256.json"),"input_entries":71,"copied_entries":37,
       "changed_inputs":changed,"declared_environment":job["environment"],
       "environment_limit":"Declared environment and binary dependency inputs are bound; effective complete process environment was not captured in runtime-command receipt."}
    return job,work,repo,receipt.exists()

class StripObservation(ast.NodeTransformer):
    def visit_Assign(self,n):
        if any(isinstance(t,ast.Name) and t.id=="export_rows" for t in n.targets):return None
        return self.generic_visit(n)
    def visit_If(self,n):
        text=ast.unparse(n.test)
        if "export_rows" in text or "getattr(self.args, 'rank_export_dir'" in text:return None
        return self.generic_visit(n)
def strip(text,path,restore_valid=True):
    tree=ast.parse(text)
    if path=="src/model/model_process.py":
        tree=StripObservation().visit(tree)
        tree.body=[x for x in tree.body if not(isinstance(x,ast.FunctionDef) and x.name=="_save_observed_query_ranks")]
    elif path=="main.py":tree=StripObservation().visit(tree)
    elif path=="src/parse_args.py":
        tree.body=[x for x in tree.body if not(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and any(isinstance(a,ast.Constant) and a.value=="-rank_export_dir" for a in x.value.args))]
    elif path=="src/data_load/KnowledgeGraph.py" and restore_valid:
        class RestoreValid(ast.NodeTransformer):
            def visit_Assign(self,n):
                if any(isinstance(t,ast.Name) and t.id=="valid_facts" for t in n.targets):
                    for v in ast.walk(n.value):
                        if isinstance(v,ast.Constant) and v.value=="valid.txt":v.value="test.txt"
                return self.generic_visit(n)
        tree=RestoreValid().visit(tree)
    return ast.dump(tree,include_attributes=False)
def classmethods(path,name):
    tree=ast.parse(path.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==name)
    return cls,{m.name:m for m in cls.body if isinstance(m,ast.FunctionDef)}
def methodtext(path,name,cls=None):
    text=path.read_text();tree=ast.parse(text)
    nodes=classmethods(path,cls)[1].values() if cls else ast.walk(tree)
    return ast.get_source_segment(text,next(x for x in nodes if isinstance(x,ast.FunctionDef) and x.name==name))

def source_audit(work,repo,r):
    changed=[];diff=[]
    for p in sorted(work.rglob("*.py")):
        f=str(p.relative_to(work));original=git(repo,"show",COMMIT+":"+f);new=p.read_text()
        if original!=new:changed.append(f)
        check(r,"scoped_upstream_AST_"+f,ast.dump(ast.parse(original),include_attributes=False)==strip(new,f))
        diff.extend(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile="a/"+f,tofile="b/"+f))
    kg=work/"src/data_load/KnowledgeGraph.py"
    loader=work/"src/data_load/data_loader.py";trainer=work/"src/train.py";main=work/"main.py"
    assignments=[n for n in ast.walk(ast.parse(kg.read_text())) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="valid_facts" for t in n.targets)]
    check(r,"valid_facts_load_only_valid_txt",len(assignments)==1 and "'valid.txt'" in ast.unparse(assignments[0].value) and "'test.txt'" not in ast.unparse(assignments[0].value))
    tr=methodtext(trainer,"run_epoch","Trainer")
    check(r,"validation_stage_true_before_evaluate",tr.index("self.args.valid = True")<tr.index("self.valid_processor.process_epoch(self.model)")<tr.index("self.args.valid = False"))
    loadertext=loader.read_text()
    check(r,"validation_loader_uses_valid_facts","self.kg.snapshots[ss_id].valid:" in loadertext and "self.valid[self.args.snapshot][idx]" in loadertext)
    train=methodtext(main,"train","experiment")
    check(r,"strict_best_validation_selection","self.best_valid < valid_res[self.args.valid_metrics]" in train and "self.best_valid = valid_res[self.args.valid_metrics]" in train)
    check(r,"native_earlystop_preserved","max(0, self.stop_epoch-5)" in train and "self.stop_epoch += 1" in train and "self.stop_epoch >= self.args.patience" in train)
    cont=methodtext(main,"continual_learning","experiment")
    check(r,"selected_checkpoint_load_before_posttrain",cont.index("self.load_checkpoint(best_checkpoint)")<cont.index("self.model.snapshot_post_processing()")<cont.index("for test_ss_id in range(ss_id+1)"))
    base,methods=classmethods(work/"src/model/BaseModel.py","BaseModel")
    for cname,parent in [("LKGE","BaseModel"),("TransE","LKGE")]:
        cls,mm=classmethods(work/"src/model/LKGE.py",cname)
        check(r,cname+"_inheritance",[ast.unparse(x) for x in cls.bases]==[parent])
        sensitive=set(REPLAY_METHODS+["pre_snapshot","snapshot_post_processing","epoch_post_processing","__getattribute__","__getattr__"])
        check(r,cname+"_no_scorer_or_postprocess_override",not(set(mm)&sensitive),sorted(set(mm)&sensitive))
        aliases=[ast.unparse(n) for n in ast.walk(cls) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr in sensitive for t in n.targets)]
        check(r,cname+"_no_instance_scorer_override",not aliases,aliases)
    emb=methodtext(work/"src/model/BaseModel.py","embedding","BaseModel")
    check(r,"BaseModel_embedding_raw_weights","return self.ent_embeddings.weight, self.rel_embeddings.weight" in emb)
    for name in ["pre_snapshot","snapshot_post_processing","epoch_post_processing"]:
        body=[x for x in methods[name].body if not(isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
        check(r,"BaseModel_"+name+"_noop",len(body)==1 and isinstance(body[0],ast.Pass))
    r["source_proof"]={"commit":COMMIT,"AST_files":22,"changed_text_files":changed,
      "complete_upstream_diff_sha256":hashlib.sha256("".join(diff).encode()).hexdigest(),
      "allowed_changes":"valid_facts test.txt -> valid.txt; optional rank/best sidecar observations only.",
      "scorer_method_sha256":{n:hashlib.sha256(methodtext(work/"src/model/BaseModel.py",n,"BaseModel").encode()).hexdigest() for n in REPLAY_METHODS},
      "replay_basis":"Actual TransE -> LKGE -> BaseModel inheritance proved without embedding/predict/norm/postprocess overrides. BaseModel raw weights score directly; postprocessing is a no-op. GCN and old_data/old_weight are used for losses/transitions, not this scorer."}
    return git(repo,"show",COMMIT+":src/model/model_process.py"),(work/"src/model/model_process.py").read_text()
def defaults_audit(job,work,run,log,r):
    defaults={}
    for n in ast.walk(ast.parse((work/"src/parse_args.py").read_text())):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="add_argument":
            kw={x.arg:x.value for x in n.keywords};dest=ast.literal_eval(kw["dest"]) if "dest" in kw else ast.literal_eval(n.args[0]).lstrip("-")
            defaults[dest]=ast.literal_eval(kw["default"]) if "default" in kw else None
    expected=defaults.copy()
    cmd=json.loads((run/"runtime-command.json").read_text())["command"]
    for i,x in enumerate(cmd):
        if x.startswith("-") and x.lstrip("-") in defaults:expected[x.lstrip("-")]=cmd[i+1]
    configtree=ast.parse((work/"src/config.py").read_text())
    configfn=next(x for x in configtree.body if isinstance(x,ast.FunctionDef) and x.name=="config")
    env={};exec(compile(ast.Module(body=[configfn],type_ignores=[]),"<pure-config>","exec"),env)
    fixture=types.SimpleNamespace(**expected);env["config"](fixture);expected=vars(fixture)
    names={};first=next((x for x in log.splitlines() if "Namespace(" in x),"")
    for k in defaults:
        m=re.search(r"(?:Namespace\(|, )"+re.escape(k)+r"=('[^']*'|\"[^\"]*\"|True|False|None|[-+.\deE]+)(?=,|\))",first)
        if m:names[k]=ast.literal_eval(m[1])
    for k,v in expected.items():
        if k in ["save_path","data_path","log_path"]:continue
        check(r,"effective_logged_"+k,names.get(k)==v,{"expected":v,"logged":names.get(k)} if names.get(k)!=v else None)
    ck=run/"checkpoint/FACT-TransE-LKGE-Margin-WO_transfer"
    check(r,"effective_checkpoint_path",names.get("save_path")==str(ck))
    check(r,"effective_data_path",names.get("data_path")==str(work/"data/FACT")+"/")
    native_log=list((run/"logs").rglob("*.log"))
    check(r,"effective_log_path",len(native_log)==1 and names.get("log_path")==str(native_log[0])[:-4])
    fixed={"epoch_num":200,"snapshot_num":5,"emb_dim":200,"patience":3,"batch_size":2048,"learning_rate":.0001,
       "margin":8.0,"neg_ratio":10,"regular_weight":.01,"reconstruct_weight":1.0,"lifelong_name":"LKGE",
       "using_regular_loss":"True","using_reconstruct_loss":"True","using_embedding_transfer":"False","using_finetune":"True","valid_metrics":"mrr"}
    for k,v in fixed.items():check(r,"fixed_native_budget_"+k,names.get(k)==v)
    r["effective_configuration"]=names

def ranks(run,snaps,r):
    expected={(i,j,"post_train") for i in range(5) for j in range(i+1)}|{(i,i,"pre_fwt") for i in range(1,5)}
    names={f"train{i}_test{j}_{p}.npz" for i,j,p in expected}
    actual={f.name for f in (run/"ranks").glob("*.npz")}
    check(r,"no_extra_native_rank_files",actual<=names,sorted(actual-names))
    out={};r["rank_files"]={}
    for i,j,phase in sorted(expected):
        name=f"train{i}_test{j}_{phase}.npz";p=run/"ranks"/name
        if not p.is_file():missing(r,name);continue
        with np.load(p,allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
        q=snaps[j]["queries"];n=len(q);x=d["rank"]
        check(r,name+"_schema",set(d)=={"head","relation","tail","rank","candidates","filtered_count","trained_snapshot","tested_snapshot","seed","phase","metric_contract"})
        check(r,name+"_queries",all(np.array_equal(d[k],q[:,a]) for a,k in enumerate(["head","relation","tail"])))
        check(r,name+"_integer_arrays",all(d[k].shape==(n,) and d[k].dtype.kind in "iu" for k in ["head","relation","tail","rank","candidates","filtered_count"]))
        check(r,name+"_metadata",int(d["trained_snapshot"])==i and int(d["tested_snapshot"])==j and int(d["seed"])==55 and str(d["phase"])==phase and str(d["metric_contract"])==CONTRACT)
        check(r,name+"_historical_candidates",bool(np.all(d["candidates"]==snaps[j]["num_ent"])))
        check(r,name+"_independent_filtered_counts",np.array_equal(d["filtered_count"],snaps[j]["filtered_count"]))
        check(r,name+"_rank_bounds",bool(np.all((x>=1)&(x<=snaps[j]["num_ent"]-d["filtered_count"]+1))))
        raw=metrics(x);rounded=native_metrics(x)
        # Keep discrepancies visible: float32 batch rounding and exact float64 are distinct.
        delta={k:raw[k]-rounded[k] for k in KEYS}
        check(r,name+"_finite_metric_recompute",all(np.isfinite(raw[k]) and np.isfinite(rounded[k]) for k in KEYS))
        out[i,j,phase]={"rank":x,"raw":raw,"rounded":rounded}
        r["rank_files"][name]={"sha256":sha(p),"queries":n,"raw_float64":raw,"native_float32_batch100_round4":rounded,
          "raw_minus_native_rounded":delta,"filter_sha256":snaps[j]["filter_sha256"]}
    r["rank_files_found"]=len(out);r["rank_files_expected"]=19
    return out

def equivalence(old,new,r):
    q=torch.tensor([[i,i%2,(3*i)%11] for i in range(7)])
    scores=torch.tensor(np.random.default_rng(655).uniform(0,1,(7,11)),dtype=torch.float64);scores[0]=.5
    labels=torch.zeros(7,11);labels[:,2]=1;labels[torch.arange(7),q[:,2]]=1
    captured=[]
    def capture(args,rows,phase):captured.append((np.concatenate(rows),phase))
    classes=[]
    for text in [old,new]:
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=="DevBatchProcessor")
        env={"torch":torch,"np":np,"_save_observed_query_ranks":capture}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"<lkge-evaluator>","exec"),env)
        classes.append(env["DevBatchProcessor"])
    for method,phase in [("process_epoch","post_train"),("process_epoch","pre_fwt")]:
        results=[];captured.clear()
        for cls,enabled in [(classes[0],False),(classes[1],False),(classes[1],True)]:
            args=types.SimpleNamespace(snapshot=0,snapshot_test=0,valid=False,device=torch.device("cpu"),seed=55,
                 test_FWT=phase=="pre_fwt",rank_export_dir="memory" if enabled else None,expand_dim=False)
            obj=cls.__new__(cls);obj.args=args;obj.kg=types.SimpleNamespace(snapshots={0:types.SimpleNamespace(sr2o_all={})})
            obj.data_loader=[(q[z:z+4,0],q[z:z+4,1],q[z:z+4,2],labels[z:z+4]) for z in range(0,7,4)]
            model=types.SimpleNamespace(eval=lambda:None,args=args,predict=lambda head,relation,stage:scores[head].clone())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore");results.append(getattr(obj,method)(model))
        pred=torch.where(labels.bool(),-torch.ones_like(scores)*1e7,scores)
        pred[torch.arange(7),q[:,2]]=scores[torch.arange(7),q[:,2]]
        rank=1+torch.argsort(torch.argsort(pred,dim=1,descending=True),dim=1)[torch.arange(7),q[:,2]]
        check(r,phase+"_original_observation_metrics_equal",results[0]==results[1]==results[2])
        check(r,phase+"_original_observation_ranks",len(captured)==1 and captured[0][1]==phase and np.array_equal(captured[0][0][:,:3],q.numpy()) and np.array_equal(captured[0][0][:,3],rank.numpy()))
        check(r,phase+"_observation_candidate_filter",len(captured)==1 and bool(np.all(captured[0][0][:,4]==11)) and np.array_equal(captured[0][0][:,5],labels.bool().sum(1).numpy()))
    r["observation_equivalence"]={"queries":7,"batches":[4,3],"exact_ties":True,"phases":["post_train","pre_fwt"],"filesystem_writes":0}

def replay(work,snaps,out,states,r,n):
    if n<=0:r["sampled_replay"]={"status":"DISABLED"};return
    tree=ast.parse((work/"src/model/BaseModel.py").read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="BaseModel")
    methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ["embedding","predict","norm_ent","norm_rel"]]
    node=ast.ClassDef(name="ScoringOnly",bases=[],keywords=[],body=methods,decorator_list=[])
    node=ast.fix_missing_locations(node)
    env={"torch":torch,"nn":nn}
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<lkge-native-scoring-only>","exec"),env)
    records={}
    for i,state in states.items():
        ent=state["ent_embeddings.weight"];rel=state["rel_embeddings.weight"]
        native=env["ScoringOnly"]();native.args=types.SimpleNamespace(snapshot=i,snapshot_test=0,expand_dim=False)
        native.kg=types.SimpleNamespace(snapshots={j:types.SimpleNamespace(num_ent=s["num_ent"]) for j,s in enumerate(snaps)})
        native.ent_embeddings=types.SimpleNamespace(weight=ent);native.rel_embeddings=types.SimpleNamespace(weight=rel)
        norm=F.normalize(ent,2,-1);rnorm=F.normalize(rel,2,-1)
        for j in range(i+1):
            d=out.get((i,j,"post_train"))
            if d is None:continue
            idx=np.unique(np.linspace(0,len(d["rank"])-1,min(n,len(d["rank"])),dtype=int));q=snaps[j]["queries"][idx]
            native.args.snapshot_test=j;predranks=[];exact=[];compat=[];delta=0.
            with torch.no_grad():
                for start in range(0,len(q),4):
                    triples=q[start:start+4];h,p,t=torch.as_tensor(triples).T
                    pred=torch.sigmoid(9-torch.norm((norm[h]+rnorm[p]).unsqueeze(1)-norm[:snaps[j]["num_ent"]],p=1,dim=2))
                    original=native.predict(h,p,stage="Test")
                    delta=max(delta,float(torch.max(torch.abs(pred-original))))
                    target=pred[torch.arange(len(h)),t].clone()
                    for z,(hh,pp,tt) in enumerate(triples):
                        pred[z,snaps[j]["filters"][int(hh),int(pp)]]=-1e7;pred[z,int(tt)]=target[z]
                    rank=1+torch.argsort(torch.argsort(pred,dim=1,descending=True),dim=1)[torch.arange(len(h)),t]
                    lo=1+(pred>target[:,None]).sum(1);hi=(pred>=target[:,None]).sum(1)
                    expected=d["rank"][idx[start:start+4]]
                    predranks+=rank.tolist();exact+=(rank.numpy()==expected).tolist()
                    compat+=((lo.numpy()<=expected)&(expected<=hi.numpy())).tolist()
            name=f"train{i}_test{j}_post_train.npz"
            check(r,name+"_native_vs_independent_scores",delta==0,delta)
            check(r,name+"_checkpoint_rank_replay_exact",all(exact),{"indices":idx.tolist(),"export":d["rank"][idx].tolist(),"cpu":predranks})
            records[name]={"indices":idx.tolist(),"export":d["rank"][idx].tolist(),"cpu":predranks,"samples":len(idx),
                           "exact_matches":sum(exact),"tie_compatible":sum(compat),"native_score_max_abs_diff":delta}
    r["sampled_replay"]={"device":"cpu","files":records,"files_replayed":len(records),"samples":sum(v["samples"] for v in records.values()),
        "exact_matches":sum(v["exact_matches"] for v in records.values()),"scope":"15 post_train evaluations, using five selected raw dim200 checkpoints. Actual LKGE scorer uses neither GCN reconstruction nor old_data/old_weight buffers; no-op snapshot postprocessing proven.",
        "exclusions":"Four pre_fwt files: expanded/initialized pre-training state not saved with exact checkpoint lineage; no replay claimed.",
        "ties":"Exact CPU ranks and GPU export rank membership in exact-score tie intervals separately reported."}
    if len(records)<15:missing(r,"15_post_train_checkpoint_replays")


def log_and_checkpoints(run,snaps,out,log,r):
    prefix=log.split('Final Result:')[0]
    states={};r['best_checkpoints']=[]
    directory=run/'checkpoint/FACT-TransE-LKGE-Margin-WO_transfer'
    headers=re.findall(r'\| Snapshot:(\d+) \|\s*MRR\s*\| Hits@1 \| Hits@3 \| Hits@5 \| Hits@10 \|',prefix)
    check(r,'five_ordered_posttrain_table_headers',headers==list(map(str,range(5))),headers)
    for i in range(5):
        marker=re.search(r'\| Snapshot:'+str(i)+r' \|[^\n]+',prefix)
        if marker:
            after=prefix[marker.end():]
            rows=[]
            for line in after.splitlines():
                if re.match(r'^\s*\|\s*\d+\s*\|',line):
                    vals=[v.strip() for v in line.strip().strip('|').split('|')]
                    if len(vals)==6:rows.append([int(vals[0])]+[float(x) for x in vals[1:]])
                    else:check(r,'table_columns_'+str(i),False)
                elif rows:break
            check(r,'table_exact_rows_'+str(i),len(rows)==i+1 and [x[0] for x in rows]==list(range(i+1)))
            for row in rows:
                j=row[0]
                if (i,j,'post_train') not in out:continue
                m=out[i,j,'post_train']['rounded']
                check(r,f'logged_test_metrics_{i}_{j}',all(abs(v-m[k])<1e-12 for k,v in zip(['mrr','hits1','hits3','hits5','hits10'],row[1:])))
        else:missing(r,'posttrain_table_'+str(i))
        p=directory/f'{i}model_best.tar';side=run/'ranks'/f'selected_checkpoint_snapshot{i}.json'
        if not p.exists() or not side.exists():missing(r,f'best_checkpoint_or_sidecar_{i}');continue
        data=torch.load(p,map_location='cpu',weights_only=True)
        check(r,f'checkpoint_schema_{i}',set(data)=={'state_dict','epoch_id'})
        epoch=data['epoch_id'];state=data['state_dict']
        check(r,f'checkpoint_epoch_{i}',type(epoch) is int and 0<=epoch<200)
        src=directory/f'{i}checkpoint-{epoch}.tar'
        check(r,f'checkpoint_original_epoch_bytes_{i}',src.exists() and sha(src)==sha(p))
        check(r,f'checkpoint_finite_tensors_{i}',all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in state.values()))
        e,rel=state['ent_embeddings.weight'],state['rel_embeddings.weight']
        check(r,f'checkpoint_embedding_shape_dtype_{i}',tuple(e.shape)==(snaps[i]['num_ent'],200) and tuple(rel.shape)==(snaps[i]['num_rel'],200) and e.dtype==rel.dtype==torch.float64)
        sel=json.loads(side.read_text())
        check(r,f'selection_schema_{i}',set(sel)=={'snapshot','epoch','checkpoint','validation_file','best_validation_metric','best_validation_value','dimension'})
        check(r,f'selection_metadata_{i}',sel['snapshot']==i and sel['epoch']==epoch and sel['validation_file']=='valid.txt' and sel['best_validation_metric']=='mrr' and sel['dimension']==200)
        check(r,f'selection_checkpoint_path_{i}',(run/'ranks'/sel['checkpoint']).resolve()==p.resolve())
        records=[]
        for line in prefix.splitlines():
            m=re.search(rf'Snapshot:{i}\s+Epoch:(\d+).*?\tMRR:([0-9.]+).*?\tBest:([0-9.]+)',line)
            if m:records.append((int(m[1]),float(m[2])/100,float(m[3])/100))
        stops=re.findall(rf'Early Stopping! Snapshot:{i} Epoch: (\d+) Best Results: ([0-9.]+)',prefix)
        check(r,f'normal_epoch_contiguity_{i}',bool(records) and [x[0] for x in records]==list(range(len(records))))
        if stops:
            check(r,f'single_earlystop_complete_chronology_{i}',len(stops)==1 and int(stops[0][0])==len(records) and 2<=int(stops[0][0])<200)
            check(r,f'stop_metric_selection_{i}',abs(float(stops[0][1])/100-sel['best_validation_value'])<1e-10)
        else:check(r,f'full_ceiling_complete_{i}',len(records)==200)
        if records:
            best=max(v for ep,v,b in records);first=next(ep for ep,v,b in records if v==best)
            check(r,f'best_epoch_matches_first_logged_strict_max_{i}',epoch==first and abs(best-sel['best_validation_value'])<1e-10,{'epoch':epoch,'first_logged_max':first,'best_logged':best,'sidecar':sel['best_validation_value']})
            check(r,f'logged_running_best_{i}',all(abs(b-max(v for ep2,v,b2 in records if ep2<=ep))<1e-10 for ep,v,b in records))
            counter=0;old=-1
            for ep,v,b in records:
                if v>old:counter=max(0,counter-5);old=v
                else:counter+=1
                check(r,f'earlystop_counter_not_terminal_before_last_{i}_{ep}',counter<3)
            if stops:check(r,f'earlystop_counter_at_stop_{i}',counter==2)
        r['best_checkpoints'].append({'snapshot':i,'epoch':epoch,'dimension':200,'sha256':sha(p),'source_epoch_sha256':sha(src),'selection_sha256':sha(side),'best_validation_metric':sel['best_validation_value'],'normal_epochs':len(records),'earlystop_epoch':int(stops[0][0]) if stops else None})
        states[i]=state
    after=log.split('Report Result:')[-1]
    lines=[x.strip() for x in after.splitlines() if x.strip().startswith('|')]
    check(r,'final_report_header',bool(lines) and [x.strip() for x in lines[0].strip('|').split('|')]==['Snapshot','Time','Whole_MRR','Whole_Hits@1','Whole_Hits@3','Whole_Hits@10'])
    rows=[[x.strip() for x in l.strip('|').split('|')] for l in lines[1:]]
    check(r,'final_report_five_ordered_rows',len(rows)==5 and all(len(x)==6 for x in rows) and [x[0] for x in rows]==list(map(str,range(5))))
    final={};times=[]
    for i,row in enumerate(rows):
        if len(row)!=6 or i>4:continue
        m=overall(out,snaps,i,'post_train');times.append(float(row[1]));check(r,'final_overall_'+str(i),all(abs(float(v)-m[k])<1e-12 for k,v in zip(m,row[2:])))
        final[str(i)]={'native_round3':m,'raw_float64':overall(out,snaps,i,'post_train',raw=True)}
    t=re.search(r'Sum_Training_Time:([0-9.]+)',after)
    each=re.search(r'Every_Training_Time:(\[[^\n]+\])',after)
    check(r,'training_time_sum',t is not None and abs(float(t[1])-sum(times))<1e-8)
    check(r,'training_time_list',each is not None and ast.literal_eval(each[1])==times)
    fwt=np.mean([out[i,i,'pre_fwt']['rounded']['mrr'] for i in range(1,5)])
    bwt=np.mean([out[4,i,'post_train']['rounded']['mrr']-out[i,i,'post_train']['rounded']['mrr'] for i in range(4)])
    z=re.search(r'Forward transfer: ([-+.0-9eE]+)\s+Backward transfer: ([-+.0-9eE]+)',after)
    check(r,'native_FWT_BWT',z is not None and abs(float(z[1])-fwt)<1e-12 and abs(float(z[2])-bwt)<1e-12)
    r['native_overall']=final;r['native_train_loop_seconds']=sum(times);r['transfer']={'fwt':float(fwt),'bwt':float(bwt)}
    return states

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--replay-samples',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(args.threads)
    root=args.root.resolve();run=root/'runs/lkge-clean-fact-s55';start=time.monotonic()
    r={'id':'lkge-clean-fact-s55','method':'LKGE_clean_split','dataset':'FACT','checks':0,'failures':[],'missing':[],'read_only':True,'auditor_sha256':sha(__file__),'scope':'independent source/provenance/data/rank arithmetic and sampled CPU posttrain checkpoint inference','torch':torch.__version__,'numpy':np.__version__}
    try:
        job,work,repo,terminal=bind(root,run,r)
        if terminal:
            log=(run/'run.log').read_text();defaults_audit(job,work,run,log,r);old,new=source_audit(work,repo,r);equivalence(old,new,r)
            snaps=rebuild(work/'data/FACT',r);out=ranks(run,snaps,r)
            if len(out)==19:
                states=log_and_checkpoints(run,snaps,out,log,r);replay(work,snaps,out,states,r,args.replay_samples)
            else:missing(r,'complete19rank_files')
            artifacts=[run/'registered-job.json',run/'copied-source-sha256.json',run/'runtime-command.json',run/'terminal.json',run/'run.log',*sorted((run/'ranks').glob('*')),*sorted((run/'checkpoint/FACT-TransE-LKGE-Margin-WO_transfer').glob('*model_best.tar'))]
            r['artifact_sha256']={str(p.relative_to(run)):sha(p) for p in artifacts if p.is_file()}
    except Exception as err:
        check(r,'auditor_exception',False,type(err).__name__+': '+str(err))
    r['status']='FAIL' if r['failures'] else 'INCOMPLETE' if r['missing'] else 'PASS_WITH_LIMITATIONS'
    r['limitations']=['Clean-validation adaptation, not exact test-selected upstream protocol; single seed55, not five-run paper mean.',
        'Historical query-snapshot candidate/filter and double-argsort semantics retained.',
        '15 posttrain files sampled8/file by default; no complete score certification or pre-FWT inference replay.',
        'Best sidecar and logs/epoch bytes verified; no independent validation ranking or complete training/optimizer reconstruction.',
        'Independent checker completed by controller after sidecar agent hit usage limit; no separate completed code review claimed.']
    r['elapsed_s']=time.monotonic()-start
    print(json.dumps(r,indent=2,allow_nan=False))
    return 1 if r['failures'] else 2 if r['missing'] else 0
if __name__=='__main__':raise SystemExit(main())
