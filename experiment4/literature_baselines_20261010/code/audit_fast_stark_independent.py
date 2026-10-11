#!/usr/bin/env python3
"""Read-only CPU audit of released FastKGE/STARK FB_CKGE native runs.
JSON goes to stdout; no output, environment, or canonical files are written.
Exit 0 = full checked evidence; 1 = failure; 2 = incomplete.
Checkpoint replay independently assembles saved merged LoRA weights, then
restores native objects/buffers in memory for a second prediction comparison.
"""
import argparse, ast, builtins, contextlib, copy, hashlib, io, json, os
from pathlib import Path
import re, subprocess, sys, time, types
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F

COMMITS = {"FastKGE": "777b3d88f2adf3a47cc773072e1fc53aba558f38",
           "STARK": "2e546b30688c660903247df89330671f4f73bf68"}
KEYS = ["mrr", "hits1", "hits3", "hits5", "hits10"]
SOURCE_FILES = ["main.py","src/parse_args.py","src/data_load/KnowledgeGraph.py",
 "src/data_load/data_loader.py","src/model/BaseModel.py","src/model/LoraKGE_Layers.py",
 "src/model/model_process.py","src/train.py","src/test.py","src/utils.py","main.sh"]
def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",",":")).encode()).hexdigest()
def git(repo, *cmd):
    return subprocess.check_output(["git", *cmd], cwd=repo, text=True)
def check(report, name, ok, detail=None):
    report["checks"] += 1
    if not ok: report["failures"].append({"check":name, "detail":detail})
def waiting(report, value):
    if value not in report["missing"]: report["missing"].append(value)
def source_tree(text):
    return ast.parse(text)
def tree_eq(a,b):
    return ast.dump(a, include_attributes=False)==ast.dump(b, include_attributes=False)
class StripObservations(ast.NodeTransformer):
    def visit_Assign(self,n):
        if any(isinstance(x,ast.Name) and x.id=="observations" for x in n.targets): return None
        return self.generic_visit(n)
    def visit_If(self,n):
        s=ast.unparse(n.test)
        if ("observation_path" in s) or s=="observations": return None
        return self.generic_visit(n)
def strip_parser(tree):
    new=[]
    for stmt in tree.body:
        if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call):
            call=stmt.value
            if isinstance(call.func,ast.Attribute) and call.func.attr=="add_argument":
                if any(isinstance(a,ast.Constant) and a.value=="-observation_path" for a in call.args): continue
        new.append(stmt)
    tree.body=new
    return tree
def source_audit(repo, method, report):
    current=git(repo,"rev-parse","HEAD").strip()
    check(report,"pinned_upstream",current==COMMITS[method],current)
    diff=git(repo,"diff")
    patch=repo/"reproduction/compatibility.patch"
    check(report,"preserved_diff",patch.exists() and patch.read_text()==diff)
    changes=git(repo,"diff","--name-only").splitlines()
    check(report,"allowed_changed_files",set(changes)<={
        "main.py","src/parse_args.py","src/model/BaseModel.py","src/model/model_process.py"},changes)
    sources={}
    for f in SOURCE_FILES:
        old=git(repo,"show","HEAD:"+f); new=(repo/f).read_text()
        sources[f]={"original_sha256":hashlib.sha256(old.encode()).hexdigest(),"current_sha256":sha(repo/f)}
        if not f.endswith(".py"):
            check(report,"unchanged_"+f,old==new)
            continue
        a,b=source_tree(old),source_tree(new)
        if f in ["main.py","src/model/model_process.py"]:
            b=StripObservations().visit(b)
        elif f=="src/model/BaseModel.py":
            a.body=[x for x in a.body if not(isinstance(x,ast.ImportFrom) and x.module=="torch_scatter")]
            check(report,"unused_scatter_import",not any(
                isinstance(x,ast.Name) and x.id in {"scatter_add","scatter_mean","scatter_max"}
                for x in ast.walk(a)))
        elif f=="src/parse_args.py":
            b=strip_parser(b)
            oldcalls=[x for x in ast.walk(a) if isinstance(x,ast.Call)]
            newcalls=[x for x in ast.walk(b) if isinstance(x,ast.Call)]
            for c,d in zip(oldcalls,newcalls):
                if isinstance(c.func,ast.Attribute) and c.func.attr=="add_argument":
                    if not any(k.arg=="type" for k in c.keywords):
                        added=[k for k in d.keywords if k.arg=="type"]
                        check(report,"numeric_parser_type_"+ast.unparse(c.args[0]),
                              all(isinstance(k.value,ast.Name) and k.value.id in {"int","float"} for k in added))
                        d.keywords=[k for k in d.keywords if k.arg!="type"]
        check(report,"original_algorithm_ast_"+f,tree_eq(a,b))
    manifest=json.loads((repo/"reproduction/source_manifest.json").read_text())
    for x in manifest["files"]:
        p=repo/x["path"]
        check(report,"prepared_manifest_"+x["path"],p.is_file() and sha(p)==x["sha256"])
    report["source"]={"commit":current,"diff_sha256":hashlib.sha256(diff.encode()).hexdigest(),
                      "files":sources}
    return git(repo,"show","HEAD:src/model/model_process.py"),(repo/"src/model/model_process.py").read_text()
def rebuild(data, report):
    entities={}; relations={}; filters={}; snapshots=[]; hashes={}; seen=set()
    for s in range(5):
        raw={}
        for split in ["train","valid","test"]:
            f=data/str(s)/(split+".txt"); hashes[str(f.relative_to(data))]=sha(f)
            rows=[tuple(l.split()) for l in f.read_text().splitlines() if l.strip()]
            check(report,f"triples_{s}_{split}",all(len(x)==3 for x in rows))
            raw[split]=rows
            for h,r,t in rows:
                for e in [h,t]:
                    if e not in entities: entities[e]=len(entities)
                if r not in relations:
                    q=len(relations); relations[r]=q; relations[r+"_inv"]=q+1
        sets=[set(raw[k]) for k in ["train","valid","test"]]
        check(report,f"split_disjoint_{s}",not any(sets[i]&sets[j] for i in range(3) for j in range(i)))
        allnew=set.union(*sets)
        check(report,f"no_cross_snapshot_duplicate_{s}",not seen&allnew); seen |= allnew
        mapped={}
        for split,rows in raw.items():
            facts=[(entities[h],relations[r],entities[t]) for h,r,t in rows]
            if split!="train": facts.sort(key=lambda x:x[1]) # stable order
            mapped[split]=facts
            for h,r,t in facts:
                filters.setdefault((h,r),set()).add(t)
                filters.setdefault((t,r^1),set()).add(h)
        for filename,mapping in [("entity2id.txt",entities),("relation2id.txt",relations)]:
            f=data/str(s)/filename
            rows=[l.split() for l in f.read_text().splitlines() if l.strip()]
            actual={x[0]:int(x[1]) for x in rows if len(x)==2}
            check(report,f"author_id_file_{s}_{filename}",actual==mapping)
            hashes[str(f.relative_to(data))]=sha(f)
        trainid=data/str(s)/"train_id.txt"
        ids=[tuple(map(int,l.split())) for l in trainid.read_text().splitlines() if l.strip()]
        check(report,f"author_train_ids_{s}",ids==mapped["train"])
        hashes[str(trainid.relative_to(data))]=sha(trainid)
        facts=mapped["test"]
        queries=np.asarray(facts+[(t,r+1,h) for h,r,t in facts],dtype=np.int64)
        frozen={k:sorted(v) for k,v in filters.items()}
        filhash=digest([[h,r,v] for (h,r),v in sorted(frozen.items())])
        snapshots.append({"num_ent":len(entities),"num_rel":len(relations),"queries":queries,
          "filters":frozen,"test":mapped["test"],"valid":mapped["valid"],"train":mapped["train"],
          "filter_sha256":filhash})
        check(report,f"filters_in_candidates_{s}",all(max(v)<len(entities) for v in frozen.values()))
    report["data"]={"sha256":hashes,"ids_sha256":digest([entities,relations]),
        "snapshots":[{k:v for k,v in s.items() if k in ["num_ent","num_rel","filter_sha256"]}
                     | {"test_triples":len(s["test"]),"queries":len(s["queries"])} for s in snapshots]}
    return snapshots
def metrics(rank):
    rank=rank.astype(np.float64)
    return {"mrr":float(np.mean(1/rank)),"mr":float(rank.mean()),
            **{f"hits{k}":float(np.mean(rank<=k)) for k in range(1,11)}}
def native_round(rank):
    # Separate CPU implementation of author float32 reciprocal sums in batches of100.
    a=torch.as_tensor(rank).float()
    sums={"mrr":sum(float(torch.sum(1/a[i:i+100])) for i in range(0,len(a),100)),
          "mr":sum(float(torch.sum(a[i:i+100])) for i in range(0,len(a),100))}
    sums.update({f"hits{k}":int((a<=k).sum()) for k in range(1,11)})
    return {k:round(v/len(a),4) for k,v in sums.items()}
class MemoryPath:
    def __init__(self,*a): self.value="/".join(map(str,a))
    def __truediv__(self,b): return MemoryPath(self.value,b)
    def mkdir(self,**kw): pass
    def exists(self): return False
    def __str__(self): return self.value
def equivalence(old,new,report):
    captured=[]
    def importer(name,*args,**kwargs):
        if name=="pathlib": return types.SimpleNamespace(Path=MemoryPath)
        return builtins.__import__(name,*args,**kwargs)
    scores=torch.tensor(np.random.default_rng(1701).uniform(0,1,(7,11)),dtype=torch.float64)
    scores[0]=0.5 # exact-tie stress, original tie policy
    triples=torch.tensor([[i,i%2,(i*3)%11] for i in range(7)])
    labels=torch.zeros(7,11); labels[:,2]=1
    labels[torch.arange(7),triples[:,2]]=1
    classes=[]
    for text in [old,new]:
        node=next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef) and x.name=="DevBatchProcessor")
        env={"torch":torch,"np":types.SimpleNamespace(concatenate=np.concatenate,
            savez_compressed=lambda path,**kw:captured.append(kw)),
             "__builtins__":dict(vars(builtins),__import__=importer)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"<native-evaluator>","exec"),env)
        classes.append(env["DevBatchProcessor"])
    outputs=[]
    for cls,enabled in [(classes[0],False),(classes[1],False),(classes[1],True)]:
        args=types.SimpleNamespace(model_name="LoraKGE_Layers",using_various_ranks=True,
          using_various_ranks_reverse=False,device=torch.device("cpu"),snapshot=0,snapshot_test=0,
          valid=False,predict_result=False,observation_path="memory" if enabled else None,test_FWT=False)
        kg=types.SimpleNamespace(snapshots={0:types.SimpleNamespace(num_ent=11,hr2t_all={})})
        model=types.SimpleNamespace(eval=lambda:None,lora_ent_embeddings_list=None,
                                   predict=lambda head,relation,stage:scores[head].clone())
        obj=cls.__new__(cls); obj.args=args; obj.kg=kg
        obj.data_loader=[(triples[i:i+4,0],triples[i:i+4,1],triples[i:i+4,2],labels[i:i+4])
                         for i in range(0,7,4)]
        outputs.append(obj.process_epoch(model))
    p=torch.where(labels.bool(),-torch.ones_like(scores)*10000000,scores)
    p[torch.arange(7),triples[:,2]]=scores[torch.arange(7),triples[:,2]]
    ranks=1+torch.argsort(torch.argsort(p,dim=1,descending=True),dim=1)[torch.arange(7),triples[:,2]]
    check(report,"original_observed_disabled_enabled_metrics",outputs[0]==outputs[1]==outputs[2])
    check(report,"observation_exact_ranks",len(captured)==1 and np.array_equal(captured[0]["rank"],ranks.numpy()))
    check(report,"observation_exact_queries",len(captured)==1 and all(
        np.array_equal(captured[0][k],triples[:,i].numpy()) for i,k in enumerate(["head","relation","tail"])))
    report["observation_equivalence"]={"fixture_rows":7,"batches":[4,3],"tie_stress":True,
        "disk_writes":0,"metrics":outputs[0],"ranks":ranks.tolist()}
def rank_audit(run,snaps,report):
    found={}; expected={(i,j,"after") for i in range(5) for j in range(i+1)}
    expected |= {(i,i,"before") for i in range(1,5)}
    report["task_matrix"]={}
    for i,j,phase in sorted(expected):
        name=f"train{i}_test{j}_{phase}.npz"; path=run/"observations"/name
        if not path.exists(): waiting(report,name); continue
        try:
            with np.load(path,allow_pickle=False) as f: d={k:f[k].copy() for k in f.files}
            q=snaps[j]["queries"]; rank=d["rank"]
            check(report,name+"_query_order",all(np.array_equal(d[k],q[:,a]) for a,k in enumerate(["head","relation","tail"])))
            check(report,name+"_rank_integral",rank.shape==(len(q),) and rank.dtype.kind in "iu")
            check(report,name+"_bounds",bool(np.all((rank>=1)&(rank<=snaps[j]["num_ent"]))))
            check(report,name+"_candidate_universe",int(d["candidate_count"])==snaps[j]["num_ent"])
            check(report,name+"_metadata",int(d["training_snapshot"])==i and int(d["test_snapshot"])==j and str(d["phase"])==phase)
            for h,r,t in q:
                if int(t) not in snaps[j]["filters"][(int(h),int(r))]:
                    check(report,name+"_target_in_filter",False); break
            m=metrics(rank); rounded=native_round(rank)
            check(report,name+"_float64_rounding",all(round(m[k],4)==rounded[k] for k in KEYS),
                  {"float64":{k:round(m[k],4) for k in KEYS},"native_float32":rounded})
            found[(i,j,phase)]={"rank":rank,"queries":q,"raw":m,"rounded":rounded}
            report["task_matrix"][name]={"sha256":sha(path),"queries":len(rank),"raw":m,"native_rounded4":rounded,
                "filter_sha256":snaps[j]["filter_sha256"]}
        except Exception as e: check(report,name+"_read",False,str(e))
    actual={p.name for p in (run/"observations").glob("train*_test*_*.npz")}
    check(report,"no_unexpected_rank_exports",actual<= {f"train{i}_test{j}_{p}.npz" for i,j,p in expected},sorted(actual))
    report["rank_files_found"]=len(found); report["rank_files_expected"]=19
    return found
def logs_and_selection(repo,run,snaps,found,report):
    logs=list(run.glob("logs/**/*.log"))
    check(report,"single_native_log",len(logs)==1,[str(x.relative_to(run)) for x in logs])
    text=logs[0].read_text() if logs else ""
    report["log_sha256"]=sha(logs[0]) if logs else None
    # Read tables separately from the final duplicated list of PrettyTables.
    segment=text.split("Final Result:")[0]
    tables={}
    for match in re.finditer(r"\|\s*Snapshot:(\d+)\s*\|",segment):
        i=int(match.group(1)); block=segment[match.end():]
        boundary=re.search(r"\n\d{4}-\d\d-\d\d ",block)
        if boundary: block=block[:boundary.start()]
        rows={}
        for line in block.splitlines():
            parts=[x.strip() for x in line.strip().strip("|").split("|")]
            if len(parts)==6 and parts[0].isdigit():
                rows[int(parts[0])]=dict(zip(KEYS,map(float,parts[1:])))
        tables[i]=rows
    for (i,j,p),d in found.items():
        if p=="after":
            if i not in tables: waiting(report,f"log_task_table_{i}")
            else: check(report,f"log_task_{i}_{j}",tables[i].get(j)=={k:d["rounded"][k] for k in KEYS},tables[i].get(j))
    overall={}; overallraw={}
    for i in range(5):
        if all((i,j,"after") in found for j in range(i+1)):
            weights=[len(snaps[j]["test"]) for j in range(i+1)]
            overall[i]={k:round(sum(found[i,j,"after"]["rounded"][k]*weights[j]
                           for j in range(i+1))/sum(weights),3) for k in ["mrr","hits1","hits3","hits10"]}
            overallraw[i]={k:sum(found[i,j,"after"]["raw"][k]*weights[j] for j in range(i+1))/sum(weights)
                            for k in ["mrr","hits1","hits3","hits10"]}
    final=text.split("Report Result:")[-1] if "Report Result:" in text else ""
    parsed_overall={}; times=[]
    for line in final.splitlines():
        parts=[x.strip() for x in line.strip().strip("|").split("|")]
        if len(parts)==6 and parts[0].isdigit():
            parsed_overall[int(parts[0])]=dict(zip(["mrr","hits1","hits3","hits10"],map(float,parts[2:])))
            times.append(float(parts[1]))
    for i,row in overall.items():
        if i in parsed_overall: check(report,f"log_overall_{i}",row==parsed_overall[i],{"expected":row,"log":parsed_overall[i]})
        else: waiting(report,f"log_overall_{i}")
    report["overall_native_rounded3"]=overall; report["overall_raw"]=overallraw
    transfer=re.search(r"Forward transfer:\s*([-+.\deE]+)\s+Backward transfer:\s*([-+.\deE]+)",text)
    if all((i,i,"before") in found for i in range(1,5)) and all((4,j,"after") in found for j in range(5)):
        fwt=sum(found[i,i,"before"]["rounded"]["mrr"] for i in range(1,5))/4
        bwt=sum(found[4,j,"after"]["rounded"]["mrr"]-found[j,j,"after"]["rounded"]["mrr"] for j in range(4))/4
        report["transfer"]={"FWT":fwt,"BWT":bwt}
        if transfer: check(report,"log_transfer",abs(float(transfer[1])-fwt)<1e-12 and abs(float(transfer[2])-bwt)<1e-12)
        else: waiting(report,"terminal_transfer_log")
    timing=re.search(r"Sum_Training_Time:([-+.\deE]+)",text)
    if timing:
        check(report,"training_time_sum",len(times)==5 and abs(sum(times)-float(timing[1]))<1e-6)
        report["native_training_seconds"]=float(timing[1])
    else: waiting(report,"terminal_training_log")
    exitfile=run/"process_exit.json"
    if exitfile.exists():
        e=json.loads(exitfile.read_text()); report["process_exit"]=e
        check(report,"process_exit0",e.get("returncode")==0)
    else: waiting(report,"process_exit.json")
    states={}; report["checkpoints"]=[]
    for i in range(5):
        meta=run/"observations"/f"selected_checkpoint_snapshot{i}.json"
        checkpoint=run/"checkpoints/FB_CKGE"/f"{i}model_best.tar"
        # Wait for evaluation table, so running selected metadata cannot be mistaken for final.
        if not meta.exists() or not checkpoint.exists() or i not in tables:
            waiting(report,f"final_best_checkpoint_{i}"); continue
        d=json.loads(meta.read_text()); epoch=int(d["epoch"])
        check(report,f"selected_{i}_identity",d["snapshot"]==i and d["lora_only"]==(i>0)
              and d["best_validation_metric"]=="mrr" and 0<=epoch<200
              and (meta.parent/d["checkpoint"]).resolve()==checkpoint.resolve())
        epochpath=checkpoint.parent/f"{i}checkpoint-{epoch}.tar"
        check(report,f"best_matches_epoch_{i}",epochpath.exists() and sha(epochpath)==sha(checkpoint))
        records=[]
        for line in segment.splitlines():
            if re.search(rf"Snapshot:{i}\s",line) and "\tEpoch:" in line:
                m=re.search(r"Epoch:(\d+).*?\tMRR:([-+.\deE]+)",line)
                if m: records.append((int(m[1]),float(m[2])/100))
        if records:
            best=max(v for _,v in records); first=next(ep for ep,v in records if v==best)
            check(report,f"validation_selected_{i}",epoch==first and abs(float(d["best_validation_value"])-best)<1e-10,
                  {"metadata":d,"logged_first_max":[first,best]})
        else: check(report,f"validation_log_{i}",False)
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if i==0:
            check(report,"checkpoint0_epoch",state.get("epoch_id")==epoch)
            state=state["state_dict"]
        check(report,f"checkpoint_{i}_finite",all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in state.values()))
        states[i]=state
        report["checkpoints"].append({"snapshot":i,"selection":d,"sha256":sha(checkpoint),
            "metadata_sha256":sha(meta),"tensors":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in state.items()}})
    return states
def feature_order(repo,method,snaps,i,report):
    before=snaps[i-1]["num_ent"]; now=snaps[i]["num_ent"]
    path=repo/"data/FB_CKGE"/str(i)/("train_distance_nodes.txt" if method=="FastKGE" else "train_distance_nodes2.txt")
    values={j:(0,0.0) if method=="FastKGE" else 0.0 for j in range(before,now)}
    for l in path.read_text().splitlines():
        a=l.split()
        if int(a[0]) in values:
            values[int(a[0])]=(int(a[1]),float(a[2])) if method=="FastKGE" else float(a[1])
    order=sorted(values,key=lambda x:values[x],reverse=method=="STARK")
    report.setdefault("features",{})[i]={"sha256":sha(path),"order_sha256":digest(order),"path":str(path.relative_to(repo))}
    layers=2 if method=="FastKGE" else 20
    length=(now-before+layers-1)//layers
    scores=[values[j][1] if method=="FastKGE" else values[j] for j in order]
    sums=[sum(scores[j*length:(j+1)*length]) for j in range(layers)]
    average=sum(sums)/layers
    base=max(150,length//20) if method=="FastKGE" else 200
    threshold=int(base*(0.9 if method=="FastKGE" else 0.25))
    ranks=[int(max(base*x/average,threshold)) or 1 for x in sums]
    report["features"][i].update({"group_length":length,"adaptive_ranks":ranks})
    return order
def output_lineage(repo,method,report):
    directory=repo/"embeddings/FB_CKGE"
    rows=[]
    if method=="STARK" and directory.exists():
        for path in sorted(directory.glob("*/*.npy")):
            try:
                arr=np.load(path,mmap_mode="r",allow_pickle=False)
                rows.append({"path":str(path.relative_to(repo)),"sha256":sha(path),
                    "shape":list(arr.shape),"dtype":str(arr.dtype),"mtime_ns":path.stat().st_mtime_ns})
            except Exception as exc:
                rows.append({"path":str(path.relative_to(repo)),"read_error":repr(exc)})
    report["repository_embedding_outputs"]={"files":rows,
        "lineage":"Native STARK run() exports after switch_snapshot(), in repository-relative embeddings/FB_CKGE, not run directory. Snapshot0-3 include next random expansion; final snapshot4 is assembled. Paths may be overwritten by future runs; inventory is read-time evidence only."}
def class_namespace(repo):
    # Execute only class definitions, never main/parser/import side effects.
    sys.path.insert(0,str(repo/"reproduction/vendor"))
    import loralib
    env={"torch":torch,"nn":torch.nn,"Parameter":torch.nn.Parameter,"F":F,"np":np,
         "deepcopy":copy.deepcopy,"xavier_normal_":torch.nn.init.xavier_normal_,"loralib":loralib}
    for f in ["src/model/BaseModel.py","src/model/LoraKGE_Layers.py"]:
        nodes=[x for x in ast.parse((repo/f).read_text()).body if isinstance(x,ast.ClassDef)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(repo/f),"exec"),env)
    return env,loralib
def restored_model(env,lora,state,old,order,i,snaps,layers,report):
    # Native checkpoints are saved after validation eval() merged the LoRA weights.
    # Set merged=True before load: no random initialization and no duplicate BA/r.
    model=env["TransE"].__new__(env["TransE"])
    torch.nn.Module.__init__(model)
    model.args=types.SimpleNamespace(snapshot=i,snapshot_test=i,snapshot_valid=i,device=torch.device("cpu"),
                                   num_ent_layers=layers,emb_dim=200)
    model.kg=types.SimpleNamespace(snapshots={j:types.SimpleNamespace(num_ent=s["num_ent"],num_rel=s["num_rel"]) for j,s in enumerate(snaps)})
    if i==0:
        model.ent_embeddings=torch.nn.Embedding.from_pretrained(state["ent_embeddings.weight"].clone(),freeze=True)
        model.rel_embeddings=torch.nn.Embedding.from_pretrained(state["rel_embeddings.weight"].clone(),freeze=True)
        model.lora_ent_embeddings_list=None; model.lora_rel_embeddings=None
        return model
    model.register_buffer("old_data_ent_embeddings_weight",old[0].clone())
    model.register_buffer("old_data_rel_embeddings_weight",old[1].clone())
    model.new_ordered_entities=order
    modules=[]
    # These plain nn.Embedding objects need exactly the loralib.Embedding metadata
    # and parameters; __new__ avoids irrelevant RNG-driven initialization.
    for prefix in [f"lora_ent_embeddings_list.{j}" for j in range(layers)]+["lora_rel_embeddings"]:
        w,a,b=(state[prefix+"."+k] for k in ["weight","lora_A","lora_B"])
        obj=lora.Embedding.__new__(lora.Embedding); torch.nn.Module.__init__(obj)
        obj.num_embeddings=w.shape[0]; obj.embedding_dim=w.shape[1]
        obj.padding_idx=None; obj.max_norm=None; obj.norm_type=2.; obj.scale_grad_by_freq=False; obj.sparse=False
        obj.r=a.shape[0]; obj.lora_alpha=1; obj.scaling=1/obj.r
        obj.merge_weights=True; obj.merged=True
        for name,v in [("weight",w),("lora_A",a),("lora_B",b)]:
            obj.register_parameter(name,torch.nn.Parameter(v.clone(),requires_grad=False))
        modules.append(obj)
    model.lora_ent_embeddings_list=torch.nn.ModuleList(modules[:-1])
    model.lora_rel_embeddings=modules[-1]; model.lora_ent_len=modules[0].num_embeddings
    # store_old_parameters() persists previous LoRA A/B as top-level buffers.
    # lora_state_dict() includes these too because their names contain "lora_".
    # Restore all saved buffers; do not ignore unexpected keys to force a pass.
    saved_buffers={k:v for k,v in state.items() if k.startswith("old_data_")}
    for name,value in saved_buffers.items():
        model.register_buffer(name,value.clone())
    incompatible=model.load_state_dict(state,strict=False)
    expected_missing={"old_data_ent_embeddings_weight","old_data_rel_embeddings_weight"}
    check(report,f"restored_native_state_{i}",not incompatible.unexpected_keys and
          set(incompatible.missing_keys)==expected_missing,
          {"missing_keys":incompatible.missing_keys,"expected_missing_rebuilt_from_previous_best":sorted(expected_missing),
           "unexpected_keys":incompatible.unexpected_keys,"saved_buffers_restored":sorted(saved_buffers)})
    model.eval()
    restored=model.state_dict()
    check(report,f"all_persisted_state_tensors_equal_{i}",
          all(k in restored and torch.equal(v,restored[k]) for k,v in state.items()))
    return model
def checkpoint_replay(repo,method,snaps,found,states,report,n):
    if n<=0:
        report["replay"]={"status":"DISABLED","limitation":"No independent checkpoint inference."}; return
    env,lora=class_namespace(repo)
    layers=2 if method=="FastKGE" else 20
    assembled={}; report["replay"]={"device":"cpu","samples_per_file":n,"files":{},"FWT":"Not replayed: initial increment RNG states were not saved.",
                                  "tie_policy":"CPU double-argsort exact match; GPU/CPU score-tie intervals checked separately.",
                                  "state_scope":"All checkpoint-covered parameters and saved old_data LoRA buffers restored; old entity/relation prediction buffers rebuilt through previous best states. Unused expanded random base embedding caches and optimizer/RNG training state are not claimed recoverable."}
    for i in range(5):
        if i not in states or (i>0 and i-1 not in assembled): continue
        state=states[i]; old=assembled.get(i-1); order=[]
        if i>1:
            previous=states[i-1]
            previous_buffers={"old_data_"+k.replace(".","_"):v for k,v in previous.items()
                              if not k.startswith("old_data_") and (k.endswith(".lora_A") or k.endswith(".lora_B"))}
            persisted={k:v for k,v in state.items() if k.startswith("old_data_")}
            check(report,f"saved_old_lora_buffers_lineage_{i}",
                  set(persisted)==set(previous_buffers) and
                  all(torch.equal(v,previous_buffers[k]) for k,v in persisted.items()),
                  {"saved":sorted(persisted),"previous_A_B":sorted(previous_buffers)})
        if i==0:
            ent=state["ent_embeddings.weight"].clone(); rel=state["rel_embeddings.weight"].clone()
        else:
            order=feature_order(repo,method,snaps,i,report)
            weights=[state[f"lora_ent_embeddings_list.{j}.weight"] for j in range(layers)]
            check(report,f"lora_group_length_{i}",len({w.shape[0] for w in weights})==1)
            for j,w in enumerate(weights):
                a=state[f"lora_ent_embeddings_list.{j}.lora_A"]; b=state[f"lora_ent_embeddings_list.{j}.lora_B"]
                check(report,f"lora_matrix_contract_{i}_{j}",a.shape[1]==w.shape[0] and b.shape==(200,a.shape[0]))
            idx=torch.tensor(list(range(snaps[i-1]["num_ent"]))+order)
            ent=torch.cat([old[0],*weights],dim=0)[idx].clone()
            rel=torch.cat([old[1],state["lora_rel_embeddings.weight"]],dim=0).clone()
        check(report,f"assembled_shapes_{i}",tuple(ent.shape)==(snaps[i]["num_ent"],200) and tuple(rel.shape)==(snaps[i]["num_rel"],200))
        assembled[i]=(ent,rel)
        model=restored_model(env,lora,state,old,order,i,snaps,layers,report)
        norm=F.normalize(ent,2,-1); relnorm=F.normalize(rel,2,-1)
        for j in range(i+1):
            d=found.get((i,j,"after"))
            if not d: continue
            inds=np.unique(np.linspace(0,len(d["rank"])-1,min(n,len(d["rank"])),dtype=int))
            queries=d["queries"][inds]; model.args.snapshot_test=j
            replay=[]; compatible=[]; exact=[]; maxscore=0.
            # Small row chunks bound CPU memory; full candidate set is always used.
            with torch.no_grad():
                for start in range(0,len(inds),4):
                    q=queries[start:start+4]; h,r,t=torch.as_tensor(q).T
                    pred=torch.sigmoid(9-torch.norm((norm[h]+relnorm[r]).unsqueeze(1)-norm[:snaps[j]["num_ent"]],p=1,dim=2))
                    native=model.predict(h,r,stage="Test")
                    maxscore=max(maxscore,float(torch.max(torch.abs(pred-native))))
                    target=pred[torch.arange(len(q)),t].clone()
                    for z,(hh,rr,tt) in enumerate(q):
                        pred[z,snaps[j]["filters"][int(hh),int(rr)]]=-1e7
                        pred[z,int(tt)]=target[z]
                    rank=1+torch.argsort(torch.argsort(pred,dim=1,descending=True),dim=1)[torch.arange(len(q)),t]
                    expected=d["rank"][inds[start:start+4]]
                    lo=1+(pred>target[:,None]).sum(1); hi=(pred>=target[:,None]).sum(1)
                    replay.extend(rank.tolist())
                    exact.extend((rank.numpy()==expected).tolist())
                    compatible.extend(((lo.numpy()<=expected)&(expected<=hi.numpy())).tolist())
            name=f"train{i}_test{j}_after.npz"
            check(report,name+"_native_vs_independent_scores",maxscore<1e-12,maxscore)
            check(report,name+"_cpu_rank_replay",all(compatible),{"indices":inds.tolist(),"expected":d["rank"][inds].tolist(),"cpu":replay})
            report["replay"]["files"][name]={"indices":inds.tolist(),"expected":d["rank"][inds].tolist(),
                    "cpu_rank":replay,"exact_matches":sum(exact),"tie_compatible":sum(compatible),
                    "samples":len(inds),"native_vs_independent_score_max_abs":maxscore}
    report["replay"]["files_replayed"]=len(report["replay"]["files"])
    report["replay"]["exact_matches"]=sum(x["exact_matches"] for x in report["replay"]["files"].values())
    report["replay"]["samples"]=sum(x["samples"] for x in report["replay"]["files"].values())
    if report["replay"]["files_replayed"]<15: waiting(report,"15_after_checkpoint_replays")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--method",choices=list(COMMITS),required=True)
    p.add_argument("--run",type=Path,help="Defaults to root/runs/METHOD-fbckge-s3407/native/seed3407")
    p.add_argument("--replay-samples",type=int,default=0,help="CPU samples per after file; default0 gives immediate metric/source audit")
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    root=a.root.resolve(); repo=root/"repos"/a.method
    run=(a.run or root/"runs"/f"{a.method.lower()}-fbckge-s3407/native/seed3407").resolve()
    report={"method":a.method,"root":str(root),"run":str(run),"checks":0,"failures":[],"missing":[],
        "audit_scope":"Native FB_CKGE only; not a frozen-clean edit split or five-seed paper reproduction.",
        "read_only":True,"script_sha256":sha(__file__),"torch":torch.__version__,"numpy":np.__version__}
    start=time.monotonic()
    try:
        old,new=source_audit(repo,a.method,report)
        launch=json.loads((run/"launch.json").read_text())
        check(report,"launch_pinned_source",launch.get("upstream_commit")==COMMITS[a.method])
        report["launch_sha256"]=sha(run/"launch.json")
        check(report,"native_launch_seed_dataset",all(
            launch["command"][launch["command"].index(k)+1]==v for k,v in [
                ("-random_seed","3407"),("-dataset","FB_CKGE"),("-epoch_num","200"),
                ("-snapshot_num","5"),("-emb_dim","200"),("-batch_size","1024")]))
        snaps=rebuild(repo/"data/FB_CKGE",report)
        equivalence(old,new,report)
        found=rank_audit(run,snaps,report)
        states=logs_and_selection(repo,run,snaps,found,report)
        output_lineage(repo,a.method,report)
        for i in range(1,5):
            if i in states:
                feature_order(repo,a.method,snaps,i,report)
                for j,rank in enumerate(report["features"][i]["adaptive_ranks"]):
                    check(report,f"adaptive_rank_from_features_{i}_{j}",
                          states[i][f"lora_ent_embeddings_list.{j}.lora_A"].shape[0]==rank)
        checkpoint_replay(repo,a.method,snaps,found,states,report,a.replay_samples)
        pdf=repo/"reproduction/sources/paper.pdf"
        if pdf.exists():
            report["paper"]={"pinned_pdf_sha256":sha(pdf),
             "primary_url":"https://www.ijcai.org/proceedings/2024/0243.pdf","table3_page":6,
             "FB_CKGE":{"mrr":0.223,"hits1":0.131,"hits3":0.257,"hits10":0.405,"training_seconds":405},
             "aggregation":"Final trained model over all seen test sets, five random runs; RTX3090Ti; patience3.",
             "boundary":"One native seed is not the published five-run mean; time includes author train-loop validation."}
    except Exception as e:
        check(report,"audit_exception",False,repr(e))
    report["elapsed_cpu_seconds"]=time.monotonic()-start
    report["status"]="FAIL" if report["failures"] else ("INCOMPLETE" if report["missing"] else
        ("PASS_WITHOUT_CHECKPOINT_REPLAY" if a.replay_samples==0 else "PASS_SAMPLED_CHECKPOINT_REPLAY"))
    print(json.dumps(report,indent=2,allow_nan=False))
    return 1 if report["failures"] else (2 if report["missing"] else 0)
if __name__=="__main__": raise SystemExit(main())
