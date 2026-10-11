#!/usr/bin/env python3
"""Read-only independent CPU audit of the frozen SAGE FACT seed55 run.
JSON stdout only. Exit1 retains original producer-integrity failure even when
scientific artifact checks pass; exit2 incomplete, exit0 no failure.
No native modules, CUDA APIs, training, file writes, or env mutation are used.
"""
import argparse, ast, difflib, hashlib, json, re, subprocess, sys, time, types, warnings
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MASTER_SHA="3d31b5a36b28aff0aa155d8fb1c26bef3b4831a22c53091a857d9c79c51a6165"
COMMIT="d40a6031c8a1eb8dae15454b9b4da7ac2c41c7f9"
DIMS=[180,210,250,270,290]
KEYS=["mrr","mr","hits1","hits3","hits5","hits10"]
CONTRACT="author double-argsort; historical query-snapshot candidates/filter"
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
def jsondiff(a,b,path=""):
    if isinstance(a,dict) and isinstance(b,dict):
        out=[]
        for k in sorted(set(a)|set(b)):
            q=path+"/"+str(k)
            if k not in a:out.append({"path":q,"change":"added","after":b[k]})
            elif k not in b:out.append({"path":q,"change":"removed","before":a[k]})
            else:out+=jsondiff(a[k],b[k],q)
        return out
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in jsondiff(x,y,path+"/"+str(i))]
    return [] if a==b else [{"path":path,"change":"changed","before":a,"after":b}]
def bind(root,run,r):
    master=root/"configs/experiments/KGE-LIT-003-jobs.json"
    check(r,"frozen_master_sha256",sha(master)==MASTER_SHA)
    jobs=json.loads(master.read_text())
    job=next(j for j in jobs if j["id"]=="sage-fact-s55")
    registered=json.loads((run/"registered-job.json").read_text())
    check(r,"registered_job_equals_master",registered==job)
    check(r,"registered_protocol_identity",job["method"]=="SAGE" and job["dataset"]=="FACT" and
          job["seed"]==55 and job["source_commit"]==COMMIT and job["gpu"]==6)
    check(r,"completion_contract",job["completion_contract"]==
          {"snapshots":5,"best_checkpoints":5,"test_rank_files":29,"terminal_jsons":2})
    work=run/"worktree"; repo=root/"repos/SAGE"
    check(r,"upstream_commit",git(repo,"rev-parse","HEAD").strip()==COMMIT)
    expected_copies={str(Path("worktree")/Path(k).relative_to("repos/SAGE")):v
                     for k,v in job["input_sha256"].items() if k.startswith("repos/SAGE/")}
    copied=json.loads((run/"copied-source-sha256.json").read_text())
    check(r,"copied_inventory_equals_master",copied==expected_copies)
    changed=[]; frozen_changes=[]
    for k,expected in job["input_sha256"].items():
        original=root/k; current=sha(original) if original.is_file() else None
        clone=(work/Path(k).relative_to("repos/SAGE")) if k.startswith("repos/SAGE/") else original
        frozen=sha(clone) if clone.is_file() else None
        check(r,"frozen_input_"+k,frozen==expected,
              None if frozen==expected else {"expected":expected,"actual":frozen})
        if frozen!=expected:frozen_changes.append(k)
        if current!=expected:changed.append({"path":k,"registered_sha256":expected,
                                             "current_sha256":current,"frozen_copy_sha256":frozen})
    runtime=json.loads((run/"runtime-command.json").read_text())
    expected_command=list(job["command"])
    expected_command[expected_command.index("-data_path")+1]=str(work/"data")+"/"
    check(r,"runtime_command_exact_frozen_remap",runtime["command"]==expected_command)
    check(r,"runtime_worktree_and_gpu",Path(runtime["cwd"]).resolve()==work.resolve() and runtime["gpu"]==6)
    r["binding"]={"master_sha256":sha(master),"registered_sha256":sha(run/"registered-job.json"),
       "copied_inventory_sha256":sha(run/"copied-source-sha256.json"),"runtime_sha256":sha(run/"runtime-command.json"),
       "master_entries":len(job["input_sha256"]),"worktree_changes":frozen_changes,"current_repository_changes":changed,
       "environment_contract":job["environment"],
       "environment_limit":"Frozen environment configuration/dependency binaries are bound; runtime-command receipt does not capture the complete effective process environment."}
    # Retain original producer receipt. Never turn this into a clean integrity pass.
    terminal=run/"terminal.json"
    if terminal.exists():
        receipt=json.loads(terminal.read_text());r["original_producer_receipt"]={"sha256":sha(terminal),"receipt":receipt}
        check(r,"producer_exit0",receipt.get("exit_code")==0)
        check(r,"producer_run_log_binding",receipt.get("log_sha256")==sha(run/"run.log"))
        r["producer_integrity_failed"]=receipt.get("source_inputs_unchanged") is not True or receipt.get("status")!="completed"
    else:
        missing(r,"terminal.json");r["producer_integrity_failed"]=bool(changed)
    r["provenance_adjudication"]={"classification":"NO_SOURCE_DRIFT" if not changed else "UNADJUDICATED_SOURCE_DRIFT"}
    if changed:
        only=[x["path"] for x in changed]==["repos/SAGE/provenance/preparation_manifest.json"]
        check(r,"source_drift_restricted_to_manifest",only,changed)
        if only:
            before=json.loads((work/"provenance/preparation_manifest.json").read_text())
            after=json.loads((repo/"provenance/preparation_manifest.json").read_text())
            diff=jsondiff(before,after)
            permitted={"/defaults/-rank_export_dir","/defaults/-result_export_dir",
                       "/source_sha256/main.py","/source_sha256/src/parse_args.py",
                       "/source_sha256/src/model/model_process.py",
                       "/primary_sources/0/resolved_url","/primary_sources/0/content_type",
                       "/primary_sources/1/resolved_url","/primary_sources/1/content_type"}
            check(r,"metadata_diff_only_observation_bookkeeping",set(x["path"] for x in diff)<=permitted,diff)
            for f in ["main.py","src/parse_args.py","src/model/model_process.py"]:
                check(r,"updated_manifest_matches_frozen_actual_"+f,after["source_sha256"][f]==sha(work/f))
                check(r,"old_manifest_preobservation_hash_"+f,before["source_sha256"][f]==
                      hashlib.sha256(git(repo,"show","HEAD:"+f).encode()).hexdigest())
            consumed=[work/"main.py",*sorted((work/"src").rglob("*.py"))]
            refs=[str(p.relative_to(work)) for p in consumed if "preparation_manifest" in p.read_text()]
            check(r,"native_source_has_no_preparation_manifest_read",not refs,refs)
            r["provenance_adjudication"]={"classification":"SUPPORTED_NONCONSUMED_METADATA_DRIFT",
                "diff":diff,"native_reference_hits":refs,
                "basis":"All master inputs bind:300 SAGE frozen copies and2 external dependency wheels; only original preparation manifest drifted. Its diff updates three observation-file hash descriptions, two export defaults, and removes response metadata. Native startup/model/data code has no read/reference to that manifest.",
                "boundary":"Static source and runtime-path evidence, not OS file-access tracing. Original failed producer receipt remains failed; controller owns any formal adjudication/promotion."}
    return job,work,repo
class StripExport(ast.NodeTransformer):
    def visit_Assign(self,n):
        if any(isinstance(x,ast.Name) and x.id=="export_rows" for x in n.targets):return None
        return self.generic_visit(n)
    def visit_If(self,n):
        if "export_rows" in ast.unparse(n.test):return None
        return self.generic_visit(n)
def source_audit(work,repo,r):
    changed=[]
    for f in git(repo,"ls-files","*.py").splitlines():
        original=git(repo,"show","HEAD:"+f);actual=(work/f).read_text()
        a,b=ast.parse(original),ast.parse(actual)
        if original!=actual:changed.append(f)
        if f=="src/model/model_process.py":
            b=StripExport().visit(b)
            b.body=[x for x in b.body if not(isinstance(x,ast.FunctionDef) and x.name=="_save_observed_query_ranks")]
        elif f=="src/parse_args.py":
            b.body=[x for x in b.body if not(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call)
                and any(isinstance(v,ast.Constant) and v.value in ["-rank_export_dir","-result_export_dir"] for v in x.value.args))]
        elif f=="main.py":
            class RestorePath(ast.NodeTransformer):
                def visit_Attribute(self,n):
                    if n.attr=="result_export_dir":return ast.Constant(value="res_diff_datasets/")
                    return self.generic_visit(n)
            b=RestorePath().visit(b)
        check(r,"original_method_ast_"+f,ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False))
    check(r,"only_observation_files_changed",set(changed)=={"main.py","src/parse_args.py","src/model/model_process.py"},changed)
    patch=work/"provenance/observation.patch"
    # This preserved artifact was produced with Python difflib, not git diff.
    expected_patch="".join("".join(difflib.unified_diff(
        (work/"provenance/original_source"/f).read_text().splitlines(keepends=True),
        (work/f).read_text().splitlines(keepends=True),fromfile="a/"+f,tofile="b/"+f))
        for f in ["src/parse_args.py","src/model/model_process.py","main.py"])
    check(r,"preserved_observation_diff",patch.read_text()==expected_patch)
    for f in ["main.py","src/parse_args.py","src/model/model_process.py"]:
        check(r,"preserved_original_"+f,(work/"provenance/original_source"/f).read_text()==git(repo,"show","HEAD:"+f))
    # Scientific source proof: LKGE/TransE inherit BaseModel's raw embedding scorer.
    tree=ast.parse((work/"src/model/LKGE.py").read_text())
    overrides=[]
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef) and cls.name in ["LKGE","TransE"]:
            overrides += [cls.name+"."+x.name for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ["embedding","predict","norm_ent","norm_rel","snapshot_post_processing"]]
    check(r,"raw_embedding_predict_inheritance",not overrides,overrides)
    r["source"]={"commit":COMMIT,"observation_patch_sha256":sha(patch),"changed_files":changed,
      "inference_basis":"LKGE/TransE have no embedding/predict/norm/post-processing override; BaseModel.embedding returns raw weights, normal predict scores them directly. Mapping is trained after post_train evaluation."}
    return git(repo,"show","HEAD:src/model/model_process.py"),(work/"src/model/model_process.py").read_text()
def runtime_defaults(job,work,run,log,r):
    defaults={}
    tree=ast.parse((work/"src/parse_args.py").read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="add_argument":
            flag=ast.literal_eval(n.args[0]);kw={x.arg:x.value for x in n.keywords}
            dest=ast.literal_eval(kw["dest"]) if "dest" in kw else flag.lstrip("-")
            defaults[dest]=ast.literal_eval(kw["default"]) if "default" in kw else None
    for k in ["dataset","seed","epoch_num","snapshot_num","emb_dim","emb_dim2","emb_dim3","emb_dim4","emb_dim5","patience",
              "batch_size","learning_rate","neg_ratio","new_dim_epoch_num","num_samples_for_retrain","lifelong_name",
              "using_finetune","using_embedding_transfer"]:
        expected={"dataset":"FACT","seed":55,"epoch_num":200,"snapshot_num":5,"emb_dim":180,"emb_dim2":210,
          "emb_dim3":250,"emb_dim4":270,"emb_dim5":290,"patience":3,"batch_size":2048,"learning_rate":.0001,
          "neg_ratio":10,"new_dim_epoch_num":1,"num_samples_for_retrain":1000,"lifelong_name":"LKGE",
          "using_finetune":"True","using_embedding_transfer":"False"}[k]
        check(r,"frozen_default_"+k,defaults[k]==expected)
    first=log.splitlines()[0]
    # Parse primitive Namespace assignments only, never eval logged code.
    names={}
    for key in defaults:
        m=re.search(r"(?:Namespace\(|, )"+re.escape(key)+r"=('[^']*'|\"[^\"]*\"|True|False|None|[-+.\deE]+)(?=,|\))",first)
        if m:
            try:names[key]=ast.literal_eval(m[1])
            except (SyntaxError,ValueError):pass
    effective=defaults.copy()
    cmd=json.loads((run/"runtime-command.json").read_text())["command"]
    for flag in ["-gpu","-data_path","-save_path","-log_path","-rank_export_dir","-result_export_dir"]:
        effective[flag[1:]]=cmd[cmd.index(flag)+1]
    effective["reconstruct_weight"]=1.0 # FACT LKGE config, independently inspect function.
    for k,v in effective.items():
        if k in ["save_path","data_path","log_path"]:continue
        check(r,"effective_logged_"+k,names.get(k)==v,{"expected":v,"logged":names.get(k)})
    check(r,"effective_data_path",names.get("data_path")==str(work/"data/FACT")+"/")
    check(r,"effective_checkpoint_path",names.get("save_path")==str(run/"checkpoint/FACT-TransE-LKGE-Margin-WO_transfer"))
    r["effective_configuration"]={k:v for k,v in names.items() if k not in ["data_path","save_path","log_path","result_export_dir","rank_export_dir"]}
    r["selection_limit"]="Validation MRR only checked through four-decimal native metric represented as percent with two decimals in logs and selected epoch/checkpoint bytes; validation ranks/scores not independently exported or recomputed."
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
def ranks(run,snaps,r):
    expected={(i,j,"post_train") for i in range(5) for j in range(i+1)}|{(i,j,"post_expand") for i in range(4) for j in range(i+1)}|{(i,i,"pre_fwt") for i in range(1,5)}
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
        check(r,name+"_mrr_hits_float32_vs_float64_round",all(round(raw[k],4)==rounded[k] for k in ["mrr","hits1","hits3","hits5","hits10"]))
        out[i,j,phase]={"rank":x,"raw":raw,"rounded":rounded}
        r["rank_files"][name]={"sha256":sha(p),"queries":n,"raw_float64":raw,"native_float32_batch100_round4":rounded,
          "raw_minus_native_rounded":delta,"filter_sha256":snaps[j]["filter_sha256"]}
    r["rank_files_found"]=len(out);r["rank_files_expected"]=29
    return out
def overall(out,snaps,i,phase,raw=False):
    w=[s["test_count"] for s in snaps[:i+1]]
    return {"whole_"+k:(sum(out[i,j,phase]["raw" if raw else "rounded"][k]*w[j] for j in range(i+1))/sum(w)
                         if raw else round(sum(out[i,j,phase]["rounded"][k]*w[j] for j in range(i+1))/sum(w),3))
            for k in ["mrr","hits1","hits3","hits10"]}
def results_and_logs(run,snaps,out,log,r):
    paths={"post_train":run/"json/FACT-num1000-easy-FACT.json","post_expand":run/"json/AGAIN-FACT-num1000-easy-"}
    check(r,"no_extra_result_jsons",{x.name for x in (run/"json").iterdir() if x.is_file()}<= {p.name for p in paths.values()})
    docs={};r["terminal_results"]={}
    for phase,path in paths.items():
        if not path.exists():missing(r,path.name);continue
        data=json.loads(path.read_text());docs[phase]=data
        length=6 if phase=="post_train" else 4
        check(r,phase+"_terminal_json_length",len(data)==length)
        r["terminal_results"][phase]={"path":str(path.relative_to(run)),"sha256":sha(path),"overall":{}}
        for i in range(5 if phase=="post_train" else 4):
            check(r,phase+f"_json_task_identity_{i}",data[i]["trained_ssid"]==i and len(data[i]["results"])==i+1)
            if not all((i,j,phase) in out for j in range(i+1)):continue
            for j in range(i+1):
                expected=dict(out[i,j,phase]["rounded"],test_ssid=j)
                check(r,phase+f"_json_metric_{i}_{j}",data[i]["results"][j]==expected,
                      None if data[i]["results"][j]==expected else {"expected":expected,"actual":data[i]["results"][j]})
            agg=overall(out,snaps,i,phase)
            check(r,phase+f"_json_overall_{i}",data[i]["aggregated_result"]==agg)
            r["terminal_results"][phase]["overall"][i]={"native_round3":agg,"raw_float64":overall(out,snaps,i,phase,True)}
    # Parse actual tables before their duplicated Final Result list.
    body=log.split("Final Result:")[0];tables={}
    for match in re.finditer(r"\|\s*Snapshot:(\d+)\s*\|",body):
        i=int(match[1]);block=body[match.end():]
        end=re.search(r"\n\d{4}-\d\d-\d\d ",block)
        if end:block=block[:end.start()]
        table={}
        for line in block.splitlines():
            v=[x.strip() for x in line.strip().strip("|").split("|")]
            if len(v)==6 and v[0].isdigit():table[int(v[0])]=dict(zip(["mrr","hits1","hits3","hits5","hits10"],map(float,v[1:])))
        tables[i]=table
    for i in range(5):
        if i not in tables:missing(r,f"native_log_task_table_{i}");continue
        check(r,f"native_log_task_table_coverage_{i}",set(tables[i])==set(range(i+1)))
        for j in range(i+1):
            if (i,j,"post_train") in out:
                check(r,f"native_log_task_{i}_{j}",tables[i].get(j)=={k:out[i,j,"post_train"]["rounded"][k] for k in ["mrr","hits1","hits3","hits5","hits10"]})
    final=log.split("Report Result:")[-1] if "Report Result:" in log else "";times=[];whole={}
    for line in final.splitlines():
        v=[x.strip() for x in line.strip().strip("|").split("|")]
        if len(v)==6 and v[0].isdigit():
            i=int(v[0]);times.append(float(v[1]));whole[i]=dict(zip(["whole_mrr","whole_hits1","whole_hits3","whole_hits10"],map(float,v[2:])))
    for i in range(5):
        if i not in whole:missing(r,f"native_log_overall_{i}");continue
        if all((i,j,"post_train") in out for j in range(i+1)):check(r,f"native_log_overall_{i}",whole[i]==overall(out,snaps,i,"post_train"))
    timing=re.search(r"Sum_Training_Time:([-+.\deE]+)",log)
    if timing:
        r["native_train_loop_seconds"]=float(timing[1]);check(r,"native_training_time_sum",len(times)==5 and abs(sum(times)-float(timing[1]))<1e-6)
    else:missing(r,"native_terminal_training_time")
    if all((i,i,"pre_fwt") in out for i in range(1,5)) and all((4,j,"post_train") in out for j in range(5)):
        fwt=sum(out[i,i,"pre_fwt"]["rounded"]["mrr"] for i in range(1,5))/4
        bwt=sum(out[4,j,"post_train"]["rounded"]["mrr"]-out[j,j,"post_train"]["rounded"]["mrr"] for j in range(4))/4
        r["transfer"]={"FWT":fwt,"BWT":bwt}
        if "post_train" in docs:check(r,"terminal_json_transfer",docs["post_train"][-1]==r["transfer"])
        match=re.search(r"Forward transfer:\s*([-+.\deE]+)\s+Backward transfer:\s*([-+.\deE]+)",log)
        if match:check(r,"native_log_transfer",abs(float(match[1])-fwt)<1e-12 and abs(float(match[2])-bwt)<1e-12)
        else:missing(r,"native_log_transfer")
    return tables
def checkpoints(run,snaps,log,tables,r):
    states={};r["best_checkpoints"]=[]
    directory=run/"checkpoint/FACT-TransE-LKGE-Margin-WO_transfer"
    for i in range(5):
        p=directory/f"{i}model_best.tar"
        if not p.exists() or i not in tables:missing(r,f"final_checkpoint_{i}");continue
        data=torch.load(p,map_location="cpu",weights_only=True);epoch=data["epoch_id"];state=data["state_dict"]
        check(r,f"best_checkpoint_epoch_{i}",isinstance(epoch,int) and 0<=epoch<200)
        selected=directory/f"{i}checkpoint-{epoch}.tar"
        check(r,f"best_equals_epoch_file_{i}",selected.exists() and sha(selected)==sha(p))
        check(r,f"checkpoint_finite_{i}",all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in state.values()))
        e,t=state["ent_embeddings.weight"],state["rel_embeddings.weight"]
        check(r,f"checkpoint_dims_{i}",tuple(e.shape)==(snaps[i]["num_ent"],DIMS[i]) and tuple(t.shape)==(snaps[i]["num_rel"],DIMS[i]) and e.dtype==t.dtype==torch.float64)
        if i>0:check(r,f"mapping_shape_{i}",tuple(state["mapping.weight"].shape)==(DIMS[i]-DIMS[i-1],DIMS[i-1]))
        records=[]
        for line in log.split("Final Result:")[0].splitlines():
            if re.search(rf"Snapshot:{i}\s",line) and "\tEpoch:" in line:
                m=re.search(r"Epoch:(\d+).*?\tMRR:([-+.\deE]+)",line)
                if m:records.append((int(m[1]),float(m[2])/100))
        if records:
            best=max(x for ep,x in records);first=next(ep for ep,x in records if x==best)
            check(r,f"first_strict_validation_max_{i}",epoch==first,{"selected":epoch,"first_log_max":first,"validation_mrr":best})
        else:check(r,f"checkpoint_validation_log_{i}",False);best=None
        stops=re.findall(rf"Early Stopping! Snapshot:{i} Epoch: (\d+)",log)
        check(r,f"native_full_training_completion_{i}",bool(stops) or len(records)==200)
        r["best_checkpoints"].append({"snapshot":i,"epoch":epoch,"validation_mrr_from_log":best,"dim":DIMS[i],"sha256":sha(p),
           "tensors":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in state.items()}})
        states[i]=state
    return states
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
        exec(compile(ast.Module(body=[node],type_ignores=[]),"<sage-evaluator>","exec"),env)
        classes.append(env["DevBatchProcessor"])
    for method,phase in [("process_epoch","post_train"),("process_epoch","pre_fwt"),("process_epoch_again","post_expand")]:
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
    r["observation_equivalence"]={"queries":7,"batches":[4,3],"exact_ties":True,"phases":["post_train","pre_fwt","post_expand"],"filesystem_writes":0}
def replay(work,snaps,out,states,r,n):
    if n<=0:r["sampled_replay"]={"status":"DISABLED"};return
    tree=ast.parse((work/"src/model/BaseModel.py").read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="BaseModel")
    methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ["embedding","predict","norm_ent","norm_rel"]]
    node=ast.ClassDef(name="ScoringOnly",bases=[],keywords=[],body=methods,decorator_list=[])
    node=ast.fix_missing_locations(node)
    env={"torch":torch,"nn":nn}
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<sage-native-scoring-only>","exec"),env)
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
            check(r,name+"_checkpoint_rank_replay",all(compat),{"indices":idx.tolist(),"export":d["rank"][idx].tolist(),"cpu":predranks})
            records[name]={"indices":idx.tolist(),"export":d["rank"][idx].tolist(),"cpu":predranks,"samples":len(idx),
                           "exact_matches":sum(exact),"tie_compatible":sum(compat),"native_score_max_abs_diff":delta}
    r["sampled_replay"]={"device":"cpu","files":records,"files_replayed":len(records),"samples":sum(v["samples"] for v in records.values()),
        "exact_matches":sum(v["exact_matches"] for v in records.values()),"scope":"15 selected post_train raw embedding checkpoints. Mapping/training buffers not needed by proven native scorer.",
        "exclusions":"No post_expand/pre_fwt exact replay: no corresponding mapping evaluation state/checkpoint association proved.",
        "ties":"Exact CPU ranks and GPU export rank membership in exact-score tie intervals separately reported."}
    if len(records)<15:missing(r,"15_post_train_checkpoint_replays")
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--run",type=Path)
    parser.add_argument("--replay-samples",type=int,default=0)
    parser.add_argument("--threads",type=int,default=2)
    a=parser.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    root=a.root.resolve();run=(a.run or root/"runs/sage-fact-s55").resolve()
    r={"method":"SAGE","root":str(root),"run":str(run),"checks":0,"failures":[],"missing":[],
       "script_sha256":sha(__file__),"read_only":True,"torch":torch.__version__,"numpy":np.__version__,
       "scope":"Frozen released FACT seed55 native execution; no clean edit split, no five-seed paper mean, no independent retraining."}
    start=time.monotonic()
    try:
        job,work,repo=bind(root,run,r)
        logs=list((run/"logs").glob("**/*.log"))
        check(r,"single_native_log",len(logs)==1)
        log=logs[0].read_text()
        r["log_sha256"]=sha(logs[0]);r["run_log_sha256"]=sha(run/"run.log")
        old,new=source_audit(work,repo,r)
        runtime_defaults(job,work,run,log,r)
        snaps=rebuild(work/"data/FACT",r)
        equivalence(old,new,r)
        out=ranks(run,snaps,r)
        tables=results_and_logs(run,snaps,out,log,r)
        states=checkpoints(run,snaps,log,tables,r)
        replay(work,snaps,out,states,r,a.replay_samples)
        r["limitations"]=[
          "Producer source integrity failure retained; any metadata-only adjudication/promotion belongs to controller.",
          "15 post_train files may have sampled checkpoint inference;10 post_expand and4 pre_fwt only query/filter/count/metric audits.",
          "No independent validation rank recomputation; selection is logged metric/epoch/byte-lineage consistency.",
          "Fixed dimension schedule180/210/250/270/290 and replay normalization follow released implementation; paper algorithm equivalence is not established.",
          "Native candidate/filter universe is historical query snapshot; not frozen clean edit protocol.",
          "One seed55 run is not the five-seed paper mean.",
          "Native training-loop sum excludes training-data selection, mapping training and full evaluation/launch overhead."]
    except Exception as exc:check(r,"audit_exception",False,repr(exc))
    r["elapsed_cpu_seconds"]=time.monotonic()-start
    r["scientific_artifact_status"]="FAIL" if r["failures"] else ("INCOMPLETE" if r["missing"] else
        ("PASS_SAMPLED_POST_TRAIN_REPLAY" if a.replay_samples>0 else "PASS_WITHOUT_CHECKPOINT_REPLAY"))
    r["status"]="FAIL" if r["failures"] else ("INCOMPLETE" if r["missing"] else
        ("PRODUCER_INTEGRITY_FAIL_SCIENTIFIC_PASS" if r.get("producer_integrity_failed") else r["scientific_artifact_status"]))
    print(json.dumps(r,indent=2,allow_nan=False))
    return 1 if r["failures"] or r.get("producer_integrity_failed") else (2 if r["missing"] else 0)
if __name__=="__main__":raise SystemExit(main())
