from pathlib import Path
import json,yaml,subprocess,re,datetime
R=Path(__file__).resolve().parents[1]
result={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"root":str(R),"experiments":{}}
for p in sorted((R/"experiments").glob("KGE-LIT-*/experiment.yaml")):
 c=yaml.safe_load(p.read_text());rec={"status":c.get("status"),"audit":c.get("audit"),"jobs":[],"active":c.get("active_processes",[])}
 for j in c.get("jobs",[]):
  v={k:j[k] for k in ["id","status","stage","gpu","exit_code"] if k in j};run=R/"runs"/j["id"];log=run/"run.log"
  if log.exists():
   with log.open("rb") as f:
    f.seek(max(0,log.stat().st_size-100000));tail=f.read().decode(errors="replace")
   epochs=re.findall(r"Epoch (\d+):.*? (\d+)/([\d]+)",tail)
   incde=re.findall(r"Snapshot:(\d+)\s+Epoch:(\d+).*?MRR:([\d.]+)",tail)
   if epochs:v["latest_epoch_progress"]=list(map(int,epochs[-1]))
   elif incde:v["latest_snapshot_epoch_validation_mrr_percent"]=incde[-1]
  ap=R/"results/KGE-LIT-001-audit"/j["id"]/"audit.json"
  if ap.exists():
   a=json.loads(ap.read_text());v["incde_audit"]={"status":a["status"],"checks":a["checks"],"failures":len(a["failures"]),"replayed_queries":a["independent_post_train_rank_replays"],"native_final":a["native_overall"][-1]}
  dp=R/"runs/candidate-diagnostic"/j["id"]/"terminal.json"
  if dp.exists():
   d=json.loads(dp.read_text());v["candidate_diagnostic"]={"status":d["status"],"queries":d["queries"],"all_query_native_replay_exact":d["all_query_native_replay_exact"],"overall":d["overall"]}
  rec["jobs"].append(v)
 result["experiments"][c["id"]]=rec
pids=sorted({x["pid"] for c in result["experiments"].values() for x in c["active"] if "pid" in x})
if pids:
 q=subprocess.run(["ps","-p",",".join(map(str,pids)),"-o","pid,stat,etime,pcpu,rss,comm"],capture_output=True,text=True);result["owned_process_resources"]=q.stdout.strip()
mem={k:int(v.split()[0]) for line in Path("/proc/meminfo").read_text().splitlines() for k,v in [line.split(":",1)] if k in ["MemTotal","MemAvailable"]};result["memory_kib"]=mem
print(json.dumps(result,indent=2))
