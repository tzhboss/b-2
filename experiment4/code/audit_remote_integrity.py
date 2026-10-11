"""Verify all remote exp4 KGE checkpoint hashes and identical input data."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={}
for kind in ("DistMult","ComplEx"):
 for seed in (401,402,403):
  j=json.loads((ROOT/"checkpoints"/f"{kind.lower()}_seed{seed}_training.json").read_text())
  assert j["seed"]==seed and j["kind"]==kind and len(j["epochs"])==25
  assert j["train_only"] and not j["TEST_touched"]
  for name,digest in j["dataset_sha256"].items():
   actual=sha(ROOT/"data"/name)
   assert actual==digest,(kind,seed,name)
   sources[name]=digest
  for rec in j["epochs"]:
   if "checkpoint_file" in rec:
    f=ROOT/"checkpoints"/rec["checkpoint_file"]
    assert f.exists() and f.stat().st_size>1000000
    assert sha(f)==rec["checkpoint_sha256"],f
  for epoch in (5,25):
   cp=ROOT/"checkpoints"/f"{kind.lower()}_seed{seed}_epoch{epoch}.pt"
   for suffix in ("edit.json","edit_v2.json") if epoch==25 else ("edit.json",):
    v=json.loads((ROOT/"results"/f"{kind.lower()}_seed{seed}_ep{epoch}_{suffix}").read_text())
    assert v["checkpoint_sha256"]==sha(cp)
    assert v["n_requests"]==27 and len(v["rows"])==27
    assert not v["TEST_used"],suffix
assert not (ROOT/"data"/"test.txt").exists()
out={"verified":"remote_experiment4_phase24",
     "training_model_count":6,"training_epochs_per_model":25,
     "checkpoint_intermediate_milestones":[5,10,15,20,25],
     "independent_seeds":[401,402,403],
     "checkpoint_hashes_verified":6*5,
     "edit_json_verified":6*3,
     "all_case_counts_equal_27":True,
     "all_test_absent":True,"source_sha256":sources}
(RootResult:=ROOT/"results"/"phase24_integrity.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print("INTEGRITY_PASS",json.dumps(out,ensure_ascii=False))
