"""Freeze repaired TRAIN-only preflight without rewriting original040."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
import yaml
from control_feasible_train41 import controls
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from common_training import sha
cfg=json.loads((root/'configs/feasible-train40r2.json').read_text());can=root/'experiments'/cfg['id']/'experiment.yaml';assert not can.exists()
prior=yaml.safe_load((root/'experiments/KGE-ENTITY-COMMON-FEASIBLE-TRAIN-PREFLIGHT-040/experiment.yaml').read_text())
assert prior['status']=='completed_audited_train_preflight_only' and not prior['active_processes']
for name,digest in prior['frozen_input_sha256'].items():assert sha(root/name)==digest,name
control=controls()
pins={name:sha(root/name) for name in ('configs/feasible-train40r2.json','src/feasible_train_optimization.py','src/feasible_train_optimization_v2.py','scripts/control_feasible_train41.py','scripts/run_feasible_train40.py','scripts/run_feasible_train_v2.py',
       'scripts/coordinate_feasible_experiment.py','scripts/register_feasible_train40r2.py','plan/feasible-train40r2-protocol.md','src/common_incremental.py','src/common_training.py','src/shared_rank_scope.py','src/shared_rank_graph.py',
       'src/shared_rank_cache_v2.py','src/shared_rank_solver_v2.py',prior['result_directory']+'/summary.json')}
can.parent.mkdir(exist_ok=False)
for name in ('output_directory','log_directory'):(root/cfg[name]).mkdir(parents=True,exist_ok=False)
state=dict(id=cfg['id'],status='registered',config='configs/feasible-train40r2.json',protocol='plan/feasible-train40r2-protocol.md',registered_utc=datetime.now(timezone.utc).isoformat(),frozen_input_sha256=pins,controls=control,active_processes=[],jobs={},scientific_utility_asserted=False)
can.write_text(yaml.safe_dump(state,sort_keys=False));print('REGISTERED40R2',len(pins),flush=True)
