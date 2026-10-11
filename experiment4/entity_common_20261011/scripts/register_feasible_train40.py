"""Freeze TRAIN-only optimization controls before accessing new outcomes."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
import yaml
from control_feasible_train40 import controls
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from common_training import sha
config='configs/feasible-train40.json';cfg=json.loads((root/config).read_text())
can=root/'experiments'/cfg['id']/'experiment.yaml';assert not can.exists()
prior=yaml.safe_load((root/'experiments/KGE-ENTITY-COMMON-RANK-CONFLICT-DIAGNOSTIC-039/experiment.yaml').read_text())
assert prior['status']=='completed_audited_train_diagnostic_only' and not prior['active_processes']
for name,digest in prior['frozen_input_sha256'].items():assert sha(root/name)==digest,name
control=controls();pins={name:sha(root/name) for name in (config,'src/feasible_train_optimization.py','scripts/control_feasible_train40.py','scripts/run_feasible_train40.py','scripts/register_feasible_train40.py',
    'plan/feasible-train40-protocol.md','src/shared_rank_cache_v2.py','src/shared_rank_solver_v2.py','src/shared_rank_graph.py','src/shared_rank_scope.py','src/common_incremental.py','src/common_training.py',
    'results/KGE-ENTITY-COMMON-RANK-CONFLICT-DIAGNOSTIC-039/mechanism-disclosure.json','results/KGE-ENTITY-COMMON-SHARED-GRAPH-UTILITY-032/summary.json','results/KGE-ENTITY-COMMON-TASK-UTILITY-020/summary.json')}
can.parent.mkdir(exist_ok=False)
for directory in ('output_directory','log_directory'):(root/cfg[directory]).mkdir(parents=True,exist_ok=False)
state=dict(id=cfg['id'],status='registered',config=config,protocol='plan/feasible-train40-protocol.md',registered_utc=datetime.now(timezone.utc).isoformat(),
           frozen_input_sha256=pins,controls=control,active_processes=[],jobs={},scientific_utility_asserted=False)
can.write_text(yaml.safe_dump(state,sort_keys=False));print('FEASIBLE_TRAIN40_REGISTERED',len(pins),control,flush=True)
