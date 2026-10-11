"""Require both complete preflights and register unchanged utility gates."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import yaml
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from common_training import sha
from common_incremental import Data
from control_feasible_train41 import controls

cfg=json.loads((root/'configs/feasible-train41.json').read_text());can=root/'experiments'/cfg['id']/'experiment.yaml';assert not can.exists()
pins={}
for experiment in ('KGE-ENTITY-COMMON-FEASIBLE-TRAIN-PREFLIGHT-040','KGE-ENTITY-COMMON-FEASIBLE-TRAIN-PREFLIGHT-040R2'):
    prior=yaml.safe_load((root/'experiments'/experiment/'experiment.yaml').read_text())
    assert prior['status']=='completed_audited_train_preflight_only' and not prior['active_processes']
    assert all(v['status']=='completed_audited' for v in prior['jobs'].values())
    for name,digest in prior['frozen_input_sha256'].items():assert sha(root/name)==digest,name
    summary=prior['result_directory']+'/summary.json';raw=json.loads((root/summary).read_text());assert raw['completed_audited_jobs']==6
    pins[summary]=sha(root/summary)
control=controls();data=Data(root,cfg);new=data.triples(1,'train').numpy()
entities=np.unique(new[:,[0,2]]);assert np.array_equal(entities[entities>=7505],np.arange(7505,11258))
assert (cfg['utility_new_mrr_ratio'],cfg['utility_old_loss_ratio'],cfg['utility_direction_mrr_ratio'],cfg['utility_direction_hits10_ratio'])==(.98,.75,.98,.95)
names=['configs/feasible-train41.json','plan/feasible-train41-protocol.md','scripts/run_feasible_train41.py','scripts/run_feasible_train40.py','scripts/run_feasible_train_v2.py',
    'scripts/control_feasible_train41.py','scripts/control_feasible_train40.py','scripts/register_feasible_train41.py','scripts/report_feasible_train41.py','scripts/coordinate_feasible_train41.py',
    'scripts/audit_exported_embeddings.py','scripts/audit_shared_graph_utility.py','src/feasible_train_optimization.py','src/feasible_train_optimization_v2.py',
    'src/shared_rank_cache_v2.py','src/shared_rank_solver_v2.py','src/shared_rank_graph.py','src/shared_rank_scope.py','src/common_incremental.py','src/common_training.py',
    'results/KGE-ENTITY-COMMON-SHARED-GRAPH-UTILITY-032/summary.json']
pins.update({name:sha(root/name) for name in names})
parent_sources={}
for job in cfg['jobs']:
    endpoint=root/'artifacts/incremental'/job['initial_endpoint'];audit=json.loads((endpoint/'audit.json').read_text())
    assert not audit['failures'] and not audit['solver_audit']['failures']
    parent_sources.update({str(endpoint/name):sha(endpoint/name) for name in ('best.pt','audit.json','training-receipt.json')})
can.parent.mkdir(exist_ok=False);(root/cfg['log_directory']).mkdir(parents=True,exist_ok=False)
for job in cfg['jobs']:assert not (root/cfg['output_directory']/job['id']).exists()
state=dict(id=cfg['id'],status='registered',config='configs/feasible-train41.json',protocol='plan/feasible-train41-protocol.md',registered_utc=datetime.now(timezone.utc).isoformat(),
     frozen_input_sha256=pins,external_source_sha256=dict(parent_sources,**data.bound),controls=control,train_covered_new_entities=3753,
     active_processes=[],jobs={},scientific_utility_asserted=False,registered_gates_changed=False)
can.write_text(yaml.safe_dump(state,sort_keys=False));print('FULL_FEASIBLE_TRAIN41_REGISTERED',len(pins),len(parent_sources),flush=True)
