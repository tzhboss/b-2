"""Freeze bounded TRAIN-only diagnosis after complete negative038 evidence."""
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from diagnose_rank_conflicts39 import controls, sha


root = Path(__file__).resolve().parents[1]
config = 'configs/rank-conflict39.json'
cfg = json.loads((root/config).read_text())
path = root/'experiments'/cfg['id']/'experiment.yaml'
assert not path.exists()
pins = {name: sha(root/name) for name in (config, 'scripts/diagnose_rank_conflicts39.py',
    'scripts/register_rank_conflicts39.py', 'plan/rank-conflict-diagnostic39-protocol.md',
    'plan/review/capacity-complete-disposition.md', 'src/common_incremental.py',
    'src/common_training.py', 'src/shared_rank_graph.py', 'src/shared_rank_scope.py',
    'src/shared_rank_cache_v2.py', 'src/shared_rank_solver_v2.py',
    'configs/rank-capacity37.json')}
for experiment in ('KGE-ENTITY-COMMON-RANK-CAPACITY-037', 'KGE-ENTITY-COMMON-RANK-CAPACITY-UTILITY-038'):
    previous = yaml.safe_load((root/'experiments'/experiment/'experiment.yaml').read_text())
    assert not previous['active_processes'] and 'terminal' in previous['status']
    assert all(v['status'] == 'completed_audited' for v in previous['jobs'].values())
    for name, digest in previous['frozen_input_sha256'].items():
        assert sha(root/name) == digest, name
    summary = previous['result_directory']+'/summary.json'
    pins[summary] = sha(root/summary)
    if experiment.endswith('038'):
        assert previous['aggregation_exit_code'] == 0
        utility = json.loads((root/summary).read_text())
        assert not utility['unavailable']
        assert all(r['decision'] == 'NO_GO_FOR_THIS_FIXED_COMBINATION' for r in utility['rows'])
        report = previous['result_directory']+'/cost-disclosure.json'
        disclosure = json.loads((root/report).read_text())
        assert not disclosure['registered_gates_changed']
        pins[report] = sha(root/report)
control = controls()
path.parent.mkdir(exist_ok=False)
for directory in ('log_directory', 'output_directory'):
    (root/cfg[directory]).mkdir(parents=True, exist_ok=False)
state = dict(id=cfg['id'], status='registered', config=config,
             protocol='plan/rank-conflict-diagnostic39-protocol.md',scope=cfg['scope'],
             registered_utc=datetime.now(timezone.utc).isoformat(), frozen_input_sha256=pins,
             external_source_sha256={}, controls=control, active_processes=[], jobs={},
             scientific_utility_asserted=False, weights_updated=False)
path.write_text(yaml.safe_dump(state, sort_keys=False))
print('RANK_CONFLICT39_REGISTERED', len(pins), control, flush=True)
