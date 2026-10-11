"""Read-only live ownership/source check after resource queue handoff."""
import argparse
import json
from pathlib import Path

import yaml
from report_rank_capacity import inputs
from resume_capacity_resources_v2 import process, command, digest


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
cfg = json.loads((root/'configs/rank-capacity-report.json').read_text())
context, canonical, checked = inputs(root, cfg)
state = canonical[cfg['experiment']]
metadata = state['resource_handoff']
assert metadata['runtime_queue_sha256'] == digest(root/'scripts/resume_capacity_resources_v2.py')
assert metadata['addendum_sha256'] == digest(root/metadata['addendum'])
assert metadata['controls_sha256'] == digest(root/'results/capacity-resource-handoff-controls-v2.json')
owner = process(state['coordinator_pid'])
assert owner['state'] in ('S', 'R') and 'scripts/resume_capacity_resources_v2.py' in command(owner['pid'])
legacy = process(metadata['legacy_coordinator']['pid'])
assert legacy['state'] == 'T' and legacy['start_ticks'] == metadata['legacy_coordinator']['start_ticks']
original_producers = []
for producer, wrapper in [(783947, 783946), (783948, 783945)]:
    current = process(producer)
    assert current['parent'] == wrapper and current['state'] in ('S', 'R')
    assert '--stage train' in command(producer) and 'scripts/run_shared_graph.py' in command(producer)
    original_producers.append(current)
jobs = []
for row in state['active_processes']:
    current = process(row['pid'])
    assert current['state'] in ('S', 'R')
    jobs.append(dict(job=row['id'], stage=row['stage'], gpu=row['gpu'], identity=current))
assert len(jobs) == 5 and len(state['queued_jobs']) == 1
before = yaml.safe_load((root/'operations/rank-capacity37-resource-handoff-v2/canonical-before.yaml').read_text())
assert before['frozen_input_sha256'] == state['frozen_input_sha256'] and before['config'] == state['config']
result = dict(status='PASS_LIVE_RESOURCE_HANDOFF_STARTUP_AND_UNCHANGED_SCIENTIFIC_PINS', checked_input_files=checked,
              owner=owner, original_producers=original_producers, jobs=jobs, queued_jobs=state['queued_jobs'],
              original_scheduler_stopped_for_exit_status_capture=True, original_config_and_frozen_hashes_unchanged=True,
              legacy_exit_capture_and_release_pending=True, scientific_utility_asserted=False)
destination = root/'results/capacity-resource-handoff-live-review.json'
assert not destination.exists()
destination.write_text(json.dumps(result, indent=2))
print(result['status'], checked, flush=True)
