"""Refresh live queue status while retaining all39 existing metric rows."""
import argparse
import hashlib
import json
from pathlib import Path

from build_common_table_v2 import build

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
path = root/'tables/common-comparison-v4/table.json'
before = json.loads(path.read_text())
build(root, 'configs/common-table-v4.json')
after = json.loads(path.read_text())
key = lambda row: (row['method'], row['model'], row['dimension'])
old = {key(row): row for row in before['rows'] if not row['method'].startswith('CapacityFrozen')}
new = {key(row): row for row in after['rows'] if not row['method'].startswith('CapacityFrozen')}
assert old.keys() == new.keys() and len(old) == 39
for slot in old:
    for name in ['metrics', 'query_checks', 'new_directional_metrics', 'training_qualifications', 'audited_seeds']:
        assert old[slot][name] == new[slot][name], (slot, name)
assert before['development_utility_screen'] == after['development_utility_screen']
assert before['additional_development_utility'] == after['additional_development_utility']
result = dict(status='PASS_RESOURCE_QUEUE_STATUS_REFRESH_AND_39_ORIGINAL_ROWS_UNCHANGED',
              unchanged_original_rows=39, completed_jobs=after['completed_jobs'], planned_jobs=after['planned_jobs'],
              table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), scientific_utility_asserted=False)
out = root/'results/capacity-resource-table-review.json'
assert not out.exists()
out.write_text(json.dumps(result, indent=2))
print(result['status'], flush=True)
