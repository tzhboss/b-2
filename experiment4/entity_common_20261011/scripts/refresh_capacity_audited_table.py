"""Refresh qualified capacity rows while retaining historical scientific data."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

from build_common_table_v2 import build


def main(root, output):
    output = root / output
    assert not output.exists(), output
    pins = {}
    for experiment in ('KGE-ENTITY-COMMON-RANK-CAPACITY-037',
                       'KGE-ENTITY-COMMON-RANK-CAPACITY-UTILITY-038'):
        state = yaml.safe_load((root / 'experiments' / experiment / 'experiment.yaml').read_text())
        for name, digest in state['frozen_input_sha256'].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
            pins[name] = digest
    path = root / 'tables/common-comparison-v4/table.json'
    before = json.loads(path.read_text())
    build(root, 'configs/common-table-v4.json')
    after = json.loads(path.read_text())
    key = lambda row: (row['method'], row['model'], row['dimension'])
    old = {key(row): row for row in before['rows'] if not row['method'].startswith('CapacityFrozen')}
    new = {key(row): row for row in after['rows'] if not row['method'].startswith('CapacityFrozen')}
    assert len(old) == 39 and old == new
    assert before['development_utility_screen'] == after['development_utility_screen']
    stable = lambda table: [row for row in table['additional_development_utility']
                           if row['experiment'] != 'KGE-ENTITY-COMMON-RANK-CAPACITY-UTILITY-038']
    assert stable(before) == stable(after)
    capacity = [dict(method=row['method'], audited_seeds=row['audited_seeds'],
                     complete_seed_set=row['complete_seed_set'])
                for row in after['rows'] if row['method'].startswith('CapacityFrozen')]
    result = dict(status='PASS_CAPACITY_TABLE_REFRESH_AND_39_HISTORICAL_ROWS_UNCHANGED',
                  historical_rows_unchanged=39, completed_jobs=after['completed_jobs'],
                  planned_jobs=after['planned_jobs'], capacity_qualification=capacity,
                  table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  frozen_input_sha256=pins, scientific_utility_asserted=False,
                  registered_gates_changed=False)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result | {'frozen_input_sha256': 'all verified'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output)
