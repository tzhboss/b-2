"""Refresh audited partial rows without making a three-seed utility decision."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(pid):
    base = Path('/proc') / str(pid)
    raw = (base / 'stat').read_text().rsplit(') ', 1)[1].split()
    assert raw[0] not in ('Z', 'X'), (pid, raw[0])
    command = (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    return dict(pid=pid, state=raw[0], parent=int(raw[1]),
                start_ticks=int(raw[19]), command=command)


def main(root, output):
    output = root / output
    assert not output.exists()
    experiment = 'KGE-ENTITY-COMMON-RANK-CAPACITY-037'
    utility = 'KGE-ENTITY-COMMON-RANK-CAPACITY-UTILITY-038'
    states = {}
    pins = {}
    for name in (experiment, utility):
        path = root / 'experiments' / name / 'experiment.yaml'
        state = yaml.safe_load(path.read_text())
        states[name] = state
        for name, digest in state['frozen_input_sha256'].items():
            assert sha(root / name) == digest, name
            pins[name] = digest
    state = states[experiment]
    owner = process(state['coordinator_pid'])
    assert 'resume_capacity_resources_v2.py' in owner['command']
    utility_owner = process(states[utility]['coordinator_pid'])
    assert 'coordinate_cpu_audits_v2.py' in utility_owner['command']
    assert states[utility]['status'] == 'waiting_resource_audit_queue'
    cfg = json.loads((root / state['config']).read_text())
    rows = []
    evidence = {}
    observed_updates = 0
    for job in cfg['jobs']:
        jid = job['id']
        status = state['jobs'][jid]
        item = dict(id=jid, seed=job['seed'], method=job['method'], state=status)
        if status['status'] == 'completed_audited':
            out = root / 'artifacts/incremental' / jid
            audit = json.loads((out / 'audit.json').read_text())
            solver = audit['solver_audit']
            assert not audit['failures'] and not solver['failures']
            assert solver['patches_replayed'] == 3753
            assert solver['terminal_cpu_counts_exact']
            assert solver['status'] == 'PASS_ALL3753_PATCHES_AND_ALL_QUERY_CPU_TERMINAL_RANKS'
            assert not solver['test_read'] and not solver['valid_read']
            tr = json.loads((out / 'training-receipt.json').read_text())
            assert sha(out / 'best.pt') == tr['checkpoint_sha256']
            metrics = json.loads((out / 'evaluation.json').read_text())
            assert metrics['old_after']['queries'] == 74428
            assert metrics['new_after']['queries'] == 12406
            assert metrics['candidates'] == 11258
            item.update(observed_updates=3753, solver_audit=solver,
                        common_test_metrics=metrics,
                        utility_decision='PENDING_COMPLETE_THREE_SEEDS_AND_038')
            for name in ('audit.json', 'solver-audit.json', 'evaluation.json', 'training-receipt.json'):
                path = out / name
                evidence[str(path.relative_to(root))] = sha(path)
        else:
            handle = process(status['pid'])
            assert jid in handle['command']
            assert handle['parent'] == owner['pid']
            children_path = Path('/proc') / str(status['pid']) / 'task' / str(status['pid']) / 'children'
            children = [process(int(pid)) for pid in children_path.read_text().split()]
            assert any(jid in p['command'] and 'run_shared_graph.py' in p['command'] for p in children)
            log = root / cfg['log_directory'] / f'{jid}-train.log'
            matches = re.findall(r'SHARED_GRAPH_PROGRESS\s+' + re.escape(jid) + r'\s+(\d+)\s+3753\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)', log.read_text())
            assert matches, jid
            last = matches[-1]
            item.update(observed_updates=int(last[0]), accepted_updates=int(last[1]),
                        observed_fallbacks=int(last[2]), observed_steps_modifying_old_rows=int(last[3]),
                        observed_elapsed_s=float(last[4]), wrapper=handle, children=children)
        observed_updates += item['observed_updates']
        rows.append(item)
    assert [r['seed'] for r in rows if r['state']['status'] == 'completed_audited'] == [17, 17]
    table = root / 'tables/common-comparison-v4/table.json'
    before_sha = sha(table)
    before = json.loads(table.read_text())
    old_rows = {r['method'] + '/' + r['model']: r for r in before['rows']
                if not r['method'].startswith('CapacityFrozen')}
    python = os.path.abspath(root / cfg['python'])
    subprocess.run([python, str(root / 'scripts/build_common_table_v2.py'),
                    '--root', str(root), '--config', 'configs/common-table-v4.json'], check=True)
    after = json.loads(table.read_text())
    current = {r['method'] + '/' + r['model']: r for r in after['rows']
               if not r['method'].startswith('CapacityFrozen')}
    assert current == old_rows
    for row in after['rows']:
        if row['method'].startswith('CapacityFrozen'):
            assert row['audited_seeds'] == [17] and not row['complete_seed_set']
    result = dict(status='PASS_PARTIAL_SEED17_AUDITS_AND_LIVE_REMAINING_HANDLES',
                  observed_utc=datetime.now(timezone.utc).isoformat(),
                  coordinator=owner, utility_coordinator=utility_owner, jobs=rows,
                  observed_updates=observed_updates, planned_updates=6*3753,
                  observed_update_fraction=observed_updates/(6*3753),
                  audited_jobs=2, total_jobs=6, historical_table_rows_unchanged=len(old_rows),
                  table_before_sha256=before_sha, table_after_sha256=sha(table),
                  frozen_input_sha256=pins, audited_evidence_sha256=evidence,
                  scientific_utility_asserted=False, gates_changed=False,
                  limits=['Progress counts exclude terminal evaluation and CPU/038 utility audits.',
                          'Only seed17 is audited; no three-seed utility decision.',
                          'Exempted TRAIN cost comparison waits for registered038.'])
    output.write_text(json.dumps(result, indent=2))
    print(result['status'], after['completed_jobs'], after['planned_jobs'], observed_updates, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output)
