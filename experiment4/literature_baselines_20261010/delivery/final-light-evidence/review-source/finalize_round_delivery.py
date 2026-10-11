"""Prepare the final read-only evidence inventory after reviewed 004 promotion.

Does not run training, alter scientific outputs, or relax audit dispositions.
"""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import yaml

R = Path('/data/lc/tzh/experiment4/literature_baselines_20261010')

def load(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

expected = {
    '001': ('completed_audited', 'PASS_WITH_LIMITATIONS'),
    '002': ('completed_audited', 'PASS_WITH_LIMITATIONS'),
    '003': ('terminal_audited_qualified_evidence', 'FAIL_RETAINED'),
    '004': ('completed_audited_qualified', 'PASS_NATIVE_NUMERICS_WITH_RESTORE_LIMITATIONS'),
    '005': ('completed_audited_derived', 'PASS_DERIVED_NUMERICS_WITH_LIMITATIONS'),
    '006': ('completed_audited', 'PASS_WITH_LIMITATIONS'),
}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
records = []
manifest_checks = 0
training_receipts = []
owned_pids = set()
for n, (status, audit_status) in expected.items():
    eid = 'KGE-LIT-' + n
    cp = R / 'experiments' / eid / 'experiment.yaml'
    c = yaml.safe_load(cp.read_text())
    assert c['status'] == status and c['audit']['status'] == audit_status, eid
    assert not c.get('active_processes'), eid
    digest = c['audit'].get('summary_sha256')
    if digest:
        assert sha(R / 'results' / eid / 'summary.json') == digest, eid
    for p in (R / 'results' / eid).rglob('manifest-sha256.json'):
        for name, digest in load(p).items():
            assert sha(p.parent / name) == digest, str(p.parent / name)
            manifest_checks += 1
    if n == '002':
        p = R / 'results' / eid / 'archive-byte-manifest-sha256.json'
        # This inventory is a file-byte check, not an extra scientific audit.
        archive = load(p)
        mapping = archive.get('files', archive.get('sha256', archive))
        if isinstance(mapping, dict):
            for name, digest in mapping.items():
                if isinstance(digest, str) and len(digest) == 64:
                    assert sha(p.parent / name) == digest, name
                    manifest_checks += 1
    if n != '005':
        jobs = load(R / 'configs/experiments' / (eid + '-jobs.json'))
        for job in jobs:
            t = load(R / job['run_directory'] / 'terminal.json')
            assert t['exit_code'] == 0, job['id']
            if n != '003':
                assert t['source_inputs_unchanged'], job['id']
            else:
                assert c['audit']['original_producer_integrity_fail_retained']
                assert c['audit']['scientific_artifact_status'] == 'PASS_SAMPLED_POST_TRAIN_REPLAY'
            training_receipts.append(dict(id=job['id'],exit_code=t['exit_code'],
                terminal_sha256=sha(R / job['run_directory'] / 'terminal.json'),
                integrity_disposition='FAIL_RETAINED_UNUSED_METADATA' if n == '003' else 'UNCHANGED'))
    if c.get('scheduler_pid'):
        owned_pids.add(c['scheduler_pid'])
    records.append(dict(id=eid,status=c['status'],audit=c['audit'],active_processes=[],canonical_sha256=sha(cp)))

assert len(training_receipts) == 10
aux = []
for eid in ['KGE-LIT-004-AUDIT-V3', 'KGE-LIT-004-SELECTION-V3']:
    cp = R / 'experiments' / eid / 'experiment.yaml'
    c = yaml.safe_load(cp.read_text())
    assert not c.get('active_processes'), eid
    assert c['status'] in ['all_audits_terminal_pending_controller_review',
                           'terminal_pending_controller_review', 'completed_controller_reviewed'], eid
    if c.get('coordinator_pid'):
        owned_pids.add(c['coordinator_pid'])
    aux.append(dict(id=eid,status=c['status'],canonical_sha256=sha(cp)))

if owned_pids:
    ps = subprocess.run(['ps','-p',','.join(map(str, sorted(owned_pids))),
                         '-o','pid=,stat=,comm='], capture_output=True, text=True)
    assert not ps.stdout.strip(), ps.stdout

aggregate = load(R / 'results/KGE-LIT-004/summary.json')
assert len(aggregate['runs']) == 2 and aggregate['independent_examples'] == 320
assert aggregate['before_after_records'] == 640 and aggregate['cpu_failures'] == 0
for x in aggregate['runs']:
    assert x['training_epochs'] == 20 and x['rank_top3_exact']
    assert x['max_target_logit_error'] <= 1e-4

continual = load(R / 'tables/continual_kge_real_results.json')
editor = load(R / 'tables/kgeditor_exposed_real_results.json')
assert len(continual['rows']) == 8 and len(editor['rows']) == 2
assert all(x['training_complete'] for x in continual['rows'])
assert editor['parent_canonical_status'] == expected['004'][0]
assert continual['not_a_mixed_SOTA_ranking'] and editor['not_heldout']

inventory = dict(snapshot_utc=now,source='fresh terminal canonical and release bytes',
    comparable_scientific_protocol=False,
    warning='Separate native continual / revised validation / derived candidate / exposed editor protocols. This is a final snapshot, not a live monitor.',
    experiments=records)
write(R / 'tables/literature-baseline-evidence-status.json', inventory)
final = dict(utc=now,status='ROUND_COMPLETE_WITH_RETAINED_LIMITATIONS_STOP_BEFORE_NEXT_ROUND',
    complete_registered_native_training_runs=10,training_receipts=training_receipts,
    release_manifest_byte_checks=manifest_checks,release_manifest_mismatches=0,
    no_registered_native_or_current_audit_process_alive=True,
    canonical_records=records,audit_coordinators=aux,
    distinct_result_tables=dict(continual_rows=8,exposed_editor_rows=2),
    parallel_evidence=load(R / 'tables/parallel-execution-evidence.json'),
    no_next_round_started=True,
    scientific_scope='Completed selected literature protocols; not a new optimized method or publication GO. SAGE original integrity FAIL and KGEditor restoration/exposure limitations retained.',
    delivery_transfer_status='REMOTE_READY_LOCAL_SHA_VERIFICATION_REQUIRED')
write(R / 'plan/review/completion-requirements-audit-final.json', final)
print(json.dumps(dict(status=final['status'],native_runs=10,manifest_checks=manifest_checks)))
