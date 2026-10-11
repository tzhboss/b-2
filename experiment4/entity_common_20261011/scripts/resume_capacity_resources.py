"""Resource-only continuation of the pinned queue, retaining existing workers."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(pid):
    raw = (Path('/proc')/str(pid)/'stat').read_text()
    fields = raw[raw.rfind(')')+2:].split()
    return dict(pid=pid, state=fields[0], parent=int(fields[1]), start_ticks=int(fields[19]), exit_status=int(fields[49]))


def command(pid):
    return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0', b' ').decode()


class AdoptedProcess:
    def __init__(self, identity, observed):
        self.pid = identity['pid']
        self.identity = identity
        self.observed = observed

    def poll(self):
        # Missing/reused handles are errors, not presumed successful training.
        current = process(self.pid)
        assert current['start_ticks'] == self.identity['start_ticks'], 'Adopted PID was reused'
        if current['state'] != 'Z':
            return None
        assert current['parent'] == self.identity['parent']
        code = os.waitstatus_to_exitcode(current['exit_status'])
        self.observed[self.pid] = code
        return code


def transformed_source(source):
    changes = [
        ("pending=list(cfg['jobs']);pending_audits=[];active={};free=set(cfg['available_gpu_indices']);preflight_passed=set()",
         "pending,pending_audits,active,free,preflight_passed=_resume(root,cfg,can,s)"),
        ("s.update(status='waiting_for_gpu_queue_dependency',coordinator_pid=os.getpid(),jobs={},preflight_passed=[],active_processes=[])",
         "s.update(status='resumed_resource_only_comparison',coordinator_pid=os.getpid(),resource_handoff=_handoff_metadata)"),
        ("    save()\n    while pending or active or pending_audits:",
         "    save()\n    _ownership_transferred()\n    while pending or active or pending_audits:"),
        ("        save()\n        if pending or active or pending_audits:time.sleep(3)",
         "        _release_legacy_if_safe()\n        save()\n        if pending or active or pending_audits:time.sleep(3)"),
    ]
    for before, after in changes:
        assert source.count(before) == 1, 'Pinned scheduler structure changed'
        source = source.replace(before, after)
    return source


def controls(root, cfg):
    source = transformed_source((root/cfg['source_coordinator']).read_text())
    compile(source, '<resource-only-queue>', 'exec')
    rows = []
    for expected in (0, 7):
        child = subprocess.Popen([sys.executable, '-c', f'import sys,time;time.sleep(.05);sys.exit({expected})'])
        try:
            identity = process(child.pid)
            captured = {}
            adopted = AdoptedProcess(identity, captured)
            deadline = time.monotonic()+5
            while True:
                code = adopted.poll()
                if code is not None:
                    break
                assert time.monotonic() < deadline
                time.sleep(.01)
            assert code == expected == child.wait()
            rows.append(dict(expected_exit_code=expected, captured_exit_code=code, zombie_status_captured=True))
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(.3)'])
    try:
        identity = process(child.pid)
        identity['start_ticks'] += 1
        try:
            AdoptedProcess(identity, {}).poll()
        except AssertionError:
            reused_rejected = True
        else:
            raise AssertionError('Reused PID must be rejected')
    finally:
        child.wait()
    missing = dict(pid=2147483647, parent=os.getpid(), start_ticks=0)
    try:
        AdoptedProcess(missing, {}).poll()
    except FileNotFoundError:
        missing_rejected = True
    else:
        raise AssertionError('Missing PID must not imply terminal success')
    result = dict(status='PASS_BOUNDED_QUEUE_HANDOFF_CONTROLS', exits=rows, reused_pid_rejected=reused_rejected,
                  missing_pid_rejected=missing_rejected, transformed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  script_sha256=digest(Path(__file__)), model_reads=0, model_updates=0, existing_processes_signalled=0)
    destination = root/'results/capacity-resource-handoff-controls.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2))
    print(result['status'], flush=True)


def resume(root, addon_path, cfg):
    can = root/'experiments'/cfg['experiment']/'experiment.yaml'
    state = yaml.safe_load(can.read_text())
    assert state['config'] == cfg['source_config']
    assert state['coordinator_pid'] == cfg['expected_legacy_coordinator_pid']
    assert state['status'] == 'running_registered_comparison'
    assert state['queued_jobs'] == cfg['expected_pending_jobs'] and not state['queued_audits']
    assert [row['id'] for row in state['active_processes']] == cfg['expected_active_jobs']
    assert all(row['stage'] == 'train' for row in state['active_processes'])
    for name, sha in state['frozen_input_sha256'].items():
        assert digest(root/name) == sha
    for name, sha in state['external_source_sha256'].items():
        assert digest(Path(name)) == sha
    qualified = json.loads((root/'results/capacity-resource-handoff-controls.json').read_text())
    assert qualified['status'] == 'PASS_BOUNDED_QUEUE_HANDOFF_CONTROLS'
    assert qualified['script_sha256'] == digest(Path(__file__))
    legacy_pid = cfg['expected_legacy_coordinator_pid']
    legacy = process(legacy_pid)
    assert legacy['state'] in ('S', 'R')
    assert str(root/cfg['source_coordinator']) in command(legacy_pid) and cfg['source_config'] in command(legacy_pid)
    identities = {}
    for row in state['active_processes']:
        identity = process(row['pid'])
        assert identity['parent'] == legacy_pid and identity['state'] in ('S', 'R')
        cmd = command(row['pid'])
        assert str(root) in cmd and '--job '+row['id'] in cmd and '--stage train' in cmd
        identities[row['id']] = identity
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    idle = {int(i): dict(name=n.strip(), memory_mib=int(m), utilization=int(u)) for line in usage.splitlines() for i,n,m,u in [line.split(',')]}
    assert all(idle[i]['memory_mib'] <= 1024 and idle[i]['utilization'] <= 20 and idle[i]['name'] == 'NVIDIA GeForce RTX 3090' for i in cfg['additional_gpu_indices'])
    source_cfg = json.loads((root/cfg['source_config']).read_text())
    for name in source_cfg['resource_claim_sources']:
        peer = yaml.safe_load((root/name).read_text())
        for row in peer.get('active_processes', []):
            try:
                identity = process(row['pid'])
            except FileNotFoundError:
                continue
            if identity['state'] != 'Z':
                assert row.get('gpu') not in cfg['additional_gpu_indices']
    rendered = transformed_source((root/cfg['source_coordinator']).read_text())
    assert hashlib.sha256(rendered.encode()).hexdigest() == qualified['transformed_source_sha256']
    folder = root/cfg['operation_directory']
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'canonical-before.yaml').write_text(yaml.safe_dump(state, sort_keys=False))
    (folder/'rendered-queue.py').write_text(rendered)
    observed = {}
    ownership = False
    released = False
    metadata = dict(scope=cfg['scope'], addendum=str(addon_path.relative_to(root)), addendum_sha256=digest(addon_path),
                    runtime_queue_sha256=digest(Path(__file__)), rendered_queue_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                    legacy_coordinator=legacy, adopted_workers=identities, additional_gpu_indices=cfg['additional_gpu_indices'],
                    verified_idle_gpu_snapshot=idle, controls_sha256=digest(root/'results/capacity-resource-handoff-controls.json'), limits=cfg['limits'])
    (folder/'receipt.json').write_text(json.dumps(metadata, indent=2))

    def initialize(runtime_root, runtime_cfg, runtime_can, runtime_state):
        assert runtime_root == root and runtime_can == can
        assert process(legacy_pid)['state'] == 'T'
        assert runtime_state == state
        active = {}
        for row in state['active_processes']:
            job = next(job for job in runtime_cfg['jobs'] if job['id'] == row['id'])
            assert process(row['pid'])['start_ticks'] == identities[row['id']]['start_ticks']
            active[row['id']] = dict(job=job, proc=AdoptedProcess(identities[row['id']], observed), gpu=row['gpu'], stage=row['stage'],
                                     log=(root/runtime_cfg['log_directory']/f"{row['id']}-{row['stage']}.log").open('ab'))
        free = set(runtime_cfg['available_gpu_indices']) | set(cfg['additional_gpu_indices'])
        free -= {row['gpu'] for row in state['active_processes']}
        pending = [job for job in runtime_cfg['jobs'] if job['id'] in state['queued_jobs']]
        return pending, [], active, free, set(state['preflight_passed'])

    def transferred():
        nonlocal ownership
        ownership = True

    def release_legacy():
        nonlocal released
        if released or len(observed) != len(identities):
            return
        current = process(legacy_pid)
        assert current['start_ticks'] == legacy['start_ticks'] and current['state'] == 'T'
        assert str(root/cfg['source_coordinator']) in command(legacy_pid)
        # All adopted wrappers are terminal, with exact exit codes captured.
        # Kill only the stopped obsolete scheduler to prevent duplicate dispatch.
        os.kill(legacy_pid, signal.SIGKILL)
        released = True
        metadata['legacy_scheduler_released_after_captured_worker_exits'] = dict(observed)
        (folder/'receipt.json').write_text(json.dumps(metadata, indent=2))
        print('OWNED_LEGACY_SCHEDULER_RELEASED', legacy_pid, observed, flush=True)

    namespace = dict(__name__='resource_handoff_queue', _resume=initialize, _handoff_metadata=metadata,
                     _ownership_transferred=transferred, _release_legacy_if_safe=release_legacy)
    exec(compile(rendered, str(folder/'rendered-queue.py'), 'exec'), namespace)
    os.kill(legacy_pid, signal.SIGSTOP)
    try:
        deadline = time.monotonic()+5
        while process(legacy_pid)['state'] != 'T':
            assert time.monotonic() < deadline
            time.sleep(.01)
        print('RESOURCE_ONLY_QUEUE_HANDOFF', legacy_pid, os.getpid(), identities, flush=True)
        namespace['main'](root, cfg['source_config'])
    except BaseException as error:
        (folder/'retained-error.json').write_text(json.dumps(dict(error=repr(error), ownership_transferred=ownership, observed_worker_exits=observed), indent=2))
        if not ownership:
            can.write_text(yaml.safe_dump(state, sort_keys=False))
            current = process(legacy_pid)
            assert current['start_ticks'] == legacy['start_ticks']
            os.kill(legacy_pid, signal.SIGCONT)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--addendum', default='configs/rank-capacity37-resource-addendum.json')
    parser.add_argument('--stage', choices=['controls', 'resume'], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    addon = root/args.addendum
    cfg = json.loads(addon.read_text())
    if args.stage == 'controls':
        controls(root, cfg)
    else:
        resume(root, addon, cfg)
