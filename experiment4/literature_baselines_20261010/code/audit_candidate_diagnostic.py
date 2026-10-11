"""CPU-only provenance and rank-derived audit for KGE-LIT-005.

Reads completed artifacts only. Never imports torch, reconstructs scores, launches
GPU work, changes canonical state, or promotes results. JSON reports go to stdout
or an explicitly selected path under plan/review.
"""
import argparse
import collections
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASTER_SHA = '7d211e0780af166f027623f57668b0ab5e61f160dce6cc875baca228b88a3951'
NATIVE_PROTOCOL_SHA = 'c4fe8740f6bf7783fc7893a2004164ec42bfd46d5b06495665a2064809b8a2fb'
NATIVE_AUDITOR_SHA = 'ccb48f4e3da560b26dcdf2d03f6809d46e34f605b539a348a15dc9b778411512'
PROTOCOL_SHA = 'e36e51c12d983f5875d326e94f3755d2dccf8b3d39af719a76daacabdd79111a'
EVALUATOR_SHA = '313903da5d506a4086bcc0846cb712c764089929f6899f1734aa5b710655af78'
RETRY_SHA = '8f81e97ae44020b11cd35beeba9683bdff0babb342db39a0f8f448fbb72d0b63'
ARMS = ('native_replay', 'candidate_expansion_only', 'strict_rank_old_candidates',
        'strict_rank_final_candidates', 'current_filter_final_candidates')
METRICS = ('mrr', 'hits1', 'hits3', 'hits10')
LIMITS = [
    'CPU provenance, query/filter contracts, numeric derivations and necessary rank identities only.',
    'No independent embedding-score or CUDA all-query rank recomputation.',
    'Source native PASS is bound and consumed, not independently repeated by this script.',
    'Rank identities and tie bounds are necessary checks; they cannot certify score provenance alone.',
    'Native validation-best precision, historical competitors and single-seed limits remain.',
    'No canonical changes or formal result promotion.',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(4 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


class InvalidArtifact(Exception):
    pass


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []

    def check(self, condition, label, n=1):
        self.count += int(n)
        if not bool(condition):
            self.failures.append(label)
        return bool(condition)

    def require(self, condition, label, n=1):
        if not self.check(condition, label, n):
            raise InvalidArtifact(label)


def snapshot_contracts(native, dataset):
    """Independently reconstruct author's IDs, native query order and truths."""
    entities, relations = {}, {}
    truths = collections.defaultdict(set)
    snapshots = []
    for ss in range(5):
        facts = {}
        for split in ('train', 'valid', 'test'):
            triples = []
            path = native / 'worktree/data' / dataset / str(ss) / (split + '.txt')
            with path.open(encoding='utf-8') as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 3:
                        raise InvalidArtifact(f'malformed triple {path.name}:{number}')
                    h, r, t = parts
                    for e in (h, t):
                        if e not in entities:
                            entities[e] = len(entities)
                    if r not in relations:
                        idx = len(relations)
                        relations[r], relations[r + '_inv'] = idx, idx + 1
                    triples.append((entities[h], relations[r], entities[t]))
            facts[split] = triples
        for triples in facts.values():
            for h, r, t in triples:
                truths[h, r].add(t)
                truths[t, r + 1].add(h)
        ordered = sorted(facts['test'], key=lambda row: row[1])
        queries = ordered + [(t, r + 1, h) for h, r, t in ordered]
        snapshots.append({'entities': len(entities), 'relations': len(relations),
                          'queries': np.asarray(queries, dtype=np.int64).reshape(-1, 3),
                          'filters': copy.deepcopy(dict(truths))})
    return snapshots


def array_schema(data, names, n, checks, label):
    checks.require(all(name in data for name in names), label + ': required arrays')
    for name in names:
        value = data[name]
        checks.require(value.shape == (n,) and value.dtype.kind in 'iu',
                       label + ': 1D integer length ' + name)


def rank_relations(data, old_n, final_n, filtered, later, t, checks):
    """Necessary rank identities; intentionally makes no score-replay claim."""
    n = len(data['native_replay'])
    native, expanded, old, final, current = (data[k] for k in ARMS)
    ties = data['equal_score_other_candidates']
    for name in ARMS:
        limit = old_n if name in (ARMS[0], ARMS[2]) else final_n
        checks.check(np.all((data[name] >= 1) & (data[name] <= limit)),
                     f't{t}: candidate rank bounds {name}', n)
    checks.check(np.all((ties >= 0) & (ties <= final_n - filtered)),
                 f't{t}: equal-score bounds after historical filter', n)
    checks.check(np.all(old <= native), f't{t}: optimistic old <= native', n)
    checks.check(np.all(final <= expanded), f't{t}: optimistic final <= expanded', n)
    checks.check(np.all(final >= old), f't{t}: strict candidate expansion monotonicity', n)
    # Cast valid ranges to signed int64 before subtracting unsigned arrays.
    old, final, current = (v.astype(np.int64) for v in (old, final, current))
    checks.check(np.all(final - old <= final_n - old_n),
                 f't{t}: added candidate count bounds rank delta', n)
    checks.check(np.all(current <= final), f't{t}: final filter cannot worsen strict rank', n)
    checks.check(np.all((final - current >= 0) & (final - current <= later)),
                 f't{t}: later truth count bounds filter gain', n)
    checks.check(np.all(native.astype(np.int64) - old <= ties),
                 f't{t}: native old tie-position bound', n)
    checks.check(np.all(expanded.astype(np.int64) - final <= ties),
                 f't{t}: expanded tie-position bound', n)
    zero = ties == 0
    checks.check(np.array_equal(native[zero], old[zero]) and
                 np.array_equal(expanded[zero], final[zero]),
                 f't{t}: zero ties imply argsort=strict', int(np.sum(zero)))
    absent = later == 0
    checks.check(np.array_equal(current[absent], final[absent]),
                 f't{t}: no later truths imply same strict filter rank', int(np.sum(absent)))
    if t == 4:
        checks.check(np.array_equal(native, expanded), 't4: native=expanded control', n)
        checks.check(np.array_equal(old, final) and np.array_equal(final, current),
                     't4: all strict controls identical', n)
        checks.check(np.all(later == 0), 't4: no later truths control', n)


def rank_metrics(ranks):
    n = len(ranks)
    if not n:
        raise InvalidArtifact('empty registered test cohort')
    return {'mrr': math.fsum(1.0 / int(value) for value in ranks) / n,
            **{'hits' + str(k): int(np.count_nonzero(ranks <= k)) / n for k in (1, 3, 10)}}


def close(value, expected):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(float(value)) and math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-12)


def audit_job(root, job_id):
    root = Path(root).resolve()
    out = root / 'runs/candidate-diagnostic' / job_id
    terminal_path = out / 'terminal.json'
    result = {'id': job_id, 'scope': 'CPU_DERIVED_NUMERICS_AND_PROVENANCE',
              'status': 'WAITING', 'checks': 0, 'failures': [],
              'independent_score_replays': 0, 'limitations': LIMITS.copy()}
    if not terminal_path.exists():
        result['waiting_reason'] = 'completed candidate terminal.json absent; partial outputs not audited'
        return result
    checks = Checks()
    hashes, table, cohort_total = {}, [], 0
    try:
        def bound(path, digest, label):
            checks.require(sha(path) == digest, label)
            hashes[str(Path(path).relative_to(root))] = digest
        master_path = root / 'configs/experiments/KGE-LIT-001-jobs.json'
        bound(master_path, MASTER_SHA, 'frozen master digest')
        jobs = [j for j in read_json(master_path) if j['id'] == job_id]
        checks.require(len(jobs) == 1, 'unique registered job')
        job = jobs[0]
        native = root / job['run_directory']
        bound(root / 'configs/protocols/literature-baselines.yaml', NATIVE_PROTOCOL_SHA, 'native protocol')
        bound(root / 'code/audit_incde_native.py', NATIVE_AUDITOR_SHA, 'required native auditor code')
        bound(root / 'configs/protocols/KGE-LIT-005-candidate-diagnostic.json', PROTOCOL_SHA, 'diagnostic protocol')
        bound(root / 'code/evaluate_candidate_expansion.py', EVALUATOR_SHA, 'mechanically fixed evaluator')
        retry_path = root / 'logs/KGE-LIT-005-mechanical-retry-registration.json'
        bound(retry_path, RETRY_SHA, 'mechanical retry registration')
        retry = read_json(retry_path)
        checks.require(retry['new_selector_sha256'] == EVALUATOR_SHA and
                       retry['prior_canonical']['protocol_sha256'] == PROTOCOL_SHA and
                       retry['native_outputs_modified'] is False, 'retry scope binds fixed evaluator')
        terminal_digest = sha(terminal_path)
        hashes[str(terminal_path.relative_to(root))] = terminal_digest
        terminal = read_json(terminal_path)
        checks.require(terminal['id'] == job_id and terminal['status'] == 'terminal_unreviewed',
                       'completed terminal identity and producer status')
        checks.require(terminal['implementation_sha256'] == EVALUATOR_SHA and
                       terminal['registration_sha256'] == PROTOCOL_SHA, 'terminal version/protocol')
        checks.check(terminal.get('all_query_native_replay_exact') is True, 'producer native replay claim')
        checks.check(close(terminal['full_process_elapsed_s'], float(terminal['full_process_elapsed_s'])) and
                     terminal['full_process_elapsed_s'] >= 0, 'producer elapsed is finite and nonnegative')
        audit_path = root / 'results/KGE-LIT-001-audit' / job_id / 'audit.json'
        bound(audit_path, terminal['source_audit_sha256'], 'source audit report bytes')
        audit = read_json(audit_path)
        checks.require(audit['status'] == 'PASS' and audit['id'] == job_id and
                       audit['dataset'] == job['dataset'] and audit['registration_sha256'] == MASTER_SHA and
                       audit['protocol_sha256'] == NATIVE_PROTOCOL_SHA and
                       audit['audit_code_sha256'] == NATIVE_AUDITOR_SHA, 'source audit identity and contract')
        ranks_expected = {f'ranks/train4_test{t}_post_train.npz' for t in range(5)}
        ckpt = 'checkpoint/' + job['dataset'] + '/4model_best.tar'
        expected_keys = ranks_expected | {ckpt, 'registered-job.json', 'terminal.json'}
        checks.require(expected_keys <= set(audit['artifact_sha256']), 'source audit exact required input coverage')
        expected_consumed = {name: audit['artifact_sha256'][name] for name in expected_keys}
        checks.require(terminal['consumed_input_sha256'] == expected_consumed,
                       'exact five post-train/checkpoint/declaration consumed mapping')
        for name, digest in expected_consumed.items():
            bound(native / name, digest, 'consumed input:' + name)
        checks.require(read_json(native / 'registered-job.json') == job, 'run registration equals master')
        native_terminal = read_json(native / 'terminal.json')
        checks.require(native_terminal['exit_code'] == 0 and
                       native_terminal.get('source_inputs_unchanged') is True, 'native completion')
        runtime = read_json(native / 'runtime-command.json')
        source = root / job['copy_repository']
        worktree = native / 'worktree'
        command = [str(worktree) + arg[len(str(source)):] if arg.startswith(str(source) + '/') else arg
                   for arg in job['command']]
        checks.require(runtime == {'command': command, 'cwd': str(worktree), 'gpu': job['gpu']},
                       'native runtime binding')
        expected_copied = {'worktree/' + str((root / name).relative_to(source)): digest
                           for name, digest in job['input_sha256'].items()
                           if (root / name).is_relative_to(source)}
        copied_path = native / 'copied-source-sha256.json'
        bound(copied_path, terminal['copied_input_manifest_sha256'], 'consumed copied manifest bytes')
        checks.require(read_json(copied_path) == expected_copied, 'copied declarations equal master')
        for name, digest in expected_copied.items():
            if Path(name).name != 'train_sorted_by_edges_betweenness.txt':
                bound(native / name, digest, 'copied raw/code:' + name)
        snapshots = snapshot_contracts(native, job['dataset'])
        names = {f'test{t}.npz' for t in range(5)}
        checks.require({p.name for p in out.glob('*.npz')} == names, 'exact five output NPZ identities')
        checks.require(set(terminal['output_sha256']) == names, 'terminal output coverage')
        tasks = terminal['task_table']
        checks.require(len(tasks) == 5 and [v['task_snapshot'] for v in tasks] == list(range(5)),
                       'terminal task rows0..4')
        final_n = snapshots[4]['entities']
        query_fields = ('head', 'relation', 'tail')
        extra_fields = ('equal_score_other_candidates', 'later_filtered_truths')
        for t in range(5):
            label = f't{t}'
            output = out / f'test{t}.npz'
            bound(output, terminal['output_sha256'][output.name], label + ': output hash')
            with np.load(output, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            checks.require(set(data) == set(query_fields + ARMS + extra_fields), label + ': exact output schema')
            snap = snapshots[t]
            n = len(snap['queries'])
            checks.require(n > 0, label + ': nonempty registered cohort')
            array_schema(data, query_fields + ARMS + extra_fields, n, checks, label)
            q = np.column_stack([data[k] for k in query_fields])
            checks.require(np.array_equal(q, snap['queries']), label + ': exact native query mapping/order', n)
            with np.load(native / f'ranks/train4_test{t}_post_train.npz', allow_pickle=False) as z:
                original = {k: z[k] for k in z.files}
            array_schema(original, query_fields + ('rank', 'candidates', 'filtered_count'), n, checks, label + ': source')
            for key, val in [('trained_snapshot', 4), ('tested_snapshot', t), ('seed', job['seed'])]:
                checks.require(original[key].shape == () and original[key].dtype.kind in 'iu' and
                               int(original[key]) == val, label + ': source scalar ' + key)
            checks.require(all(np.array_equal(data[k], original[k]) for k in query_fields),
                           label + ': exported query identity matches consumed source', n)
            checks.check(np.array_equal(data['native_replay'], original['rank']),
                         label + ': all native exported ranks equal bound source ranks', n)
            filtered = np.asarray([len(snap['filters'][int(h), int(r)]) for h, r, tail in q], dtype=np.int64)
            later = np.asarray([len(snapshots[4]['filters'][int(h), int(r)] -
                                    snap['filters'][int(h), int(r)]) for h, r, tail in q], dtype=np.int64)
            checks.check(np.all(original['candidates'] == snap['entities']), label + ': source historical candidates', n)
            checks.check(np.array_equal(original['filtered_count'], filtered), label + ': reconstructed historical filters', n)
            checks.check(np.array_equal(data['later_filtered_truths'], later), label + ': independently reconstructed later truths', n)
            rank_relations(data, snap['entities'], final_n, filtered, later, t, checks)
            scores = {arm: rank_metrics(data[arm]) for arm in ARMS}
            row = {'task_snapshot': t, 'queries': n, 'old_candidates': snap['entities'],
                   'final_candidates': final_n, 'scores': scores,
                   'strict_candidate_rank_worse_fraction': int(np.count_nonzero(data[ARMS[3]] > data[ARMS[2]])) / n,
                   'native_candidate_rank_worse_fraction': int(np.count_nonzero(data[ARMS[1]] > data[ARMS[0]])) / n,
                   'tie_fraction': int(np.count_nonzero(data[extra_fields[0]] > 0)) / n,
                   'queries_with_later_filtered_truths': int(np.count_nonzero(later > 0))}
            producer = tasks[t]
            checks.require(set(producer['scores']) == set(ARMS), label + ': metric arms')
            for key in ('queries', 'old_candidates', 'final_candidates', 'queries_with_later_filtered_truths'):
                checks.check(producer[key] == row[key], label + ': summary ' + key)
            for arm in ARMS:
                checks.require(set(producer['scores'][arm]) == set(METRICS), label + ': metric schema ' + arm)
                for metric in METRICS:
                    checks.check(close(producer['scores'][arm][metric], scores[arm][metric]),
                                 label + ': rank-derived ' + arm + '/' + metric)
            for key in ('strict_candidate_rank_worse_fraction', 'native_candidate_rank_worse_fraction', 'tie_fraction'):
                checks.check(close(producer[key], row[key]), label + ': derived ' + key)
            table.append(row)
            cohort_total += n
        overall = {arm: {metric: math.fsum(row['scores'][arm][metric] * row['queries'] for row in table) / cohort_total
                         for metric in METRICS} for arm in ARMS}
        checks.require(set(terminal['overall']) == set(ARMS), 'overall five-arm schema')
        for arm in ARMS:
            checks.require(set(terminal['overall'][arm]) == set(METRICS), 'overall metric schema:' + arm)
            for metric in METRICS:
                checks.check(close(terminal['overall'][arm][metric], overall[arm][metric]),
                             'query-weighted overall:' + arm + '/' + metric)
        checks.check(terminal['queries'] == cohort_total, 'terminal total cohort')
        checks.check(terminal['checks'] == 2 * cohort_total, 'producer count for strict and native gates')
        # A terminal is produced last. Ensure all reads still identify the same inputs/outputs.
        for name, digest in hashes.items():
            checks.check(sha(root / name) == digest, 'stable audit read:' + name)
        result.update(status='PASS' if not checks.failures else 'FAIL', dataset=job['dataset'],
                      queries=cohort_total, per_task=table, overall=overall,
                      artifact_sha256=hashes, selector='exact five final post_train inputs, pre_fwt excluded')
    except Exception as error:
        if not isinstance(error, InvalidArtifact):
            checks.failures.append(f'{type(error).__name__}: {error}')
        result['status'] = 'FAIL'
        result['artifact_sha256'] = hashes
    result.update(checks=checks.count, failures=checks.failures,
                  auditor_sha256=sha(Path(__file__)), numpy_version=np.__version__)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', action='append', help='registered native job ID; omit to inspect all four')
    parser.add_argument('--output', type=Path, help='optional report path, root-relative under plan/review')
    args = parser.parse_args()
    ids = args.job_id or [j['id'] for j in read_json(ROOT / 'configs/experiments/KGE-LIT-001-jobs.json')]
    registered = {j['id'] for j in read_json(ROOT / 'configs/experiments/KGE-LIT-001-jobs.json')}
    if any(j not in registered for j in ids) or len(set(ids)) != len(ids):
        parser.error('job IDs must be unique entries from the frozen master')
    report = {'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'CPU_DERIVED_NUMERICS_AND_PROVENANCE', 'jobs': [audit_job(ROOT, j) for j in ids]}
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        output = (ROOT / args.output).resolve()
        if not output.is_relative_to((ROOT / 'plan/review').resolve()):
            parser.error('--output must stay under plan/review; inputs/canonical are read-only')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')
    return 1 if any(j['status'] == 'FAIL' for j in report['jobs']) else \
        2 if any(j['status'] == 'WAITING' for j in report['jobs']) else 0


if __name__ == '__main__':
    sys.exit(main())
