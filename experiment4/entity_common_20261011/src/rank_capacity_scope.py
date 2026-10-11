"""Explicit TRAIN-certified capacity exemptions from an audited reference scope."""
import json, time
from pathlib import Path
import numpy as np
from common_training import sha


def filtered_scope(keys, refs, counts, exempt_keys):
    keep_keys = np.asarray([tuple(map(int, q)) not in exempt_keys for q in keys])
    kept = refs[keep_keys[refs[:,0]]].copy(); used = np.unique(kept[:,0])
    assert len(kept) and len(used)
    mapping = np.full(len(keys), -1, dtype=np.int64); mapping[used] = np.arange(len(used))
    kept[:,0] = mapping[kept[:,0]]
    return keys[used], kept, counts[used], used


def bound_sources(root, cfg, seed):
    source = root/cfg['source_directory']/f'shared-rank-scope-s{seed}'
    diagnostic = root/cfg['diagnostic_directory']
    receipt = json.loads((source/'receipt.json').read_text()); audit = json.loads((source/'audit.json').read_text())
    evidence = json.loads((diagnostic/'summary.json').read_text()); gpu = json.loads((diagnostic/'native-gpu-audit.json').read_text())
    split = json.loads((root/cfg['split_audit']).read_text())
    assert audit['status'].startswith('PASS') and not audit['failures']
    assert audit['receipt_sha256'] == sha(source/'receipt.json')
    assert receipt['output_sha256']['scope.npz'] == sha(source/'scope.npz')
    assert not evidence['test_read'] and not evidence['valid_read']
    assert evidence['output_sha256'] == sha(diagnostic/'certificates.npz')
    assert gpu['status'] == 'PASS_FIXED_CERTIFICATE_NATIVE_GEMM_CHECK' and not gpu['tf32']
    assert gpu['diagnostic_summary_sha256'] == sha(diagnostic/'summary.json')
    assert gpu['certificate_arrays_sha256'] == sha(diagnostic/'certificates.npz')
    assert split['status'] == 'PASS_GLOBAL_SPLIT_DISJOINTNESS'
    assert all(n == 0 for row in split['rows'] for n in row['overlap'].values())
    bound = dict(evidence['input_sha256']); bound.update(receipt['input_sha256']); bound.update(split['input_sha256'])
    for p in [source/'scope.npz', source/'receipt.json', source/'audit.json', diagnostic/'summary.json',
              diagnostic/'certificates.npz', diagnostic/'native-gpu-audit.json', root/cfg['split_audit']]: bound[str(p)] = sha(p)
    for name, digest in bound.items(): assert sha(Path(name)) == digest, name
    return source, diagnostic, bound


def prepare(root, cfg, job):
    started = time.monotonic(); source, diagnostic, bound = bound_sources(root, cfg, job['seed'])
    with np.load(source/'scope.npz') as z: arrays = {name:z[name].copy() for name in z.files}
    with np.load(diagnostic/'certificates.npz') as z:
        exempt = z[f"s{job['seed']}_query_keys"].copy(); ids = z[f"s{job['seed']}_query_ids"].copy()
    assert np.array_equal(arrays['source_keys'][ids], exempt)
    assert arrays['affected_mask'][ids].all()
    exempt_set = set(map(tuple, exempt.tolist())); removed = {}
    for mode in ['full', 'scoped']:
        keys, refs = arrays[mode+'_keys'], arrays[mode+'_refs']
        kk, rr, cc, used = filtered_scope(keys, refs, arrays[mode+'_counts'], exempt_set)
        removed[mode] = dict(query_keys=len(keys)-len(kk), reference_members=len(refs)-len(rr))
        arrays[mode+'_keys'] = kk; arrays[mode+'_refs'] = rr; arrays[mode+'_counts'] = cc
        arrays[mode+'_source_query_ids'] = arrays[mode+'_source_query_ids'][used]
    arrays['exempted_query_keys'] = exempt; arrays['exempted_source_query_ids'] = ids
    out = root/cfg['output_directory']/job['id']; out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'scope.npz', **arrays)
    receipt = dict(id=job['id'], seed=job['seed'], test_read=False, valid_read=False,
                   split_integrity_evidence_reads_valid_test=True, elapsed_s=time.monotonic()-started,
                   input_sha256=bound, output_sha256={'scope.npz':sha(out/'scope.npz')},
                   exempted_queries=len(exempt), removed=removed,
                   scope='Explicit removal of reference members only at TRAIN-certified saturated affected queries',
                   implementation_sha256=sha(Path(__file__)),
                   limits=['Changes TRUE and Prediction preservation at certified TRAIN query keys.',
                           'No blanket old-knowledge preservation or old MRR guarantee.',
                           'Outside certified keys, every source reference is retained exactly.',
                           'Structural preparation only; trained utility requires separate unchanged gates.'])
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)); print('CAPACITY_SCOPE_PREPARED', job['id'], removed, flush=True)


def audit(root, cfg, job):
    started = time.monotonic(); source, diagnostic, bound = bound_sources(root, cfg, job['seed'])
    out = root/cfg['output_directory']/job['id']; receipt = json.loads((out/'receipt.json').read_text())
    assert not receipt['test_read'] and not receipt['valid_read'] and receipt['input_sha256'] == bound
    assert sha(out/'scope.npz') == receipt['output_sha256']['scope.npz']
    checked = 0; removed = {}
    with np.load(source/'scope.npz') as src, np.load(out/'scope.npz') as result, np.load(diagnostic/'certificates.npz') as certificates:
        exempt = certificates[f"s{job['seed']}_query_keys"]
        assert np.array_equal(result['exempted_query_keys'], exempt)
        certificate_ids = certificates[f"s{job['seed']}_query_ids"]
        assert np.array_equal(result['exempted_source_query_ids'], certificate_ids)
        assert np.array_equal(src['source_keys'][certificate_ids], exempt)
        assert src['affected_mask'][certificate_ids].all()
        independent_exempt = {tuple(map(int, row)) for row in exempt}
        for mode in ['full', 'scoped']:
            source_keys = src[mode+'_keys']; source_refs = src[mode+'_refs']
            # Reconstruct actual (head,relation,gold) reference identities,
            # independently of the producer's masking/compaction function.
            expected = [(int(source_keys[q,0]), int(source_keys[q,1]), int(g))
                        for q,g in source_refs if tuple(map(int,source_keys[q])) not in independent_exempt]
            keys = result[mode+'_keys']; refs = result[mode+'_refs']
            actual = [(int(keys[q,0]), int(keys[q,1]), int(g)) for q,g in refs]
            assert expected == actual and len(set(actual)) == len(actual)
            assert not any(tuple(map(int,k)) in independent_exempt for k in keys)
            source_index = {tuple(map(int,k)):i for i,k in enumerate(source_keys)}
            used = np.asarray([source_index[tuple(map(int,k))] for k in keys])
            assert np.array_equal(result[mode+'_counts'], src[mode+'_counts'][used])
            assert np.array_equal(result[mode+'_source_query_ids'], src[mode+'_source_query_ids'][used])
            assert (result[mode+'_counts'] < cfg['k']).all()
            removed[mode] = dict(query_keys=len(source_keys)-len(keys), reference_members=len(source_refs)-len(refs))
            checked += len(actual)
        for name in src.files:
            if not name.startswith(('full_', 'scoped_')): assert np.array_equal(src[name], result[name]), name
    assert removed == receipt['removed'] and receipt['exempted_queries'] == len(exempt)
    result = dict(status='PASS_ALL_RETAINED_AND_EXEMPTED_REFERENCE_IDENTITIES', failures=[],
                  references_independently_reconstructed=checked, exempted_queries=len(exempt), removed=removed,
                  test_read=False, valid_read=False, receipt_sha256=sha(out/'receipt.json'),
                  elapsed_s=time.monotonic()-started,
                  limits=['Independent reference identity replay; score certification inherits bound TRAIN diagnostic and native GPU evidence.',
                          'No model update or utility result. Separate global split-integrity evidence read VALID/TEST.'])
    (out/'audit.json').write_text(json.dumps(result, indent=2)); print('CAPACITY_SCOPE_AUDIT_PASS', job['id'], flush=True)
