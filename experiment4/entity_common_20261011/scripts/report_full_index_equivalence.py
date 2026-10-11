"""Summarize all nine audited same-trajectory engineering replays."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
eid = 'KGE-ENTITY-COMMON-FULL-INDEX-EQUIVALENCE-034'
canonical_path = root/'experiments'/eid/'experiment.yaml'
canonical = yaml.safe_load(canonical_path.read_text())
assert canonical['status'] == 'completed_audited_artifact_checks_only' and not canonical['active_processes']
assert all(row['status'] == 'completed_audited' for row in canonical['jobs'].values())
for name, digest in canonical['frozen_input_sha256'].items():
    assert sha(root/name) == digest
config = json.loads((root/canonical['config']).read_text())
out = root/canonical['result_directory']
summary = json.loads((out/'summary.json').read_text())
assert summary['jobs_passed'] == summary['jobs_total'] == 9
rows = []
for item in summary['rows']:
    receipt, audit = item['result'], item['audit']
    assert receipt['status'] == 'PASS_ALL3753_CACHED_SOLVER_SOURCE_TRAJECTORY'
    assert audit['status'] == 'PASS_ALL3753_INDEPENDENT_TRACE_IDENTITY_AND_INHERITED_FULL_CPU_SCOPE' and not audit['failures']
    assert receipt['steps_matched'] == audit['steps_replayed'] == 3753
    assert receipt['terminal_weights_bitwise_equal'] and receipt['fixed_validation_identical'] and audit['exact_all_state_hashes'] and audit['exact_all_replay_arrays']
    directory = root/config['output_directory']/receipt['id']
    assert audit['receipt_sha256'] == sha(directory/'receipt.json')
    assert json.loads((directory/'receipt.json').read_text()) == receipt
    assert json.loads((directory/'audit.json').read_text()) == audit
    for name, digest in receipt['bound_source_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in receipt['bound_runtime_sha256'].items():
        assert sha(Path(name)) == digest
    original, cached = receipt['source_postprocessing_elapsed_s'], receipt['cached_postprocessing_elapsed_s']
    assert original > 0 and cached > 0
    rows.append(dict(id=receipt['id'], method=receipt['method'], seed=receipt['seed'],
                     original_postprocessing_seconds=original, cached_postprocessing_seconds=cached,
                     reduction_fraction=1-cached/original,
                     original_source_inclusive_seconds=receipt['source_inclusive_incremental_elapsed_s'],
                     cached_source_inclusive_seconds=receipt['cached_inclusive_incremental_elapsed_s']))
assert {row['id'] for row in rows} == {job['id'] for job in config['jobs']}
original = statistics.mean(row['original_postprocessing_seconds'] for row in rows)
cached = statistics.mean(row['cached_postprocessing_seconds'] for row in rows)
result = dict(status='PASS_ALL9_COMPLETE_TRAJECTORY_COST_DISCLOSURE', jobs=9, rows=rows,
              mean_original_postprocessing_seconds=original, mean_cached_postprocessing_seconds=cached,
              reduction_of_mean_postprocessing=1-cached/original, total_patch_replays=9*3753,
              source_summary_sha256=sha(out/'summary.json'), canonical_sha256=sha(canonical_path),
              source_utility_decisions_changed=False, limits=['One repetition per seed/device under concurrent conditions.',
                    'Full same-math postprocessing; source SGD reused, no new independent source training.',
                    'Exact source endpoint inherits030 TEST/CPU evidence; no new complete TEST rescore.',
                    'Engineering equivalence/efficiency does not reverse032 NO_GO or establish scientific utility.'])
lines = ['# 完整缓存索引重放的耗时与等价性', '',
         '9/9个任务独立审计通过，共33777次实体更新轨迹重放。全部保存的状态哈希和重放数组一致，终态权重及固定VALID结果一致。旧完整CPU保护与TEST证据来自同一原终态的030审计；没有新增全量TEST重评分。', '',
         '| 方案／seed | 原后处理秒 | 缓存后处理秒 | 耗时减少% |', '|---|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['method']}／{row['seed']} | {row['original_postprocessing_seconds']:.2f} | {row['cached_postprocessing_seconds']:.2f} | {100*row['reduction_fraction']:.2f} |")
lines += ['', f'九项平均后处理耗时从{original:.2f}秒降至{cached:.2f}秒，均值减少{100*(1-cached/original):.2f}%。逐项source-inclusive成本另存JSON。', '',
          '每个种子／设备仅一次重放，且并行环境不同；不能据此作显著性或等成本方法优势声明。这项优化改变实现耗时，保留原算法轨迹。原032三个方法均NO_GO的收益结论不变。', '',
          '本结果使用原V3索引求解器；后来通过评分控制的adaptive scorer未参与034，未被安装进运行中的037。']
assert not (out/'cost-summary.json').exists() and not (out/'report.md').exists()
(out/'cost-summary.json').write_text(json.dumps(result, indent=2))
(out/'report.md').write_text('\n'.join(lines), encoding='utf8')
print(result['status'], original, cached, 1-cached/original, flush=True)
