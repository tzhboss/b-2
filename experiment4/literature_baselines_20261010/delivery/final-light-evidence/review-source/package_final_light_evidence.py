"""Copy only completed light artifacts for local reading and checksum verification."""
from pathlib import Path
import json
import hashlib
import shutil
import datetime

R = Path('/data/lc/tzh/experiment4/literature_baselines_20261010')
review = json.loads((R / 'plan/review/completion-requirements-audit-final.json').read_text())
assert review['status'] == 'ROUND_COMPLETE_WITH_RETAINED_LIMITATIONS_STOP_BEFORE_NEXT_ROUND'
assert review['complete_registered_native_training_runs'] == 10
assert review['no_registered_native_or_current_audit_process_alive']
assert '本轮完成，按用户要求先停' in (R / 'report.md').read_text()
dest = R / 'delivery/final-light-evidence'
assert not dest.exists(), 'Preserve previously packaged evidence; never overwrite a release silently.'
dest.mkdir(parents=True)
shutil.copy2(R / 'report.md', dest / 'report.md')
for n in range(1, 7):
    name = 'KGE-LIT-%03d' % n
    shutil.copytree(R / 'results' / name, dest / 'evidence' / name)
for name in ['continual_kge_real_results.json', 'continual_kge_real_results.csv',
             'kgeditor_exposed_real_results.json', 'kgeditor_exposed_real_results.csv',
             'literature-baseline-evidence-status.json', 'parallel-execution-evidence.json',
             'KGE-LIT-004-completed-qualified.json', 'incde-paper-targets.json',
             'incde-native-training-times.json']:
    p = dest / 'tables' / name
    p.parent.mkdir(exist_ok=True)
    shutil.copy2(R / 'tables' / name, p)
review_files = ['completion-requirements-audit-final.json',
                'KGE-LIT-002-final-current-byte-check.json',
                'pre-final-completed-release-check.json',
                'final-delivery-script-registration.json',
                'kgeditor-paper-public-config-check.md',
                'kgeditor-auditor-controller-review.md']
for name in review_files:
    p = dest / 'review-source' / name
    p.parent.mkdir(exist_ok=True)
    shutil.copy2(R / 'plan/review' / name, p)
for name in ['baseline-use-disposition.md', 'selection-and-comparison.md', 'threat-review.md']:
    shutil.copy2(R / 'plan' / name, dest / 'review-source' / name)
for relative in ['scripts/finalize_round_delivery.py', 'scripts/write_final_report.py',
                 'scripts/promote_completed_kgeditor.py', 'scripts/package_final_light_evidence.py',
                 'code/audit_kgeditor_native_v2.py',
                 'code/replay_kgeditor_samples_v3_native_alias.py',
                 'code/audit_kgeditor_selection_binding_v2.py',
                 'scripts/build_result_tables.py']:
    shutil.copy2(R / relative, dest / 'review-source' / Path(relative).name)
(dest / 'README.md').write_text('''# KGE 文献基线：本轮完成，先停

主目录：`/data/lc/tzh/experiment4/literature_baselines_20261010`。文献目录：`/data/lc/tzh/kge_editing_literature_20261010`。数据、模型、训练、完整检查点和逐查询产物在远程；本目录为报告和轻量审计副本。

本轮选定的10路完整训练、数值审计和结果归档已终端，不启动下一轮。`report.md`为最终报告；`tables/`保存分协议真实结果；`evidence/`保存原审计状态、失败和归档清单；`review-source/`保存复核说明与实际审计/交付代码。

- IncDE四划分与FastKGE接近论文表值，单种子不能说明改善。
- STARK为公开实现结果，完整论文机制对应仍有限。
- SAGE原producer完整性FAIL保留，仅作带限制数值证据。
- LKGE采用valid.txt选点修订，区别于作者test.txt选点。
- Pons动机候选集合诊断是已知机制的独立实现，不是新发现。
- KGEditor两路均完整20epochs、各160例抽样重推理精确匹配，限定为原生共享embedding恢复及暴露评估数值。WN18RR成功率明显低于论文；完整表值见报告。原epoch解释和分离storage重推理失败保留。

KGEditor早期WN组件summary中的parent_complete:false和当时FB仍运行说明为冻结历史快照；当前父summary/canonical已整体完成。原历史失败及限制不改写。

`local-delivery-sha256.json`仅核验传输字节，不能替代科学审计。完成本地核验后另存`local-transfer-verification-final.json`。缺测GPU峰值为null；不同协议分表，不作为混合SOTA排名。尚未获得新方法收益、已确认新颖性或投稿GO。

今后实验、数据和模型下载继续优先在远程 `/data/lc/tzh/experiment4`。本轮按用户要求先停。
''')
files = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(dest.rglob('*')) if p.is_file()}
(dest / 'local-delivery-sha256.json').write_text(json.dumps(dict(
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Light completed artifacts only; byte-transfer inventory, not a new scientific audit',
    files=files), ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(directory=str(dest),files=len(files),bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()))))
