"""Bound complete TRAIN diagnostic evidence and disclose the mechanism limits."""
import json
from pathlib import Path

import yaml

from diagnose_rank_conflicts39 import sha


root=Path(__file__).resolve().parents[1]
state=yaml.safe_load((root/'experiments/KGE-ENTITY-COMMON-RANK-CONFLICT-DIAGNOSTIC-039/experiment.yaml').read_text())
assert state['status']=='completed_audited_train_diagnostic_only' and not state['active_processes']
assert all(j['status']=='completed_audited' for j in state['jobs'].values())
for name,digest in state['frozen_input_sha256'].items():
    assert sha(root/name)==digest,name
folder=root/state['result_directory']
raw=json.loads((folder/'summary.json').read_text())
assert raw['audited_seeds']==raw['total_seeds']==3
assert [r['seed'] for r in raw['rows']]==[17,29,43]
assert not raw['scientific_utility_asserted']
lines=['# 完整TRAIN保护边界诊断039','',
       '仅原Full／Scoped和容量Full／Scoped终态的TRAIN保护查询。模型不更新，VALID/TEST查询不读取；本报告不是新方法收益或创新性认证。','',
       '| 方法／seed | 源目标替换受限实体 | 至少一项旧不可动容量限制 | 只受其他新候选限制 | 行仍不同但终态替换可行 |',
       '|---|---:|---:|---:|---:|']
records=[]
cells=0
bound={}
for seed in raw['rows']:
    audit=seed['audit'];result=seed['result']
    assert not audit['failures'] and audit['cpu_gpu_score_bits_exact']
    artifact=root/'artifacts/rank-conflict39'/f"s{seed['seed']}"
    assert audit['source_summary_sha256']==sha(artifact/'summary.json')
    assert audit['fixture_sha256']==sha(artifact/'fixtures.npz')==result['fixture_sha256']
    assert not result['test_read'] and not result['valid_read'] and not result['weights_updated']
    for name,digest in result['input_sha256'].items():
        assert sha(Path(name))==digest,name
    for filename in ('summary.json','audit.json','fixtures.npz'):
        bound[str((artifact/filename).relative_to(root))]=sha(artifact/filename)
    cells+=audit['score_cells_replayed']
    for row in result['rows']:
        assert row['seed']==seed['seed']
        assert row['blocked_target_entities']==row['immutable_old_blocked_entities']+row['blocked_only_by_new_slot_entities']
        assert row['endpoint_rows_different_from_source']==row['blocked_target_entities']+row['different_but_endpoint_source_replacement_feasible']
        item={k:row[k] for k in ('method','seed','query_groups','references',
          'blocked_target_entities','immutable_old_blocked_entities','blocked_only_by_new_slot_entities',
          'different_but_endpoint_source_replacement_feasible','blocking_pairs','immutable_old_count_histogram')}
        records.append(item)
        lines.append(f"| {row['method']}／{row['seed']} | {row['blocked_target_entities']} | {row['immutable_old_blocked_entities']} | {row['blocked_only_by_new_slot_entities']} | {row['different_but_endpoint_source_replacement_feasible']} |")
lines+=['',
        '旧容量限制指最低保护成员上方已有9个不可动旧候选：源目标跨过该边界会使此成员rank>=11。即使其它新候选全部降到边界以下，这一项仍违规。因此单纯重排新实体或只协调其它新实体不能解除这项固定边界阻碍。', '',
        '原Full受限实体中99.3%–100%有至少一项此旧容量限制；原Scoped为95.4%–99.1%。容量豁免后的比例仍为94.8%–100%。这说明新候选slot竞争不是这些终态源目标被拒的主要必要约束，但不提供换合同或联合更新的收益保证。', '',
        '每种方案还有2–7个新行虽不同于源目标，却在当前终态可单行恢复为源目标。该数量只表示可复查的局部机会；没有执行第二遍更新，也没有证明哪种顺序最好。', '',
        '很多最低保护成员不在旧TRAIN真事实集合内。“未标为TRAIN真”不是“错误知识”；原QPTrue去掉全部预测保护已在013旧损失门槛失败。不能将本诊断当作许可删除全部预测保护，不能复活原固定64旧行联合方案的032负结论。', '',
        f'独立CPU审计共重算{cells}个分数，覆盖每seed四policy各32个完整查询（共384行），包括完整7505旧候选、3753当前候选及3753源目标候选。评分位值、cut、rank计数和全部3753个单行替换资格在这些fixture上精确一致。完整TRAIN范围为GPU扫描，未声称每个GPU分数均经CPU独立重算。', '',
        '下一机制的候选范围：预测参考合同的质量／容量分配，或者面向实际保护瓶颈的有限旧参数协调。任何合同变化需要对原TRUE／预测成员分别披露损失；任何参数调整仍需完整scope和同任务方向收益检验。现有证据不支持扩大半径、仅调顺序或重跑原TRUE-only／原joint来宣布优化。']
assert cells==5764224
context=dict(status='PASS_COMPLETE_TRAIN_SLOT_DIAGNOSIS_DISCLOSURE',rows=records,
             cpu_score_cells_replayed=cells, source_sha256=bound,
             diagnostic_summary_sha256=sha(folder/'summary.json'), scientific_utility_asserted=False,
             weights_updated=False, original_gates_changed=False,
             next_mechanism_requires_new_controlled_validation=True)
assert not (folder/'report.md').exists() and not (folder/'mechanism-disclosure.json').exists()
(folder/'report.md').write_text('\n'.join(lines),encoding='utf8')
(folder/'mechanism-disclosure.json').write_text(json.dumps(context,indent=2))
print(context['status'],cells,flush=True)
