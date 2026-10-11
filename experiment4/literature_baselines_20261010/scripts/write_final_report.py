"""Finalize the readable report only after the last producer and audits are promoted."""
from pathlib import Path
import json
import datetime
import yaml

R = Path('/data/lc/tzh/experiment4/literature_baselines_20261010')
cp = yaml.safe_load((R / 'experiments/KGE-LIT-004/experiment.yaml').read_text())
assert cp['status'] == 'completed_audited_qualified' and not cp['active_processes']
a = json.loads((R / 'results/KGE-LIT-004/summary.json').read_text())
assert len(a['runs']) == 2 and a['cpu_failures'] == 0
old = (R / 'report.md').read_text()
backup = R / 'plan/review/report-before-finalization.md'
assert not backup.exists(), 'Preserve the existing finalization receipt; do not overwrite.'
backup.write_text(old)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
header = f'''# 远程文献基线复现报告（本轮完成，按用户要求先停）

完成时间：{now}。本轮选定的10路完整训练及审计均已终端；结果、失败记录和比较限制已归档，当前注册训练/审计无活动进程，不启动下一轮。数据、模型、训练、完整检查点及主要产物位于 `/data/lc/tzh/experiment4/literature_baselines_20261010`，文献来源 `/data/lc/tzh/kge_editing_literature_20261010`；本地仅保留报告与轻量证据副本。

IncDE四划分和FastKGE接近论文表值；STARK为公开实现结果，LKGE为验证集选点修订对照。SAGE原完整性FAIL保留。KGEditor两路均完整20epochs，固定公开实现数值得到带恢复限制的重推理支持；WN18RR成功率明显低于论文，两路逐项差异见表，不认证为论文性能复现。本轮完成的是选定文献的实验与审计收尾，尚未得到可声称优于原论文的新方法或投稿GO。

资源使用以实际进程收据为证：2026-10-10T11:49:23Z有7个已注册训练进程重叠，GPU0至6各一路（IncDE四路、FastKGE、STARK、SAGE）。这证明并行执行；不声称所有显卡每时刻均100%利用率。末期只剩原定KGEditor长任务，未为填满显卡而新增实验。证据见 `tables/parallel-execution-evidence.json`。

'''
body = old[old.index('## 1.'):]
start = body.index('## 5.')
end = body.index('## 6.')
old5 = body[start:end]
lkge = old5[old5.index('经典LKGE修订对照'):]
lines = ['## 5. KGEditor两路完成；LKGE修订对照完成', '',
    'KGEditor固定公开作者源码、HF模型和数据修订，seed0、完整20epochs。两路exit0，登记科学输入未变。论文指标仅作描述性对照；原论文训练预算的具体参数来自固定作者shell，而非声称正文完整声明了所有预算。', '',
    '| 数据集 | Succ@1实测/论文 | Succ@3实测/论文 | ER_roc实测/论文 | RK@3实测/论文 | RK_roc实测/论文 | optimizer steps | 总进程时间 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for x in a['runs']:
    c = x['article_comparison']
    pairs = [f'{v:.6f}/{w:.3f}' for v, w in zip(c['exposed_run'], c['article'])]
    lines.append('| ' + x['native_metrics']['config']['data_type'] + ' | ' +
        ' | '.join(pairs) + f" | {x['global_step']} | {x['outer_elapsed_s']:.2f}s |")
lines += ['',
    '两路与论文表值的描述性差异如上，WN18RR成功率明显低于论文。不能用较弱实测结果宣布强baseline已被击败，也不后验改变模型、训练预算或参数追分。论文Time .226/.232为编辑时间列，不能与这里完整训练进程耗时直接算加速。', '',
    '这是native exposed evaluation：edit_test同时用于选择best和最终评估，不是独立heldout。workers4替代作者32，原RMSprop(centered)保持；ER_roc/RK_roc是排名变化率，不是ROC AUC。来源34个文件的原blob/LFS哈希核验通过。']
for x in a['runs']:
    m = x['native_metrics']
    lines.append(f"{m['config']['data_type']}实际评估{m['edit']['count']}条edit、{m['locality']['count']}次locality暴露，只有{m['locality']['unique_source_rows']}个unique stable行；不能将重复暴露等同完整unique L-Test。")
lines += ['',
    f"CPU v2两路共{a['cpu_checks']}项检查0失败；两路selected epoch/step/monitor score绑定通过。每路固定16批、160例（32 edit+128 locality）、320条before/after记录，两路合计320例/640记录，独立CUDA重推理rank和Top3全部精确匹配。"]
for x in a['runs']:
    alias = x['native_restore_semantics']
    lines.append(f"{x['native_metrics']['config']['data_type']}最大target-logit误差{x['max_target_logit_error']:.9g}，序列化两组embedding最大差{alias['max_checkpoint_embedding_difference']:.9g}；参数hash/version及输入未变。")
lines += ['',
    '**恢复限制必须保留：**作者CPU构造时`.weight.data`共享存储，Lightning root load先加载ex_model、再加载ex_learner，后者覆盖基础embedding和tied decoder；最终effective模型不等于原ex_model序列化prefix。v3仅按源码复放该原生恢复顺序，支持此加载协议的固定数值，不能证明无恢复副作用的基础模型保留。WN两embedding有646653项不同，最大差.01770751；严格分离storage、各自恢复prefix的v2重推理实际FAIL，最大logit差.23060799，失败完整保留。没有修改训练、原最终导出、采样或容差。', '',
    '首次WN CPU审计的epoch断言FAIL保留：PL1.3.1 dump_checkpoint为恢复用途将epoch/global_step加1；raw epoch19实际对应训练epoch18。v2按实际框架源码修正解释，源码/hash/diff/fixture/版本化登记已归档。独立代码侧审因额度未完成，主控接管复核并补选择绑定，不声称完整独立代码复核通过，也未重建整个优化器轨迹或独立全量验证前向。', '',
    '`results/KGE-LIT-004`父canonical为completed_audited_qualified，审计PASS_NATIVE_NUMERICS_WITH_RESTORE_LIMITATIONS。两路CPU/重推理/选择绑定与修正依据已归档。先前WN单路summary中的parent_complete:false及当时FB仍运行说明是冻结历史快照；当前整体完成状态由父summary及canonical给出，不改写历史。原v1/v2失败和superseded状态保留。', '', lkge]
body = body[:start] + '\n'.join(lines) + '\n' + body[end:]
body += '''
## 8. 本轮交付和停止点

`tables/continual_kge_real_results.json/.csv`有8行真实持续KGE结果；`tables/kgeditor_exposed_real_results.json/.csv`有2行暴露编辑结果。`tables/literature-baseline-evidence-status.json`为完成时canonical/hash快照；`plan/review/completion-requirements-audit-final.json`核查10路训练收据、终端状态、归档字节及无活动进程。GPU峰值未测，表中为null，不填0。候选集合诊断另列第4节，既不当新方法，也不混入持续KGE或PLM编辑排名。

本轮到此停止。未启动下一轮、额外种子、模型网格或新数据实验；未来实验、下载和模型继续优先在远程 `/data/lc/tzh/experiment4`。当前结果支持可靠基线准备和协议风险判断，不自动满足“做出可相比原论文或优化方法”的更长期研究目标。
'''
(R / 'report.md').write_text(header + body)
p = R / 'plan/baseline-use-disposition.md'
s = p.read_text().replace('本轮两路20epoch及最终评估/审计仍待完全收尾。',
    '本轮两路20epoch、暴露评估及带恢复限制审计均已收尾；WN18RR论文成功率未匹配，两路描述性比较见最终报告。')
p.write_text(s)
print(json.dumps(dict(report=str(R / 'report.md'),utc=now,runs=len(a['runs']))))
