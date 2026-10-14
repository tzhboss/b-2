import json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
lines=['# 082：一步softplus代价决策没有改变保护动作','',
    '固定seed17真实Adam提案、三个方向各256固定TRAIN探针，要求一步损失至多为普通提案的1.001倍且旧竞争proxy改善。absolute_guard和blend_guard各370步全部接受乘数1，因此恒定平均强度控制也为1；没有形成有效的选择性干预。', '',
    '| 方法 | 新filtered MRR | 旧VALID掉榜 | 旧TRAIN掉榜 | 旧→新保持比 | 新→旧保持比 | 两端新保持比 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for name,m in s['metrics'].items():
    r=m['directional_ratios'];nv=m['new_valid']['all']['filtered_train_truths']['mrr'];vd=m['old_valid']['drops']['filtered_train_truths']['drops'];td=m['old_train']['drops']['filtered_train_truths']['drops']
    lines.append(f'| {name} | {nv:.9f} | {vd} | {td} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} |')
lines+=['', '决策组与对应固定保护的完整已保存排名指标相同；这不是额外方法收益。一步局部loss预算可全部通过，而终态新MRR仍有损失，说明本设定下该proxy和阈值不足以控制排名退化。不证明所有TRAIN探针或实际Adam决策都无效。', '',
    '| 方法 | 后10epoch运行秒 | 其中决策秒 | 实际提案次数 | 平均保护乘数 |',
    '|---|---:|---:|---:|---:|']
for name,c in s['costs']['training'].items():lines.append(f"| {name} | {c['seconds']:.3f} | {c['selection_seconds_included']:.3f} | {c['proposals']} | {c['mean_multiplier']:.6f} |")
lines += ['',f"本轮阶段墙时{s['costs']['total_seconds']:.3f}秒，含四臂完整TRAIN评价；复用058/052/077准备和prefix费用仍非零。无TEST、独立审计、SHA或硬超时。两个决策器增加约1.5秒候选评估而未改变动作，当前没有部署收益。",'',
    '083仅把新学习代价换成有限候选池TRAIN排名，不增加学习器。当前所有结果均为已暴露开发配置，未作统计显著或全局安全声明。']
(root/'结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
