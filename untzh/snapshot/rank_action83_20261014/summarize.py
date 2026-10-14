import json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
lines=['# 083：一步排名探针只改变两步动作，未获得联合收益','',
    '三个新方向各256个TRAIN答案；prefix10过滤top64＋64合法均匀候选固定为128候选排名proxy，实际Adam提案下每方向RR至少为普通更新的1/1.001。370步中368步接受全部保护，2步改为一半；平均乘数.997297297。', '',
    '| 方法 | 新filtered MRR | 旧VALID掉榜 | 旧TRAIN掉榜 | 旧→新保持比 | 新→旧保持比 | 两端新保持比 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for name,m in s['metrics'].items():
    r=m['directional_ratios'];nv=m['new_valid']['all']['filtered_train_truths']['mrr'];vd=m['old_valid']['drops']['filtered_train_truths']['drops'];td=m['old_train']['drops']['filtered_train_truths']['drops']
    lines.append(f'| {name} | {nv:.9f} | {vd} | {td} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} |')
lines += ['', '相对原absolute，旧TRAIN只少1个掉榜，新MRR稍低；相对同平均强度控制，旧TRAIN少3个掉榜，新MRR也更低。变化太小，且没有共同改善，不能作为有效决策模块。', '',
    '| 方法 | 后10epoch运行秒 | 决策秒（包含） | 提案次数 | 平均乘数 |',
    '|---|---:|---:|---:|---:|']
for name,c in s['costs']['training'].items():lines.append(f"| {name} | {c['seconds']:.3f} | {c['selection_seconds_included']:.3f} | {c['proposals']} | {c['mean_multiplier']:.9f} |")
lines += ['',f"本轮阶段墙时{s['costs']['total_seconds']:.3f}秒；无TEST、硬超时、审计、SHA或归档。决策训练3.326秒，对应恒定乘数1.419秒；成本有增加，质量没有支持附加效用。历史底座/学习tape/cohort/prefix/logZ成本为只读复用且非零。",'',
    '有限TRAIN候选池排名只反映即时更新，未证明它能提前识别跨多个训练步累积的损伤；也未证明该探针覆盖实际VALID变化。下一假设应比较一段实际训练后的新旧收益与代价，而不是继续缩小这里的阈值追正。', '',
    '结果和完整raw/filtered/Hits/方向指标在summary.json，模型、候选cache、逐步尝试和完整排名留远程。历史保护方法仍作对照；不宣布原创方法或整体研究完成。']
(root/'结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
