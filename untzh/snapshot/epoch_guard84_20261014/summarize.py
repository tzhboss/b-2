import json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
lines=['# 084：训练段决策的实际结果','',
    '固定seed17，前10轮普通学习后，每epoch比较同状态普通/.5/1倍absolute的37步真实Adam分支。选择仅使用1024旧TRAIN及三个方向各256新TRAIN的全候选排名，所有分支成本计入。local要求相对本段普通MRR至少.998，cumulative要求相对普通影子MRR至少.98。无TEST或未见验证。','',
    '| 方法 | 新filtered MRR | 旧VALID MRR | 旧VALID掉榜 | 旧TRAIN掉榜 | 旧→新保持比 | 新→旧保持比 | 两端新保持比 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,m in s['metrics'].items():
    r=m['directional_ratios'];nv=m['new_valid']['all']['filtered_train_truths']['mrr'];ov=m['old_valid']['all']['filtered_train_truths']['mrr']
    vd=m['old_valid']['drops']['filtered_train_truths']['drops'];td=m['old_train']['drops']['filtered_train_truths']['drops']
    lines.append(f'| {name} | {nv:.9f} | {ov:.9f} | {vd} | {td} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} |')
lines+=['','| 配置 | 10段乘数 | 平均乘数 | 运行秒 | 其中分支训练秒 | 其中排名探针秒 | 无可行预算段数 |',
    '|---|---|---:|---:|---:|---:|---:|']
for name,c in s['costs']['training'].items():
    lines.append(f"| {name} | {','.join(map(str,c['selected_sequence']))} | {c['mean_multiplier']:.3f} | {c['seconds']:.3f} | {c.get('branch_training_seconds_included',0):.3f} | {c.get('probe_seconds_included',0):.3f} | {c.get('no_feasible_epochs',0)} |")
lines += ['', '实际判断：local只在第17轮用半强度，均值.05。相同均值固定控制同时提高新MRR、旧MRR并减少旧TRAIN掉榜，local没有附加效用。cumulative均值.9，在第19轮跳过保护，比均值固定控制新MRR高但旧TRAIN掉榜也更多，仍是取舍；固定随机排列恰好与原序列相同，终态E/R逐位一致并复用rank，不能算独立确认。', '',f"准备{s['costs']['preparation']['seconds']:.3f}秒，普通影子轨迹+探针{s['costs']['shadow_seconds']:.3f}秒，评价{s['costs']['evaluation_seconds']:.3f}秒，本轮程序墙时{s['costs']['total_seconds']:.3f}秒。历史底座、prefix10、学习tape、资格缓存和校准来源仍是有成本复用。",'',
    '动作乘数平均一致不等于实际Adam位移或全轨迹累计保护norm完全一致。TRAIN探针的段后预算也不能证明VALID或全旧知识安全；本轮终态需要与固定均值和随机调度共同比较。保存raw/filtered、Hits及全部方向，不只展示总体新MRR。', '',
    '两个决策各30个37步分支，另10个普通影子分支及四个10段调度控制；共110段4070更新，其中被选分支直接保留，不重新训练。准备/分支/探针全写日志，无独立审计、SHA门槛、归档或硬超时。']
(root/'结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
