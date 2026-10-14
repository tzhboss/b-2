import json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
lines=['# 085：固定训练段决策的跨seed比较','',
    '沿用084固定local(.998每段)/cumulative(.98影子)两规则，不改参数。seed29/43复用080 initial/prefix10/tape/cohort/teacher/calibration，每seed全部六配置提交后评价VALID；对每个不同终态完整评价旧TRAIN。新旧TRAIN探针全候选排名，所有影子/分支/探针成本计入。不是未见数据集或统计确认。','',
    '| Seed | 配置 | 新filtered MRR | 旧VALID MRR | 旧VALID掉榜 | 旧TRAIN掉榜 | 旧→新保持比 | 新→旧保持比 | 两端新保持比 |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for seed,arms in s['metrics'].items():
    for name,m in arms.items():
        r=m['directional_ratios'];nv=m['new_valid']['all']['filtered_train_truths']['mrr'];ov=m['old_valid']['all']['filtered_train_truths']['mrr']
        vd=m['old_valid']['drops']['filtered_train_truths']['drops'];td=m['old_train']['drops']['filtered_train_truths']['drops']
        lines.append(f'| {seed} | {name} | {nv:.9f} | {ov:.9f} | {vd} | {td} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} |')
lines+=['','| Seed | 配置 | 被选10段乘数 | 平均乘数 | 运行秒 | 无可行预算段数 |',
    '|---|---|---|---:|---:|---:|']
for seed,costs in s['costs'].items():
    for name,c in costs['training'].items():
        lines.append(f"| {seed} | {name} | {','.join(map(str,c['selected_sequence']))} | {c['mean_multiplier']:.3f} | {c['seconds']:.3f} | {c.get('no_feasible_epochs',0)} |")
lines += ['', '实际判断：seed29 cumulative全选1，和fixed absolute完全相同；TRAIN旧→新探针保持.999796但VALID只有.976114，预算未泛化。seed29 local比均值控制新MRR高但旧TRAIN掉榜更多；seed43 local相对均值控制新MRR/旧MRR/旧TRAIN掉榜有小幅改善，三个方向保持均>.98，但跨seed不一致。seed43 cumulative比均值控制新MRR低、旧TRAIN掉榜少，仍是取舍。三个seed没有支持固定规则稳定提高联合效用，不能只挑seed43声明方法成立。', '',f"本轮程序墙时{s['total_seconds']:.3f}秒；具体准备、影子、分支、探针和评价秒数见summary.json。每seed两个决策各30分支，另10普通影子段、四10段控制，共110段4070实际Adam更新；两seed共8140步。没有另行重训被选分支。",'',
    '重复模型只有E/R逐位相同才复用历史/本轮排名，别名在evaluation/reused-equal-models.json。候选/探针缓存与模型均留远程。raw/filtered及Hits完整保留，不能仅用总体MRR或单seed漂亮点宣称成功。', '',
    '固定TRAIN探针预算与VALID终态不是同一指标；段后预算可以通过而新方向VALID保持比不达.98，诊断见probe-vs-endpoint.json。平均乘数匹配不保证Adam位移匹配。随机调度仅一个固定排列，若碰巧同原序列，则是同轨迹复用，不是独立重复。', '',
    '完整科学判断要同时看相同平均乘数控制、旧MRR/掉榜和三个新方向；没有多增量/未见TEST/原生最近邻验证。本轮无硬超时、CPU审计、SHA或归档。']
(root/'结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
