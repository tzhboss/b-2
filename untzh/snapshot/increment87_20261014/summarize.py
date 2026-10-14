import json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text());data=json.loads((root/'data.json').read_text())
lines=['# 087：snapshot1→2新增实体学习子任务','',
    '固定seed17/DistMult200，父状态为052普通0→1终态。旧11258实体/474关系冻结，新增1876实体normal std .12/生成器1087，fresh Adam .007。原snapshot2 TRAIN37214方向事实中21566涉及新实体，15648为旧—旧事实，本轮后者只用于truth过滤，不能称完整snapshot2新增事实学习。旧TRAIN累计260498方向事实，每epoch22批、20epoch440步。','',
    '| 方法 | 新filtered MRR | 旧VALID MRR | 旧VALID掉榜/资格 | 旧TRAIN掉榜/资格 | 旧→新保持比 | 新→旧保持比 | 两端新保持比 |',
    '|---|---:|---:|---|---|---:|---:|---:|']
for name,m in s['metrics'].items():
    r=m['directional_ratios'];nv=m['new_valid']['all']['filtered_train_truths']['mrr'];ov=m['old_valid']['all']['filtered_train_truths']['mrr']
    vd=m['old_valid']['drops']['filtered_train_truths'];td=m['old_train']['drops']['filtered_train_truths']
    lines.append(f"| {name} | {nv:.9f} | {ov:.9f} | {vd['drops']}/{vd['eligible']} | {td['drops']}/{td['eligible']} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} |")
lines+=['','| 配置 | 后10轮训练秒 | 后10轮步数 |', '|---|---:|---:|']
for name,c in s['costs']['training'].items():lines.append(f"| {name} | {c['seconds']:.3f} | {c['steps']} |")
lines += ['', '实际判断：absolute相对none新MRR+约.221%、旧TRAIN掉榜−约11.068%，旧MRR提高、旧VALID掉榜仍7。三个新方向均保留至少.98，但旧→新仍下降约1.94%，不是所有方向改善。blend旧→新.977936未达.98，仍没有支持跨增量的双信号优势。mass旧TRAIN保护更强、总体新MRR更低；normmatched/absolute075是相近取舍，尚无原创方法或统计确认。', '',f"程序接续墙时{s['costs']['total_seconds']:.3f}秒；teacher资格{s['costs']['teacher_seconds']:.3f}秒，cohort构建{s['costs']['cohort']['seconds']:.3f}秒，共享prefix10 {s['costs']['prefix_seconds']:.3f}秒，校准{s['costs']['calibration_seconds']:.3f}秒，评价{s['costs']['evaluation_seconds']:.3f}秒。首次准备完整墙时未保存，tape本次只读取复用，不能将接续墙时当首次准备+接续总费；历史父模型成本非零。",'',
    '首版compile括号错误在任何运行前修正；原37batch/740步假定在准备阶段断言失败，尚未训练，log/source保留。新动态循环根据实际22batch执行，未追加epoch或重训科学臂。', '',
    '新VALID筛选涉及新增实体后固定随机1024，旧VALID从0/1共抽1024；各组全部训练结束后才读取。所有六臂完整旧TRAIN/VALID raw/filtered、Hits及三个方向齐全，资格为该父模型自己的旧池Top10。', '',
    '这是此前配方未经该阶段选优的跨增量开发比较，但数据集相同、父状态/部分旧VALID已有暴露，且只有单seed；不能称独立数据集、统计显著、全新事实任务完成或全局安全。没有MEOR/replay原生最近邻，不证明原创性。无TEST、硬超时、独立CPU审计、SHA或归档。']
(root/'结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
