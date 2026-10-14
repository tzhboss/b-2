import json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
lines=['# 080：双信号保护的跨种子优势未确认','',
    '固定077/078配方，在历史seed29/43初态上重新生成共享学习tape、TRAIN资格和保护缓存，完成12组训练，并对每组评价完整旧TRAIN与1024+1024开发VALID。无TEST、跨增量或未见数据确认。两个种子的blend新MRR都低于absolute；旧TRAIN保护略强，但没有复现seed17的联合优势。','',
    '| Seed | 方法 | 新filtered MRR | 旧VALID MRR | 旧VALID掉榜 | 旧TRAIN掉榜 | 旧→新保持比 | 新→旧保持比 | 两端新保持比 |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for seed,arms in s['metrics'].items():
    for arm,m in arms.items():
        nv=m['new_valid']['all']['filtered_train_truths']['mrr'];ov=m['old_valid']['all']['filtered_train_truths']['mrr']
        vd=m['old_valid']['drops']['filtered_train_truths']['drops'];td=m['old_train']['drops']['filtered_train_truths']['drops']
        r=m['directional_ratios']
        lines.append(f'| {seed} | {arm} | {nv:.9f} | {ov:.9f} | {vd} | {td} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} |')
lines+=['','seed29 blend旧→新保持比.966313，低于原开发.98要求；absolute为.976114，absolute075为.981258但旧TRAIN保护较弱。seed43 blend三个方向均超过.98，但新MRR仍低于absolute和单独mass；mass在本seed新MRR更高、旧TRAIN掉榜更少，不能把blend定为普遍更优。',
    '', '保护系数按同一RMS规则校准，每步各自限幅；normmatched控制在各自模型状态上进行，不是共轨迹Adam位移匹配。每seed使用自己teacher资格，不跨seed复用187470分母。','',
    f"接续程序墙时{s['total_seconds']:.3f}秒。首次启动在teacher评分前因缺CUBLAS_WORKSPACE_CONFIG退出，未训练任何模型；保留initialization-failure.log和原脚本，补环境后复用已经生成的seed29 tape，不重做该缓存。首次tape生成时长未保存，所以总初次+接续墙时未知，不能把322.575秒称完整端到端总费用。历史底座费用另为复用，非零。",'',
    '| Seed | tape生成/读取秒 | teacher TRAIN评分秒 | cohort构建秒 | 共享prefix10秒 | 校准秒 | 六臂后10轮训练合计秒 | 评价秒 | 本seed接续墙时秒 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for seed,c in s['costs'].items():
    t=c['tape'].get('seconds',c['tape'].get('reuse_seconds'))
    train=sum(x['seconds'] for x in c['training'].values())
    lines.append(f"| {seed} | {t:.3f} | {c['teacher_seconds']:.3f} | {c['cohort']['seconds']:.3f} | {c['prefix_seconds']:.3f} | {c['calibration_seconds']:.3f} | {train:.3f} | {c['evaluation_seconds']:.3f} | {c['total_seconds']:.3f} |")
lines+=['','完整raw/filtered、Hits与方向指标见summary.json；逐模型完整排名、采样cache和模型留远程。两个新seed是配方未选优的轨迹，VALID查询及seed历史资产都已有暴露；不能把本轮叫未见测试集或数据集独立确认。',
    '', '同时读取058已有TRAIN轨迹作时点诊断：平均softplus gap从.113822降至.025582，cohort掉榜比例从.354%升至2.808%；32/128尾部竞争比例与全局平均不同。该诊断没有新训练或评分，不证明尾部proxy可直接决定有效干预。',
    '', '下一步081固定两个竞争压力触发阈值，明确报告开启时点及相同时点等价控制，检验时点假设。当前结论是双信号组合有可用取舍，但其比简单absolute更优的主张未获跨种子支持。']
(root/'结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
