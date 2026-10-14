import json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
lines=['# 081：两个竞争压力触发规则未带来更好取舍','',
    '复用080两个seed全部准备缓存；至少5轮普通学习，128合法新候选超过旧gold的答案比例达到0.5%或1%时开启双信号保护。阈值在本轮VALID读取前固定；seed29的080失败已暴露，这是开发假设验证。','',
    '| Seed | 策略 | 开启轮次 | 新filtered MRR | 旧VALID掉榜 | 旧TRAIN掉榜 | 旧→新保持比 | 新→旧保持比 | 两端新保持比 |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for seed,arms in s['metrics'].items():
    for name,m in arms.items():
        onset=10 if name=='fixed10' else s['costs'][seed]['training'].get(name,{}).get('onset')
        onset='未保护' if onset is None else str(onset)
        r=m['directional_ratios'];nv=m['new_valid']['all']['filtered_train_truths']['mrr'];vd=m['old_valid']['drops']['filtered_train_truths']['drops'];td=m['old_train']['drops']['filtered_train_truths']['drops']
        lines.append(f'| {seed} | {name} | {onset} | {nv:.9f} | {vd} | {td} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} |')
lines+=['','seed29的0.5%在第5轮触发，与fixed5完全相同，直接复用模型和指标；seed43的0.5%在第6轮触发，整体新MRR、三个新方向和旧TRAIN掉榜都比fixed5更差。两个seed的1%始终未触发，监测普通学习轨迹结束后逐位核对与080 none模型一致，复用已保存指标，不重复评分。',
    '', '第5/6轮提前保护使旧→新保持比约.960，比固定第10轮更差。1%不触发保留了普通新学习，但同时失去保护收益。不能在此结果上继续补阈值追正，也不能把自适应开启本身视为有效模块。',
    '', '此压力比例不是掉榜概率：抽样未命中越位候选仍可能实际掉榜；不同旧rank容量也未进入这两个规则。单条轨迹上的单次触发与事后知道相同时点的固定策略等价；需要跨轨迹共同规则体现收益才能支持调度价值。',
    '', f"本轮接续墙时{s['total_seconds']:.3f}秒；080缓存/teacher/prefix10/控制及历史底座均为有成本的复用，不能把本轮读缓存费用当整个算法准备费用。",'',
    '| Seed | 共享prefix5秒 | 分支 | 实际运行秒 | 检测秒（已计入运行） | 校准秒（已计入运行） | 保护步数 | 复用终态 |',
    '|---|---:|---|---:|---:|---:|---:|---|']
for seed,c in s['costs'].items():
    for arm,t in c['training'].items():
        alias=t.get('reused_arm','none' if t['onset'] is None else '新保护终态')
        lines.append(f"| {seed} | {c['prefix_seconds']:.3f} | {arm} | {t['seconds']:.3f} | {t['detector_seconds']:.3f} | {t.get('calibration_seconds',0):.3f} | {t['protected_steps']} | {alias} |")
lines+=['','共6个预定义策略配置：3个不同新保护终态、2个为监测触发而运行的普通轨迹、1个触发时直接复用fixed5。fixed10/none来自080，所有模型不是独立重复；完整TRAIN评分只对3个不同新保护终态执行。',
    '', 'basic-reuse-check.json记录两个未触发普通轨迹与080 none逐位一致；另用ps现场确认080/081原PID均已退出。无独立CPU科学审计、SHA或重复排名。所有raw/filtered、Hits及三个方向指标完整保留。',
    '', '后续研究接口应从“当前风险高就保护”转向实际候选干预的收益与代价：在同一当前Adam状态下比较普通更新和保护更新，仅用TRAIN探针估计旧答案改善及三个新方向学习损失，再决定保护强度。这个接口尚未实现或验证；过去160短分支收益学习器和Euclidean梯度投影失败不能忽略，也不能把下一假设说成已有成功方法。']
(root/'结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
