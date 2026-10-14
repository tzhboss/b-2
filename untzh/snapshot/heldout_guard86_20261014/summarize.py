import json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
lines=['# 086：内部留出探针的匹配比较','',
    '固定seed17，留出3721原始新TRAIN事实及其逆向（7442方向事实），所有新臂只对29772保留方向事实更新；仍20epoch740步。保留全部原训练活动实体，三个方向各256决策探针只取内部留出，完整候选TRAIN filtering包含原全部TRAIN。本轮主要对照为相同80%训练数据的matched_none/matched_absolute，不能与历史100%学习量ordinary混比。','',
    '| 方法 | 新filtered MRR | 旧VALID MRR | 旧VALID掉榜 | 旧TRAIN掉榜 | 旧→新保持比 | 新→旧保持比 | 两端新保持比 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,m in s['metrics'].items():
    r=m['directional_ratios'];nv=m['new_valid']['all']['filtered_train_truths']['mrr'];ov=m['old_valid']['all']['filtered_train_truths']['mrr']
    vd=m['old_valid']['drops']['filtered_train_truths']['drops'];td=m['old_train']['drops']['filtered_train_truths']['drops']
    lines.append(f'| {name} | {nv:.9f} | {ov:.9f} | {vd} | {td} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} |')
lines+=['','| 配置 | 10段保护乘数 | 平均乘数 | 运行秒 | 无可行预算段数 |',
    '|---|---|---:|---:|---:|']
for name,c in s['costs']['training'].items():
    lines.append(f"| {name} | {','.join(map(str,c['selected_sequence']))} | {c['mean_multiplier']:.3f} | {c['seconds']:.3f} | {c.get('no_feasible_epochs',0)} |")
lines += ['', '实际判断：local新MRR高于同均值控制，但旧TRAIN掉榜3941>3929、旧MRR略低，仍是取舍；cumulative被matched_absolute同时在新MRR、旧MRR和旧TRAIN掉榜上超过，VALID掉榜都6。内部留出未建立更好联合效用。80%matched_none新MRR较历史100%普通降低约7.39%，不能忽略为决策保留事实的学习代价。', '',f"程序接续墙时{s['costs']['total_seconds']:.3f}秒；首次准备因旧probe构造器要求完整首epoch索引而失败，未训练模型，修成用原完整tape构建随后丢弃的旧探针，只用内部留出新探针做决策。划分及过滤tape直接复用，原失败源码/log保留。首次准备的完整墙时未保存，因此接续时间不能称从首次启动计的完整总费。",'',
    '学习样本744280→595440方向事实draw；历史100%普通学习新VALID MRR .150805134，当前80%matched_none的指标要同时报告，不能把少学习数据造成的旧伤减少当保护创新。训练事实和逆向成对排除以避免反向泄漏；剩余学习数据仍覆盖原训练所有活动实体。', '',
    '探针已由参加更新的TRAIN改为内部留出事实，但这个变化是否帮助动作决策必须看匹配均值与随机调度控制。内部留出会反复参与选择动作，不等于未见TEST或独立确认。一般预算比率依旧不能当VALID终态或全旧知识安全保证。','',
    '全部8模型提交后才读取开发VALID；所有不同模型完整旧TRAIN排名。raw/filtered、Hits及三个方向保留，完全相同模型才能复用rank。无固定超时、独立CPU审计、SHA或归档。']
(root/'结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
