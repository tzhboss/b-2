"""Compile the measured diagnostic, without semantic relabeling or invented TVR."""
import json
from pathlib import Path
from prepare76 import write_json

ROOT=Path(__file__).resolve().parent


def main():
    results=json.loads((ROOT/'evaluation/metrics.json').read_text())
    stats=json.loads((ROOT/'data-statistics.json').read_text())
    traincost=json.loads((ROOT/'training-costs.json').read_text())
    evalcost=json.loads((ROOT/'evaluation-costs.json').read_text())
    arms=results['arms']
    lines=['# ICEWS05-15：时间持续学习中的历史排名漂移确实存在',
           '', '日期：2026-10-14。完整诊断已结束，五臂全部完成2005–2015顺序训练和逐年评价。不是新方法实验，也未运行独立审计、SHA门槛、归档或固定超时。',
           '', '**结论：固定历史日期和已知真答案，未来年份训练会使旧排名下降；旧候选参数更新与新增候选竞争都能观测。冻结旧参数的对照仍出现下降，明确支持候选干扰成分。ICEWS不提供事实撤回/有效区间，因此不能把下降比例当成经语义确认的错误率，也不能给出TVR或“合理30%/错误70%”。现象存在不等于研究问题的新颖性已成立。**',
           '', '## 数据、模型和评价范围',
           '', f"原始标准split：TRAIN {stats['split_facts']['train']:,}、VALID {stats['split_facts']['valid']:,}、TEST {stats['split_facts']['test']:,}事实；{stats['entities']:,}实体、{stats['relations']}关系、{stats['dates']:,}日期。日历2005–2015共11片。TRAIN与TEST、VALID与TEST的exact事实交集均0。",
           '', f"完整TRAIN；完整主cohort TEST共{stats['primary_eligible_directional_queries']:,}双向查询，其实体/关系/日期在各自原年有TRAIN表示。2005固定cohort为8,514查询（头尾预测来自4,257测试事实，双向不独立）。所有比例按方向查询计，不作独立样本推断。",
           '', '普通微调仅用当前年TRAIN。Replay每年从以前TRAIN均匀取4,096原始事实，额外1/4 batch回放；这是读取历史TRAIN构建的离线memory控制，非严格在线固定buffer实现，且增加计算。两架构各复用共同2005模型与Adagrad状态。TDistMult为简单日期乘法扩展，TComplEx按TKBC复数公式实现；固定seed17/128实数宽度、初始50epoch、增量每年15epoch，完整已见实体softmax CE+N3。未用VALID/TEST调参或选epoch；不是TIE/TKBC原生复现。',
           '', '历史查询的日期、关系和gold始终固定。raw/filtered都保存，filtered仅屏蔽同一精确日期的已记录其他真答案；未来不同日期的事件不能进入过去日期过滤。活动候选来自截至当前TRAIN出现过的实体。冷启动/heldout-only实体查询单列覆盖率，主cohort缺失查询不是正确答案或零损伤。',
           '', '## 同一批2005查询：从2005更新到2015',
           '', '| 实验臂 | 原年MRR | 2015 MRR | 2015原候选MRR | 排名恶化率 | 原Top10掉榜率 | 新候选影响率 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    summary={}
    for arm,values in arms.items():
        last=values['annual'][-1]['filtered'];a=last['anchor2005_drift']
        summary[arm]={'anchor2005':a,'final_current':last['current'],'final_historical':last['historical']}
        lines.append(f"| {arm} | {a['before']['mrr']:.6f} | {a['after']['mrr']:.6f} | {a['after_fixed_candidates']['mrr']:.6f} | {a['worsened_rate']:.2%} | {a['old_top10_drop_rate']:.2%} | {a['entity_interference_rate']:.2%} |")
    lines += ['', '排名恶化/新候选影响率分母都是8,514；原Top10掉榜率的分母为各架构原年Top10查询（TComplEx为6,162）。新候选影响率表示最终模型下至少一个原候选之外实体高于gold，不等于新增候选独自造成总下降。',
              '', '冻结旧参数的TComplEx：所有10个更新的历史查询旧候选rank完全不变；2005原候选MRR始终0.558634，扩张候选后最终0.522750。3,570/8,514查询恶化，200/6,162原Top10掉榜；无查询改善。旧参数漂移贡献精确0。该控制当前学习较差，不能当胜过Replay的方案。',
              '', '## 两种机制怎样分解',
              '', '`HRD总 = [r(新参数,旧候选)−r(旧参数,旧候选)] + [r(新参数,新候选)−r(新参数,旧候选)]`',
              '', '第一项为固定候选排名变化，第二项为新增候选rank贡献；对每个查询精确相加。逐年相对上一片、累计相对查询原片两个版本都保存。这是同一已训练模型上的排名分解，不是两个独立干预的因果效应。',
              '', '| TComplEx普通微调、2005→2015恶化查询类别 | 数量 | 占恶化查询 |',
              '|---|---:|---:|']
    a=arms['TComplEx_finetune']['annual'][-1]['filtered']['anchor2005_drift']
    labels={'parameter_only':'参数恶化、新候选贡献为0',
            'candidate_required_despite_nonpositive_parameter':'参数不恶化、需新增候选才出现总下降',
            'mixed_positive_parameter_and_candidates':'参数恶化且新候选有正贡献'}
    for key,value in a['worsening_categories'].items():
        lines.append(f"| {labels[key]} | {value:,} | {value/a['worsened_count']:.2%} |")
    param_mrr=a['before']['mrr']-a['after_fixed_candidates']['mrr']
    new_mrr=a['after_fixed_candidates']['mrr']-a['after']['mrr']
    total_mrr=a['before']['mrr']-a['after']['mrr']
    lines += ['',f"该臂MRR损失也可按同一反事实路径相加：固定候选参数更新损失{param_mrr:.6f}，新增候选额外损失{new_mrr:.6f}，合计{total_mrr:.6f}。此路径中新增候选约占MRR损失的{new_mrr/total_mrr:.2%}；不能把54.63%新候选影响率误写成54.63% MRR损失。",
              '', f"平均rank：{a['before']['mean_rank']:.2f}→{a['after']['mean_rank']:.2f}；固定原候选最终平均rank为{a['after_fixed_candidates']['mean_rank']:.2f}，虽然其平均rank改善，MRR仍下降。头部答案和长尾行为不同，平均rank及候选数量增长不能单独证明忘记量。",
              '', '“合理的时间变化”是事实语义/查询日期这一轴；“参数还是候选”是损伤机制这一轴。两轴不能强行划成三个互斥占比。未来当前日期CEO=B，可以和历史2010日期CEO=A同时正确；前者不授权修改后者。',
              '', '## 所有历史年份与当年学习',
              '', '| 实验臂 | 最终历史查询数 | 原年MRR（按同一cohort合并） | 最终历史MRR | 最终历史Hits10 | 最终当前MRR | 当前新增实体MRR |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm,values in arms.items():
        last=values['annual'][-1]['filtered'];c=last['cumulative_origin_drift']
        lines.append(f"| {arm} | {last['historical']['n']:,} | {c['before']['mrr']:.6f} | {last['historical']['mrr']:.6f} | {last['historical']['hits10']:.6f} | {last['current']['mrr']:.6f} | {last['current_new_entity']['mrr']:.6f} |")
    lines += ['', '最终历史为2005–2014，当前为2015；当前新增实体子集仅168方向查询，规模小，不作泛化优劣结论。Frozen控制历史原年性能也因冻结后新片学习受限而较弱，应以同一起点2005 cohort解释机制。Replay历史保留较好，但额外计算且稍损当前学习；不是全面胜出宣称。',
              '', '## 2005固定cohort逐年MRR',
              '', '| 加入截至年份 | TDistMult FT | TDistMult Replay | TComplEx FT | TComplEx Replay | TComplEx frozen |',
              '|---|---:|---:|---:|---:|---:|']
    names=list(arms)
    for j,year in enumerate(range(2005,2016)):
        numbers=[arms[arm]['annual'][j]['filtered']['anchor2005']['mrr'] for arm in names]
        lines.append('| '+str(year)+' | '+' | '.join(f'{v:.6f}' for v in numbers)+' |')
    cases=json.loads((ROOT/'evaluation/TComplEx_frozen_history-examples.json').read_text())
    case=cases[0]
    lines += ['', '## 一个可复查的纯新增候选案例',
              '', f"日期{case['date']}，查询实体`{case['query_entity']}`，关系`{case['relation']}`，gold=`{case['gold']}`。原rank={case['original_rank']:g}；2015参数下固定原候选仍rank={case['final_rank_original_candidates']:g}；扩张候选后rank={case['final_rank']:g}。新增候选贡献{case['new_candidate_contribution']:g}，参数贡献0。",
              '', f"最高竞争者包含`{case['top_filtered_competitors'][0]['entity']}`（TRAIN首次出现{case['top_filtered_competitors'][0]['train_first_year']}年）。这些竞争者不在该精确日期的已记录真答案中。首次在TRAIN出现晚不等于现实中出生晚；缺少记录也不等于能证明该候选现实无效，所以这里只认定基准外推干扰。全部选择案例和top10名称/分数见各臂examples.json；统计从完整cohort计算，案例未用于筛选结果。",
              '', '## 方向判断与已知工作的重合',
              '', '值得继续做时间语义诊断，但不宜宣称“首次发现时间持续学习会漂移”。[TIE（SIGIR2021）](https://github.com/JiapengWu/Time-Aware-Incremental-Embedding) 已研究增量时间KG，README明确有added/deleted facts、temporal regularization及historical replay；其公开代码已有`deleted_facts_mrr`、`both_positive_mrr`、`mean_relative_ranks`和跨时间评价矩阵。',
              '', 'TIE代码的relative ranks比较当前正答案与先前有效、当前不再记录的答案的倒数排名差，不是本轮同一固定日期gold的前后HRD。这说明定义有差异，不能单凭名称判断相同，也不能把相邻定义差异直接当创新。本轮未原生运行TIE或新候选修正方法；用户提出的四类完整baseline比较尚属于后续研究，不是这次存在性诊断已完成内容。',
              '', '下一步优先WIKI/YAGO的明确有效区间和功能关系：并排评价固定历史日期与滚动当前日期，标注持续有效/明确失效/未知，并在这些语义层内分解参数与候选。实体TRAIN首次出现不能替代现实有效性或出生日期；缺失边不能自动标失效。然后与TIE已有added/deleted评价及CKGE候选干扰协议做具体比较，才能判断新增诊断是否有独立信息。',
              '', '## 费用与文件',
              '', f"准备{stats['prepare_seconds']:.3f}秒；五臂训练阶段墙时{traincost['train_total_wall_seconds']:.3f}秒；统一评价墙时{evalcost['evaluation_wall_seconds']:.3f}秒。已测阶段和{stats['prepare_seconds']+traincost['train_total_wall_seconds']+evalcost['evaluation_wall_seconds']:.3f}秒，**不含下载、来源读取、编写代码、汇总和本地镜像，也不是完整部署成本**。共享首片按架构仅计算一次，Replay的batch额外费用保存在逐epoch文件。",
              '', '远程完整资产：`/data/lc/untzh/temporal_drift76_20261014`。原始数据在`data/src_data/ICEWS05-15`；52个实际checkpoint（两架构共享首片+五分支各10片）、memory身份和prepare数组留远程。五分支55个年度排名文件（共享首片评价复用）、metrics.json、annual-metrics.csv、案例、协议、source和日志已镜像本地。无待续训练或评价进程。',
              '', '数据来源：[TKBC作者仓库](https://github.com/facebookresearch/tkbc)，[作者下载脚本](https://github.com/facebookresearch/tkbc/blob/main/tkbc/scripts/download_data.sh)，[原始ICEWS数据DOI](https://doi.org/10.7910/DVN/28075)。TIE来源保存README/metrics_collection/evaluation_filter_global原文；arXiv摘要下载失败，不冒称已读完整TIE论文。',
              '', '本轮固定单seed、单数据、简单适配模型的开发诊断，不是多seed/多数据独立确认，也不验证原创方法收益。']
    (ROOT/'结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary['mrr_decomposition_TComplEx_finetune_anchor2005']={'parameter_loss':param_mrr,'new_candidate_loss':new_mrr,'total_loss':total_mrr,'new_candidate_share':new_mrr/total_mrr}
    summary['measured_stage_sum_seconds']=stats['prepare_seconds']+traincost['train_total_wall_seconds']+evalcost['evaluation_wall_seconds']
    write_json(ROOT/'summary.json',summary)
    print('SUMMARY_COMPLETE', len(lines),flush=True)


if __name__=='__main__':main()
