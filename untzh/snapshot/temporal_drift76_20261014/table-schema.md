# 076时间诊断结果字段

- `evaluation/metrics.json`: 五臂各11年度；raw/filtered给出current、current_new_entity、historical，step_drift（同cohort前一年→当前）、cumulative_origin_drift（各原年→当前）；filtered额外anchor2005及冻结不变量。before/after/after_fixed_candidates均同组gold。
- `evaluation/<arm>-<year>.npz`: ids为prepared TEST方向索引，原始TEST先tail后inverse head；raw_full/filtered_full为当年候选，previous_candidates为上年候选，origin_candidates为各query原年候选。gold_score为屏蔽前gold分数；seconds为已完成评价计时。共享首年重用，无额外评分。
- `evaluation/annual-metrics.csv`: 每臂每片current/historical/anchor2005 MRR和漂移率；首年无historic，空值非0。2005 current_new_entity是全初始实体，不称增量实体。
- worsened_rate：rank_after>rank_before方向查询比例，为固定历史基准UFR proxy，非语义判错率。entity_interference_rate：新候选rank贡献>0比例，不等于独自导致总下降或MRR损失份额。
- old_top10_drop_rate：before<=10中after>10比例，与全部查询分母不同。
- parameter_only：总恶化、参数贡献>0、新候选=0；candidate_required_despite_nonpositive_parameter：总恶化、参数<=0、新候选>0；mixed_positive_parameter_and_candidates：总恶化、两贡献>0。三类和等于恶化数，不含真实世界合理变化标签。
- semantic_TVR：null；无ICEWS有效区间/撤回标签，不可当0/100%。缺失记录=未知。
- summary.json：2005固定8514方向cohort最终概要、current/historical及该路径MRR损失分解。
- training-costs.json：共享2005每架构计一次，其后五分支各十片。stage wall含保存/memory，非下载/写代码；training逐epoch日志记replay额外实例。
- evaluation-costs.json：共享首年0额外计时，实际评价wall单列。原data/prepare/mapping/memory/model远程；全部评价本地镜像。
