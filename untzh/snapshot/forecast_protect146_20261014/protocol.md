# 146：固定抽样内的前瞻Anchor触发，单次P2证伪

144/145同公式在FB17/43/29的VALID晚期新增掉榜排序出现重复信号，尚无校准或保护收益优势。本轮检验该信号能否指导实际保护；不是query池搜索，不改既存cohort/原每步32查询抽样。只用144/145已冻结的oldTRAIN预测，官方VALID/reference gold不进入动作。

三个既存FB DistMult200父、121zero初态/prefix10含Adam、原各自学习tape/负例、固定rho.1、后段220步、旧E/R冻结。复用TRAIN query-space、Replay缓存；无新父/前缀训练、初值或penalty搜索。六臂：普通、固定Anchor、固定Replay、forecast_anchor、current_anchor、random_matched。

前瞻触发严格用stationary_mean_rank>10（Top10任务的自然边界，不优化阈值），当前触发prefix rank>10；没有“高风险即必要”的先验保证。每步原32query保持不变，只将该query的Anchor损失乘固定0/1；随机对照在同32个查询内用固定rng1460017均匀选取与前瞻完全相同数量，保持逐步触发次数相同。均值用于决策，情景只保留诊断，不叠加模块。WN TRAIN风险无正例且原参考盲区已知，不将WN全不保护当成功，不做此轮WN联动。

所有六臂都计算学习/fullAnchor/forecastAnchor/currentAnchor/randomAnchor/Replay梯度、自身ordinary影子、同完整queryspace投影，比较相同计算包络；仅普通/Replay不接受投影，三个稀疏及固定Anchor接受累计相对ordinary的投影。每非零步保护梯度范数为学习的.1，零触发不保护；累计强度、投影位移、查询触发比例另报，不能以同rho当同累计强度。固定Anchor/普通如参数等126/121复用rank，非独立确认。

18后段模型全部提交后评价完整旧TRAIN/旧VALID、新VALID2786、raw/filtered、三方向、旧Top10掉榜、完整漂移及费用。GO升级条件预先固定：前瞻臂在三个父各自的新MRR不低于固定Anchor、Replay、匹配随机的最佳值；旧VALID MRR不低于三者最佳值、掉榜不多于三者最小值，每父至少一项严格改善；新三方向各不低于普通的.98，并且额外风险成本合理。任一失败则NO-GO这个触发规则，不修阈值、不扩P3、不换情景/当前组合追小数。

诊断已暴露开发数据，不是未见确认；目标是新旧联合表现，自适应没有收益不能给识别模块功劳。historical144/145预测/teacher/父/tape费用非零，seed43首次stage计时未持久化为unknown，不写成0。无TEST/硬进程超时/独审/SHA/归档/自动push，观察超时沿原句柄接续。
