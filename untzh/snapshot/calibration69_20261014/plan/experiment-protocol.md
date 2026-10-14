# 069 新候选尺度校准

固定seed17开发探索，无新训练。来源067 contrastive_none、068 mix50_none/mix50_absolute及052ordinary/058uniform。前3种均来自prefix10切换后10epoch，gamma固定.7/.8/.85/.9/.95/1；ordinary/uniform matched简单控制gamma .9/.95，共22配置。每个新实体向量乘正gamma，旧E及R不变。DistMult头侧缩放为正公共query因子，排名上等价于只把新候选分数乘gamma（浮点边界另列）。不是新的embedding算法或原创校准思想。

解释假设：contrastive/mixed改善新候选学习，同时产生竞争压力；校准削减新候选与旧池的竞争，希望保留更好的相对新候选排序。必检查两个以新答案为gold方向、以及新头旧gold方向，不能凭总体新MRR/旧掉榜取胜。

所有22模型先物化后一次读已暴露VALID；筛选及非支配旧TRAIN评价沿063，所有失败保留。参数网格是开发搜索，任何找到的gamma须后续TRAIN-only选参/固定配置验证，不能称未见确认。无TEST/硬超时/CPU审计/SHA/归档；成本包括评分/物化及历史学习成本来源说明。
