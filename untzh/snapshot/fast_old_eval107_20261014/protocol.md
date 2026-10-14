# 107冻结旧行下的评价优化

只优化旧TRAIN260498与旧VALID86832评价。固定seed29/17的旧实体11258与关系冻结，旧候选排名直接复用各自teacher；缓存旧查询gold分数，仍按原64行FP64 GEMM提取，避免elementwise gold最后bit差异。每个终态只计算1876新增候选超过gold的数量，TRAIN-only过滤口径、strict greater/tie不变。新实体VALID2786继续原完整评分。

首次缓存gold费用计入准备，历史teacher费用非零。new-gold小于1e-10*(1+abs(gold))的块回退原完整评分以处理临界数值，不放宽rank匹配。固定105两方法×两seed的旧TRAIN/完整旧VALID共八数组，与保存rank逐条核对；全相同才允许后续新模型复用。失败保留记录并恢复原评分，不补阈值找一致性。

这是实现所需验证，不是独立科学确认或CPU审计。仅重算新候选和首次缓存gold，既存完整神经rank不重算（临界fallback除外）；报告缓存准备/bytes/每次评价耗时，无异时可靠加速比。无新训练、TEST、SHA、归档、硬超时。
