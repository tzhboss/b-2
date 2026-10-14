# 090 同梯度范数规则下的容量截断动作比较

088 seed29第二增量新实体子任务，全部只读复用initial/prefix10 Adam/tape/cohort/teacher/risk模型/TRAIN truths。后220步；旧11258行/474R冻结，新增1876行更新。无TEST。

八配置：absolute128、absolute_full、violation_full、budget_full × uniform/tree。所有非零保护梯度统一归一化到学习新行梯度norm的.1，零梯度则0，不重新RMS校准。故匹配的是同一当前状态规则，不是同Adam位移；budget稀疏时平均实际强度更低，明确记录。absolute128统一归一化是必须的新控制，不能仅拿088旧固定系数控制解释信号效用。

uniform每步32答案使用088既有选择tape；tree与089一致，每epoch当前模型迁移树top5%池、26池内+6全池且无重复。absolute128只算固定128合法候选；其余三种算完整1876新池并过滤TRAIN已知答案。absolute_full作为搜索预算控制。violation_full对score>oldgold的候选用正gap平方；gap除旧query范数，表示单候选跨边界位移的局部尺度。budget_full允许10−oldrank个新越位候选，保留最大gap，惩罚剩余最小正gap；按自由分数坐标实现截断平方距离，是已知容量/截断思想的实验变体，不称原创算法。

全池目标每答案除合法候选数后平均。容量保证仅是固定分数表达中的零损失条件，不能推共享embedding/优化轨迹全局安全，rank1→10的退化仍需旧MRR检查。非零归一化会放大局部稀疏梯度，不是硬保护。记录active queries/pairs/steps、实际gradient norm、训练检测评价成本。

所有八臂提交后同088固定1024+1024VALID及完整oldTRAIN评价；raw/filtered、旧MRR/掉榜、三新方向都报告。已有088与089结果已暴露，属于开发机制比较；不调容量/阈值追正，不用未来标签路由。新费用与历史资产生成费用分列。无硬超时、独立CPU审计、SHA或归档。
