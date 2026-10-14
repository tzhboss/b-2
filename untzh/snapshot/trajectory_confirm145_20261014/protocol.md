# 145：144同公式在既存FB43/29父上的P1重复诊断

固定FB DistMult200、相同1→2新实体子任务、121 zero初态与prefix10含Adam、父各自原440步学习tape/候选truth、126固定Anchor终态。这两个父及VALID已在历史开发中暴露，不是未见确认或P3方法扩展。本轮零新父/零训练配置，不调初始化、query池、教师边界、penalty、预测H或情景方差。

完整复用144实现：第11epoch22个TRAIN batch在原prefix求梯度，均值/二阶矩外推剩余220步，共同经验情景按1/sqrt(H)。首batch一步Adam×H作单提案基线；current完整rank/gap与136固定128候选压力、旧rank边界作对照。136纯TRAIN父各自cohort以及完整旧VALID资格，特征构造与预测在两个父均保存后才读任何终态。VALID gold只作离线诊断，不进入部署动作。

评价前冻结：晚期新增掉榜为主目标，普通自恢复/Anchor额外救回为独立目标。AP/1%5%10%预算命中、Brier与恒零风险/评价期先验、排名误差/情景宽度/实际成本。均值预测GO仅限风险排序候选：在两个追加父各自VALID晚期事件上，AP均超过current完整rank、current完整gap及首batch外推，并且前5%召回不低于current rank；否则停止此平稳矩排序公式，不调参。经验情景仅当两个父AP均超过均值且前5%命中不差才保留为增量，否则NO-GO情景模块。即使排序GO，也不称风险概率/校准/保护必要性/自适应收益。

无TEST、神经训练rollout、硬进程超时、独审、SHA、归档或自动GitHub同步；所有历史准备与父训练成本非零。本轮新梯度44batch、hypothetical moment440轮，不计新训练步。
