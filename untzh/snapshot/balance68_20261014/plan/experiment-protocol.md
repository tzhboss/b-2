# 068 学习压力与保护平衡

seed17/同DistMult200/FB-CKGE0→1/旧E,R冻结/prefix10/后370同tape。067 contrastive提高newMRR但旧伤大，故15固定配置：original与contrastive混合mu .1/.25/.5，各none/absolute/mass_sample，对应rho .1/.2/.3；纯contrastive+absolute/mass_old各rho .5/1/2。这是host/regularizer共同开发取舍筛选，所有none control必须报告，不把纯学习收益当保护创新。

每loss含同L2，混合两项权重和1；同10negative tape、32保护答案及初态8batchRMS lambda .1，再lambda/cap共同乘rho/.1。概率质量定义沿067，TRAIN-only old logZ/合法candidate，mass_sample128估计有log偏差。全部训练先完成才评价暴露VALID，无TEST。筛选/非支配旧TRAIN沿063，模型全部报告，NA未算。

单seed开发、多次试验非独立确认；profile prepare/train/eval，prefix历史成本归058；不设硬超时/独立审计/SHA/归档，新资产/data/lc/untzh/balance68_20261014。
