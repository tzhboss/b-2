# 070 按TRAIN证据识别校准范围

无新训练，四来源ordinary/uniform/068mixed/mixed_protected；44固定配置：query_scale/relation_scale gamma .7/.8/.9/.95，query_shift delta .1/.25/.5。输入信息只有TRAIN、实体新旧身份、当前模型分数；VALID gold只用于终点评分。

query模式仅对old-head且该(h,r)在newTRAIN没有新增gold时校准新候选；new-headed query全部不校准，保留两个新头方向。scale乘gamma；shift从new候选分数减delta×当前old候选分数std。relation模式仅对old-head且(r,newcandidate)未在newTRAIN出现时乘gamma。其他新候选分数及所有旧候选分数不变。不能把TRAIN未出现理解为必然假答案；未知真新答案可能被惩罚，必须报告旧头新gold性能。

这是TRAIN-conditioned inference校准，模型E/R未修改；评价wrapper才执行规则，每配置source/规则保存calibration-configs。所有44配置物化后一次读暴露VALID，筛选/Pareto旧TRAIN沿063；失败全报，NA未算。新头方向应和原source整数rank完全一致，后续基本检查该不变量。不冒称原创或独立确认，无TEST/硬超时/CPU审计/SHA/归档。新资产/data/lc/untzh/conditional70_20261014。
