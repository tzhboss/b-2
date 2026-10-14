# 073 边界保护、局部冲突和学习负例对照

seed17/DistMult200/FB-CKGE0→1/旧E,R冻结、058prefix10/后370同tape。16固定配置：boundary oldranktop20%路由 × 实体/逐坐标非冲突 × 旧头新gold学习权重1/1.1/1.2；uniform下权重1.1/1.2各none/absolute；TRAIN过滤已知正例negative的none/absolute/mass_sample；mass_sample实体/坐标投影；boundary mass实体投影学习权重1.1。权重不改变L2，负例过滤后按每row实际合法数量平均，无合法时负项0，不重抽。

保护32×128候选/初态8batchRMS按host校准rho .1/逐步cap .1；geometry在同base校准后执行。boundary26+uniform6，稳定oldrank路由不需要未来风险标签。mass_sample沿067旧logZ/新exp池估计，不叫无偏。所有控制/失败报告。

全部16训练结束后才VALID；筛选/非支配TRAIN沿063，固定原资格。prepare/train/eval成本记录，新增合法negative缓存属于TRAIN准备。固定单seed暴露开发，多轮探索不是独立验证，无TEST/硬超时/CPU独审/SHA/归档。新产物/data/lc/untzh/refine73_20261014。
