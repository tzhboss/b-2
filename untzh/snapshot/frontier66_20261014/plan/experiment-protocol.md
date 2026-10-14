# 066 语义保护与强度匹配

相同DistMult200/FB-CKGE0→1/seed17/旧E,R冻结、prefix10/同tape后370步。13配置：均匀、关系候选排除、暴露query权重.1、仅old-only query，各rho .2/.3；候选关系排除×query软权重组合rho .1/.2/.3；old-only逐坐标非冲突rho .2/.3。原rho .1参照来自058/064，不重训。

每损失校准lambda .1源规则不变；新rho把lambda及逐步cap同时乘rho/.1。控制组同样增强保护，避免只调新方法。每步32×128合法新候选，所有语义只来自TRAIN。13臂全部训练后一次VALID，筛选/非支配旧TRAIN评价同063；进一步比较须考虑各方向与旧MRR，不把更弱保护当方法胜出。

本轮开发搜索，固定两个新增强度点，不据VALID改变门槛或隐藏失败。单seed、多重开发比较，不是独立确认。记录prepare/train/eval耗时，prefix历史费用单列；无TEST/硬超时/CPU审计/SHA/归档；新产物/data/lc/untzh/frontier66_20261014。
