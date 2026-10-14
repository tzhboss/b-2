# 127 WN两模型zero/half初值上的简单保护与Replay控制

112 DistMult与118真实ComplEx父只读，122 zero/random_half初态/prefix含Adam/ordinary终态全部复用。两模型×两初值×simple absolute/answer anchor/full replay joint共12新后段保护配置，110步；全部模型提交后统一完整旧TRAIN66960/VALID22320、新VALID3720评价，raw/filtered/三方向/旧MRR及Top10drops都有。

旧E/R固定，112 tape/cohort/new truths不变；各模型q/空间/replay正例与mask缓存复用，ComplEx head/tail正确分开。simple absolute32×128、RMS8batch系数+.1逐步cap；answer full4093合法新候选、rho .1与jointAdam累计相对自身ordinary shadow正交补；replay joint用既存旧正例尾/头两方向完整合法候选penalty，同rho .1，不投影，持久jointAdam。旧参数冻结的Replay梯度权限不同于全参数原生Replay，不当作者原生复现。shadow全部匹配自身122普通，额外compute计费，非独立方法。

旧父/teacher/tape/prefix费用非零；不按本轮VALID调配方或第一模型结果修改第二模型，现有VALID均开发暴露。无新父训练/TEST/硬超时/独审/SHA/归档。目标是更强初始化底座下的附加保护及Replay比较，不把它代替风险指导干预目标，也不称基本方法原创。
