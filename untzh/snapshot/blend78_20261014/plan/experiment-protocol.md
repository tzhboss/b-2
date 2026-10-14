# 078 固定50/50梯度组合的机制控制

固定077cohort_blend0.5：absolute梯度与mass_sample梯度分别同初态8batchRMS/rho.1校准和逐步cap，再各半相加。保护norm上界.1，host/Adam/32答案×128候选等同058，旧E,R冻结。mass_sample old-logZ/cohort候选/合法newcount全部只读缓存077/058，不重算大准备。

9新配置：保护起点0/5epoch各absolute/mass/blend，20epoch统一终态；onset10 absolute_normmatched/mass_normmatched两控制，以及absolute rho.075控制。normmatched每步计算该当前状态下blend保护的norm，但只保留absolute或mass方向缩放到相同norm，用于排除组合单纯降强度解释。对应两component统一rho.1与初态calibration，不以candidate未来轨迹的norm指导控制。prefix5真实Adam缓存076、prefix10缓存058、初态Adam0。

onset10 blend直接引用077已有模型/评价，不重复重训或评分。所有9新模型完成后一次读暴露VALID，筛选/非支配TRAIN沿063；所有失败保留，必要方向比较另补TRAIN。按起点改变仅是同seed时点稳健性，非独立数据确认。normmatched有两梯度费用，无免费部署结论。

新资产/data/lc/untzh/blend78_20261014；无TEST/新seed/硬超时/CPU独审/SHA/归档。固定单seed多次开发，统计描述不能抵消选择偏倚或主张新颖性。
