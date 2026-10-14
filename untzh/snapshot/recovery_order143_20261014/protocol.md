# 143：风险感知学习顺序，单次固定规则证伪

新假设：风险可能更适合决定更新的先后，先学习能恢复旧参考排名的批次，而不是把所有冲突实体统一保护。保持每个epoch的全部原TRAIN事实draw、与事实绑定的负例、batch边界和110后段学习步不变，只改变epoch内11个batch的提交顺序，不改初始化/query池/penalty/rho。

复用140 WN ComplEx父/zero/prefix/Adam、112学习tape、141固定保护/Replay/space、142固定256旧TRAIN内部参考probe；参考q/teacher gold/rank和合法新候选过滤一次读取。无需新父、前缀、随机负例或新学习器。官方VALID仅五/六终态全部提交后评价，TEST不用，历史开发暴露不改称独立确认。

每个后段epoch开始，全部臂在各自host状态/相同Adam矩上，为该epoch的11个原batch各做一次普通学习Adam预览，读取probe真实filtered rank/MRR。预览不提交到host。固定规则：recovery_first按预览MRR降序，recovery_last按升序，并列均原batch编号；random_order固定1430017打乱epoch内batch；none/fixed_anchor/fixed_replay仍原顺序。

实际提交时重新计算梯度，避免使用过期预览梯度。六臂都有相同11×10预览、学习/Anchor/Replay两保护梯度、ordinary shadow和完整投影操作包络；只固定Anchor实施累计位移投影，Replay不投影，其余无保护。shadow沿各臂实际选定的batch顺序执行，不能套原顺序ordinary。预测scores仅TRAIN输入、未来终态标签不用。

全部原batch每epoch恰一次、每个学习draw和负例pair不丢不重；记录原batchid序列、预览MRR和成本。原顺序三个固定模型应与141参数相同，rank可复用；随机不是同参数控制、反向排序检验预测顺序是否有独立意义。六训练轨迹不等于六独立方法确认。

GO候选需在相同数据/训练步/操作包络下，相对固定强保护与随机顺序有新旧联合收益及合理方向/成本；否则NO-GO本规则，不换排序权重/probe数量/阈值继续。先仅一父条件，不能直接P3或称已有课程排序原创。此实验检验风险感知学习调度及对保护依赖，不替代最终风险→动作→联合评价目标。
