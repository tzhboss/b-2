# 137 P2 单次固定实体动作选择证伪

核心假设：固定136 prefix纯TRAIN风险与连接度行gate选择无保护/Anchor/Replay，能否在相同计算预算下超过固定保护？136已显示该gate没有Anchor触发，须明确为当前退化二动作机制，不能据结果改阈值凑三动作。

两父FB17/WN ComplEx、zero/prefix/Adam/newTRAIN/tape/space不变。六臂none/fixed_anchor/fixed_replay/random_matched/risk_adaptive/degree_only。risk读取136冻结gate；random_matched只将同动作计数固定rng1370017打乱，数量/预算相同；degree_only对有跨旧query支持degree≤3行Anchor、其余Replay，无风险输入，是识别必要性消融。

固定Anchor为126/127同32旧TRAIN答案×全合法新候选softplus、累计相对ordinary影子位移投影；Replay为105/112/118已有全历史TRAIN32正例头/尾完整候选softplus、旧E/R冻结适配，不称原生全参数Replay。行action1用Anchor梯度，action2用Replay梯度，action0无保护梯度。拼接后整个新行保护梯度norm≤rho.1学习norm；jointAdam后仅Anchor行采用空间投影，其他行保持host提案，学习间接耦合允许存在。固定gate后段不变，不读未来标签。

严格计算包络：所有臂每步计算同一学习梯度、Anchor梯度、Replay梯度、完整空间投影、同普通shadow，普通臂也计算但丢弃保护梯度/投影。等后段220/110学习步、相同额外操作上限/可训练权限；真实部署可能省略未选动作，部署费不能用本轮计费代替。分别记录学习/两保护/影子/投影/选择/总费用，CUDA同步计时引入开销，跨臂时间非精确相同。完整诊断总费（136）单列，已有父/cache不免费。

所有12终态提交后统一full旧TRAIN/VALID、新VALID、三方向raw/filtered、128旧半群组和旧原Top10排名漂移。无TEST/硬超时/独审/SHA/归档，extra VALID参考133不作公平竞争臂。none/anchor与既存逐位比较用于缓存复用，若匹配则不重复评分，不当独立科学重复；Replay FB zero首次训练。

GO门槛：risk需要相对random_matched具有独立新旧联合收益，并相对强fixed Anchor/Replay无以方向显著损失换掉榜改善；两个对象可评估范围内一致，费用合理。P3稳定性未做即不得升论文方法。若未达，NO-GO停止该gate及反复阈值修补；按失败原因提出不同可检验假设。
