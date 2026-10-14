# 136 P0/P1：冻结基线，风险与实际损伤、少连接实体诊断

停止初始化/query-pool/教师边界/微penalty搜索，135判NO-GO（WN all/qualified同池等效，改善不跨对象且仍方向取舍）。纯TRAIN主固定基线采用126/127的zero普通+固定答案Anchor，Replay采用冻结旧E/R的完整历史TRAIN适配，不能当原生全参数Replay。135不作新调参入口。

两对象FB17 DistMult/WN ComplEx，父/zero初态/prefix/ordinary以及teacher/rank/tape/space既存。本轮无训练。P1对原Top10旧TRAINcohort及完整旧VALID资格：prefix早期样本32/128候选gap、mean softplus gap、最大gap、excess估计与原rank边界，提前预测ordinary后10epoch新掉榜。首prefix真实旧rank由既存候选增量评分计算，作为标签资格/机制诊断，不混入廉价特征。完整新候选压力是费用另计的诊断上限；gold旧已知TRAIN/reference用于评分，VALID特征仅离线识别评价，不用于策略。

抽样候选每对象rng1360017固定128行，共享前32；合法mask只TRAIN过滤。所有prefix特征/row risk/action gate先存，随后加载终态rank和模型。标签严格区分prefix已经掉榜、后段新掉榜、救回；未来标签不输入动作。AP/recall@1/5/10%按prefix安全旧事实群体、随机对照；无正例AP=null。固定Anchor个体救回/新增损傷分别看，并按预测风险分层，不能据表观相关声称保护必要或因果效应。

新实体连接度从新增TRAIN偶数关系原事实计数，避免逆向重复；0/1/2–3/4+分层，跨旧连接支持另列。纯TRAINcohort prefix对合法新行的正gap和超过容量的真实竞争用于row风险诊断，固定可检验gate：已见跨旧query支持且有gold超越的degree≤3行→Anchor、degree>3行→Replay；无超越→不保护，双新only/缺跨旧query支持→unknown且Replay。此为当前风险代理，不是终态安全保证。row预测再对后段新越界对比较，pair-crossing不是query掉榜。

记录全部费用、规模、风险/动作比例、连接度分层、完整旧漂移。纯TRAIN与extra VALID参考候选分开，不新增TEST/硬超时/独审/SHA/归档，数据已暴露开发。136 GO只允许137一次固定gate动作对比，不允许阈值扫点；若随机/固定控制不被超过则NO-GO停止此route，不靠P3扩规模寻找正值。
