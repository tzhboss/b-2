# 124 候选生成前高斯容量风险估计及零初值机制核对

给定固定旧参考查询q、gold分数g、原filtered rank r≤10，新增M个合法候选行独立各坐标N(0,sigma²)时，单新行超gold概率p=Phi(-g/(sigma||q||))，超越数理想条件分布Binomial(M,p)，掉出Top10先验概率P(X>10-r)。ComplEx用真实tail_query，DistMult q=oldh×R。M扣除已见TRAIN已知新正例。零初值时所有新得分0，排名应精确等于r+M*1[g<0]；qnorm0同样按常数0处理。该解析表达是基本概率/排序统计思路，不宣称原创或训练后安全保证。

固定五父：FB43/17/29与WN DistMult17/ComplEx17；std .12/.03/.015/zero，对应117/121/114/115/118/122既存扩展初态rank，不训练、不重评分神经候选。只从旧参数提取querynorm和既存gold/rank、newTRAIN truth数，预测存盘后读扩展标签。包含TRAIN/VALID原Top10，reference gold已知，不是发现未知答案。比较容量概率、单候选正态tail概率、期望超越数减容量、负gold、旧rank边界、固定随机。报告AP、1/5/10/20%预算、Brier/期望总数及预定概率分箱，单固定seed和共享实体相关事件不足以证明概率校准。

高斯独立初值为解析模型假设，mean/shuffle/flip邻居方向初值不服从该先验，不套公式。未来TRAIN优化会改变分布，本轮只研究候选加入前的风险，不把current检测/后十epoch预警混用。不用future VALID标签拟合/调阈值。现有VALID为暴露开发，未见确认仍缺。

WN DistMult低尺度TRAIN扩展完整rank没有历史缓存则跳过该项，不为凑齐重复神经评分；其他可得全部项保留。无TEST/硬超时/独审/SHA/归档，成本包含querynorm提取、概率计算/缓存/标签评价，既有旧teacher/gold获取成本非零。
