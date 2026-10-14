# 088：第二增量seed29固定保护与跨阶段风险预测

FB-CKGE snapshot1→2新实体子任务，DistMult200/seed29。父模型固定080 seed29普通0→1终态；与087父052seed17的训练底座不同，不能称纯新增初始化重复。复用087 cache/train.npz与映射，oldTRAIN260498/newTRAIN21566/extra旧—旧15648，旧11258行与474R冻结，新增1876行normal std .12/生成器1099。新任务freshAdam .007、20epoch440步，共享前10epoch220步。同六臂none/absolute/mass/blend/absolute_normmatched/absolute075，无新强度/时点选优。

重新计算模型相关teacher TRAIN资格、16384cohort/q/gold/tau/logZ；不能复用seed17模型摘要。tape按seed29+71一次生成，所有六臂共享。准备、普通轨迹特征/标签/epoch状态、保护训练、评价费用分别报告。普通轨迹训练一遍保存21个epoch模型/特征/rank，prefix10含Adam，终态none直接保留，不再次训练该臂。

风险标签仅TRAIN：当前安全rank<=10到下一epoch rank>10的转换，同时报告全体下一状态掉榜。全新池1876候选评分仅为生成标签，廉价检测只128抽样候选特征。按(h,r)整组60/20/20划分；前8epoch训练查询标签拟合，epoch8–9开发、epoch10–19留出查询/时间验证。旧rank、capacity、旧score尺度/counts/current压力及epoch提供当前小树；lookahead为真实Adam一步拟议变化×22步外推，不称真实未来轨迹。

固定检测器random/boundary/pressure32/pressure128/lookahead、058已有current-tree直接迁移、本阶段前8段TRAIN拟合current-tree（相同固定80轮/7叶/深3/.08/L2=1）。迁移树只用current列0..14及20，与原列契约一致，不用原37步预测特征。预测比较包含macro安全AP、top5/10/20%预算检出/precision及事件分母；状态-答案事件相关，不称独立样本。高召回不等于准确率/保护收益。

所有保护组及风险模型均由TRAIN确定后才读取统一旧/newVALID，查询选择沿用087。六臂完整旧TRAIN与raw/filtered/Hits/三方向比较。即便某检测器预测好，也不将它加成已获保护收益的模块。本轮不风险路由；下一联动需另行公平训练试验。

无TEST/硬超时/独立审计/SHA/归档。资产/data/lc/untzh/risk_increment88_20261014。本轮配方在新VALID前固定，不按087/本轮VALID补阈值。
