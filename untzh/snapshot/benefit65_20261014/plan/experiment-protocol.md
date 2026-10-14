# 065 真实短分支干预收益学习器

DistMult200/FB-CKGE 0→1/seed17/旧E,R冻结；复用058 prefix10/16384旧TRAIN资格cohort与052tape。TRAIN查询组split0的128个action拟合，split1的32个action开发检验：每组一半从当前风险top20%中抽，一半均匀；split2旧查询不用作拟合或标签总体。

每action从同prefix10及Adam状态复制，37步/epoch11学习；absolute保护同一答案、128合法新候选，rho .1/既有lambda，产生真实分支终态。无保护37步为共同基准。各action旧收益为其对应查询组总体平均RR改善+.1*掉榜率改善；新学习成本为三个新TRAIN方向各256固定随机探针的正向MRR损失比均值。只TRAIN评分，不用VALID标签。

用TRAIN action标签std分别归一化收益/成本，净标签=收益/std−成本/std；Ridge(alpha10)/小HGB(maxleaf3/depth2/60iter)仅fit128 actions。输入同058当前风险特征前15维，无未来完整rank、epoch或proposal特征。报告32开发actions Spearman及top8真实收益，不把一个query干预的收益当32query组合的精确效果。

四实际路由配置：ridge/tree × 静态prefix10打分/每epoch当前打分。相同top20%取26+全池6、32×128候选、训练370步及梯度calibration/cap .1，与历史uniform/risk/replay对照。动态使用prefix10学习器是状态外推，时间泛化未独立验证。

所有4组训练结束后读取已暴露VALID，筛选/非支配旧TRAIN评价同063。保存所有VALID结果；未全TRAIN为NA。单seed开发搜索，不称创新/独立确认/净收益标签普适。短分支标签生成成本必须计入总成本；无TEST/硬超时/CPU审计/SHA/归档。新产物在/data/lc/untzh/benefit65_20261014。
