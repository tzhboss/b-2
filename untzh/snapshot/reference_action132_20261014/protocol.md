# 132 相同历史参考监督下的直接保护对照

两个固定父FB17/WN ComplEx；zero与130 history-reference constrained两初值，prefix/ordinary复用121/122/131，不重训前缀。两保护动作reference_softplus/reference_budget，共8新后段模型。历史校准半群组与131完全相同，只用原旧Top10已知事实，不访问另一半或新VALID标签。

额外历史事实监督必须给直接保护对照：固定calibration eligible池，每步32不重复旧查询，rng1320017一次保存，同父两动作和初值共享选择序列。全1876/4093合法新候选，仅TRAIN-truth过滤。reference_softplus用mean softplus(new score)的完整池penalty，是旧参数冻结后的Replay适配对照；不当全参数原生Replay。reference_budget按(old gold-new score)/qnorm定义正超越gap，允许10-oldrank个最大正gap，惩罚剩余正gap平方。均jointAdam当前学习梯度+保护梯度，非零保护梯度norm≤rho .1学习norm；不做答案空间投影，不含普通shadow。预算截断为已知方法，不宣称原创。

先提交全部8终态，再统一完整TRAIN/VALID及128旧校准/留出评价；zero/constraint自身ordinary和答案anchor缓存控制单列。报告新三方向raw/filtered、旧MRR/掉榜、非零保护步数与累计强度，不能用相同rho冒充同累计强度。参考cache/teacher/tape/prefix复用，历史费另计。新VALID不选动作/规则；已开发暴露，非未见确认。无TEST/硬超时/独审/SHA/归档。
