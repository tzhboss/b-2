# 133 参考保护方向与强度控制

两个固定父FB17/WN ComplEx，zero/130历史约束两初值，所有prefix/ordinary复用121/122/131；132保存历史calibration参考q/合法mask/选择tape只读复用，不重新采样或teacher评分。每初值reference_softplus_anchor/reference_softplus_normmatched两后段方法，共8新训练配置，完整评价同132。

持久jointAdam学习梯度+参考softplus保护梯度（32校准旧Top10 query×完整合法新候选，rho .1）。每步在线同初值普通shadow推进，累计delta=当前newE-shadownewE；anchor将delta投到新TRAIN跨新旧query空间正交补，normmatched按投影norm缩放未投影delta。控制范数在各自状态计算，非共同轨迹因果匹配；影子额外计算计费且终态需匹配自身既存ordinary。

同132 extra历史参考监督，不当TRAIN-only/原生全参数Replay；所有8终态提交后才评价完整及128旧留出/校准、新三方向raw/filtered/费用。和132未投影参考softplus、131仅TRAIN保护及zero控制比较，不以TRAIN分数约束宣称VALID/双新排名安全。正交方法已知，不宣原创。无TEST/硬超时/独审/SHA/归档。
