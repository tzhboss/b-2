# 112 WN_CKGE固定配方跨数据集开发比较

数据直接复用FastKGE作者仓库WN_CKGE：snapshot0 24567实体/22方向关系，33480 TRAIN事实、11160 VALID；snapshot1共28660实体、新增4093，5580 TRAIN事实全部涉及新实体，711原事实两端新。双向训练，snapshot0旧TRAIN66960方向、snapshot1新TRAIN11160方向，snapshot1本轮覆盖完整新增TRAIN（和FB1→2的子任务区别明报）。TRAIN-only过滤，不读TEST。

没有可直接复用的同口径WN DistMult父模型：固定seed17初始E/R std .12、Adam .007、同softplus正负损失及.0005正则、TRAIN50epoch学旧E/R。无VALID选模/早停；一次负采样缓存，父训练成本另计，不假称作者原生模型或性能复现。父训练50epoch配方在任何评价前固定，不根据旧VALID质量调参。

增量冻结旧E/R，新增行初始化1087，fresh Adam .007，20epoch220步，共享prefix10 110步，后110固定七臂ordinary/simpleabsolute/answer_anchor/query_anchor/query_pair4/fullReplay_joint/fullReplay_anchor。公式沿110/111，rho .1；simpleabsolute8batch RMS+.1 cap、32旧Top10 TRAIN答案×128合法新候选；anchor/query/fullReplay完整4093新候选；query每步32互异(h,r)、RNG1010029。query_pair4用全部1422双新TRAIN方向事实4次有限分数修正，沿111，非安全保证。fullReplay每步32全历史TRAIN正例、完整合法头尾候选各半，旧常数梯度省略。cohort最多16384，若资格不足按所有资格、仍记录分母，不以人工合成补资格。

teacher旧TRAIN/gold、cohort/query空间/TRAIN tapes一次缓存。在线普通影子计算真实计费，不保存逐步快照。七模型全部提交后读取完整旧VALID22320/新源VALID3720，仅涉及新实体部分用于new指标（若存在旧—旧则报告排除数），teacher旧VALID一次；旧候选固定rank+新增竞争的缓存算法复用，新VALID完整评分。ordinary/query首中尾64行必要实现核对，失败则旧评价全部恢复原评分。

新MRR、三方向、旧TRAIN/VALID MRR/Top10drops、原父资格/覆盖率、实际费用。单seed新数据集的开发迁移，不是未见测试确认/强模型性能保证/原创证明。保留失败，不追阈值。无TEST、硬超时、独审、SHA、归档。
