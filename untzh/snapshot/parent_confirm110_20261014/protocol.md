# 110第三父状态固定配方

DistMult200/FB_CKGE snapshot1→2新实体子任务不变，父为080 seed43 ordinary终态11258实体/474方向关系。新增1876行normal std .12、扩展生成器1070+43=1113；fresh Adam .007。只学087的21566涉及新实体方向事实，15648旧—旧仅truth过滤。20epoch440步，共享prefix10 220步，后220比较none/simple_absolute/answer_anchor/query_uniform/fullReplay_joint/fullReplay_anchor。无调参/新阈值，fullReplay和fullabsolute保护rho .1。

学习tape采用既有core.tape seed43只生成一次，共享负例；cohort16384旧TRAIN原Top10答案、候选128、保护采样均按既有core.cohort seed43生成一次。simple_absolute用087八batch初态RMS校准+.1逐步cap；answer_anchor用094 fullabsolute及已见新TRAIN空间正交补；query_uniform按101每步32互异query，RNG1010029；fullReplay使用105数学完整合法头/尾候选各半梯度、旧常数省略，每步32历史正例seed1040043。普通参考为106在线影子，不保留330MB轨迹。

父模型相关teacher TRAIN排名/gold在同一64行FP64 GEMM循环生成一次，并缓存TRUTH新候选过滤索引；TRAIN query空间double SVD cutoff1e-10只生成一次。六模型全部保存后读取102完整旧VALID86832、新实体VALID2786（开发数据已暴露），新父teacher旧VALID资格单算一次，不能沿用seed17/29分母。旧行/关系每模型逐位冻结；107只新增竞争rank算法复用。固定ordinary/query模型各split首中尾64行与原评分核对，任一不等则本父所有模型恢复原评分并记录，不以小样本核对当全量独立审计。新VALID完整原评分。

指标旧TRAIN/旧VALID raw及TRAIN-only filtered MRR/Top10drops，完整新VALID三方向保持；报告真实梯度比例、位移norm、score残差、准备/训练/评价/缓存费。原普通终态在线影子逐位相同才认可参考实现，无TEST/硬超时/独审/SHA/归档。第三父状态属于开发确认，不是新数据泛化；失败保留，不微调追正。
