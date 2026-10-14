# 071 新实体查询/候选角色分开

模型由共享DistMult改为仅新实体具有不同H/T角色向量（各200维）；旧实体H=T=原E、R仍冻结，旧池分数不变。CP/角色分离是已知模型思想，本轮是更新结构的适配测试，不称原创。可训练参数由750600增至1501200，须单列容量控制。

固定seed17，同FB-CKGE0→1/TR/tape。共享/拆分结构各从原initial新行值开始warmup原学习10epoch，保存各自真正Adam状态，后10epoch同tape：shared none/absolute；split original/mixed .5各none/absolute rho .1/.3/mass_sample rho .1；split contrastive none/absolute rho .3/1/mass_sample rho .3，共14组。

学习ps=H[h]R[r]T[t]、同10负例，original/contrastive/mixed定义同067/068；所有共享/拆分组相同对称L2：.00025*(Hh²mean+Tt²mean)+.0005*R²mean。与历史原L2有表达变化，因此matched shared controls必要，不能只与历史DistMult比称保护收益。newTRAIN含原始/逆方向，整体h/t分布对称，但不冒称每batch两种L2逐位相同。

保护只作用T候选参数，old query缓存合法。32×128合法候选，absolute/mass_sample；初态8batch按候选T的学习梯度norm校准/逐步rho cap，H梯度保持学习host，不把H更新norm当额外保护预算。新headed query仍可能经T候选竞争受到间接影响。共享结构参数单个Adam，拆分两个role各有真实warmup Adam状态。

所有14组训练完才读暴露VALID，三方向/raw/filtered/MRR/旧固定Top10；筛选/ParetoTRAIN沿063，全失败保留。模型保存H/E(尾)/R，评价必须使用H查询向量，不能当普通E查询模型。无TEST/硬超时/CPU独审/SHA/归档，成本prepare/warmup/train/eval都记录；新资产/data/lc/untzh/roles71_20261014。单seed开发，原生作者模型/多seed确认仍缺。
