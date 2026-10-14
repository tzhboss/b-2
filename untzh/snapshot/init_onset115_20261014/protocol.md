# 115 WN初始化尺度与开启时点

114显示旧VALID原Top10 446条在随机新候选扩展时已掉榜，0→10/10→20没有新增掉榜而分别救回59/28。固定新增三个配置：lowinit_none、lowinit_query_late、standard_query_start0；lowinit仅把112已保存同一随机新行乘.25，std .12→.03，不重抽初值/负例/teacher/cohort/SVD。低初值普通prefix10一次，其后ordinary与query共享；standard start0使用原112初态，保护220步、普通参考220步。采样还是32互异query/RNG1010029/fullabsolute/rho .1/原query空间，旧E/R冻结。

低尺度是针对已暴露WN开发诊断的单一预设试验，不声称未见泛化、原创方法或按旧VALID自适应选σ；不继续扫σ。对照112 standard ordinary/query_late只读复用。两条普通参考各与对应已保存ordinary终态逐位核对；start0更多保护步，不能把同rho当同累计预算。

三个模型全提交后统一完整VALID与TRAIN。使用112旧rank/gold及truth缓存，计算lowinit初始竞争变化并计费，保持同新VALID。旧TRAIN父状态不变因此资格各同66919/6566；新学习MRR和三方向、掉榜/救回、费用都报告。无TEST/硬超时/独审/SHA/归档。
