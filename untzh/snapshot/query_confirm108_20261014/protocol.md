# 108查询级分配seed17固定确认

固定101 query_uniform/query_random_pool两臂，绝对完整1876新候选保护信号、rho .1、joint Adam、普通参考累计差投影、每步32互异(h,r)不变。随机集中池各epoch固定top5%、26池内+6其余，不重新选择阈值/强度；101 RNG 1010029和1010000+epoch不变。seed17来自087初态/prefix10/tape/cohort/TRAIN truths，095空间，不用seed29模型空间。cohort同查询代表任一答案的q相同；absolute_full不依赖gold答案，仍按101每query一个代表。

使用106已验证的GPU在线普通影子，无逐步快照。两新模型保存完后才统一评价完整旧TRAIN260498、旧VALID86832、新实体VALID2786。107旧排名缓存只在summary all_equal条件下使用；新VALID保持原完整评分。每模型检查旧E/R逐位冻结。普通终态核对影子，费用包含普通路径。对照只读102的seed17 none/absolute/anchor等，不作独立重复。全部VALID已暴露开发数据，不读TEST。

主比较：能否较简单absolute或answer-uniform同时减少旧VALID掉榜、提高新MRR并改善各新方向；同时记录旧MRR、TRAINdrops、曝光集中度、实际累计差norm和费用。结果有取舍则如实报告，不补采样种子/阈值追正。风险路由本轮不加新网络；尚未验证其端到端价值。无硬超时、独审、SHA、归档。
