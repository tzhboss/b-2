# 096两个种子的共同累计差norm控制

启动前固定seed29、seed17各一组anchor_common_norm。学习与保护损失、rho.1、joint Adam、prefix/tape/32旧答案×1876合法新候选均与094/095相同。仅将每步相对同一步普通影子的累计差保留自身raw方向，缩放到该seed已存anchor_null每步实际累计差norm。可放大也可缩小，记录scale、放大步数、FP32 norm误差。该控制用于检验方向与累计差强度，不是待部署方法。

复用094/095普通影子缓存、TRAIN-only产生的220个目标norm；不重建SVD/影子/cohort/tape/teacher。历史缓存成本非零、明确另计。每个种子自己的目标norm，不能跨父模型复用。

两种子模型均保存后统一评价既有各自oldTRAIN全体和old/new VALID各1024，TRAIN-only filtering、raw/filtered并列。没有新抽样、TEST、超时、CPU独审、SHA、归档。历史VALID已暴露，不声称未见确认。

主要比较同种子anchor_null/普通/简单absolute/强fullabsolute的newMRR、三个新方向、oldMRR和Top10掉榜；norm匹配不等于Adam每步路径或旧查询暴露强度完全相同，不借控制单独宣布因果或原创。只有方向机制和学新—保旧价值获得支持，才考虑把风险识别加入这一动作。
