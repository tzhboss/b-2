# 094 累计参数差相对普通轨迹的查询空间保护

093的当前附加位移保持TRAIN分数但不能阻止joint Adam历史对未来host更新的损失。本轮只检验这个具体缺口，不调原loss/阈值。

088seed29第二增量、initial/prefix10 Adam/tape/cohort/teacher、32旧TRAIN答案×完整1876合法新候选fullabsolute，nonzero保护gradient norm rho.1；新臂joint optimizer，后220步。旧11258E/474R冻结，新1876行更新。复用092 per-new-row跨端新TRAIN查询空间。

额外生成一次普通学习prefix后220步的各步新行状态，CPU缓存330.176MB并保存（实际bytes以输出为准），末态与088none逐位核对，生成费用归属本方法，不称免费或独立重复。每步joint更新后，以同一步普通模型为reference，累计newE差投影到固定TRAIN查询空间正交补(anchor_null)；anchor_normmatched仅将累计差缩至本状态投影后的norm。两组真实optimizer都保留joint梯度moments。

null组对已见跨新旧TRAIN正例分数的约束可跨步保持为普通参考；仍不保证负例/VALID/双新端/排名，也不是保持保护臂本来host提案。普通影子轨迹增加训练/缓存成本，须全部报告。控制是按自身状态norm规则匹配，不代表共轨迹同强度。094为开发机制实验，不称原创/全局安全。

两模型全部完成后同088固定1024+1024VALID/fulloldTRAIN，214095/416资格，raw/filtered、旧MRR/drops与三新方向一起评价；复用090/093控制。所有开发VALID已暴露，无TEST/future风险标签/硬超时/CPU独审/SHA/归档。
