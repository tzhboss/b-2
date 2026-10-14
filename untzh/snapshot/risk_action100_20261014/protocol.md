# 100风险×可干预程度，启动前固定三臂

seed29、088snapshot1→2新实体子任务；094同一累计差query-space投影动作/fullabsolute/rho.1/joint Adam/32旧答案×1876合法新候选。前10epoch普通及后220步tape/cohort/teacher/空间/普通影子复用，不重SVD或影子。

099几何缓存16384×128的fraction=||P_j q||²/||q||²。每epoch从各自当前状态找128固定合法候选中分数最高者，查其fraction a。树风险分数p来自058迁移模型，只作排序权重，非校准概率。

- risk_actionable：p*a，乘方固定1，不调阈值。
- actionable_only：a，去除风险信号。
- risk_shuffled_geometry：p*a[固定query置换]，保留权重分布、破坏当前query几何对应；置换固定seed1000029，一次缓存。

每epoch自身top5%池，每步26池+6全池无重复、保护采样种子同097。首prefix features基本核对，不读当前future标签或VALID选规则。

三模型全提交后同088 oldTRAIN全体与old/new VALID1024评价，TRAIN-only raw/filtered并存；资格214095/416。对照097risk-only/pressure/random与094uniform直接复用，非独立重复。若只是强度变化，要补共同累计差norm控制；若不优于简单控制则保留失败，不追乘方/阈值。

候选fraction只是一阶允许方向的几何量，不是实际Adam收益或未来排名安全。099分桶属于TRAIN开发诊断而非独立证据。本轮没有新学习器，无TEST、硬超时、独审、SHA、归档。历史geometry/树拟合/空间/影子费用非零，记录加载、检测、训练、统一评价成本。
