# 111新—新TRAIN分数附加修正，三个父状态固定比较

110第三父状态query_uniform两端新VALID保持.957363，提示旧端固定查询空间不覆盖新—新分数附加变化。固定seed43/17/29各pair4/pair16/query_normmatched_pair4三臂，共九新配置。沿101/108/110同query采样、fullabsolute/rho .1/joint Adam、prefix10、tape、旧E/R冻结、在线普通影子。不生成新负采样、teacher、SVD或新学习器。

用全部924新—新TRAIN方向事实（对应462原事实，两方向关系各保留，不把不同关系去重）。每步先普通累计差旧查询空间投影，再在新行FP64上做4或16次已见新—新分数差修正：目标为同一步普通影子实际分数；各事实误差除以当前两端梯度norm平方之和，聚合两端梯度后按新行参与度平均，再投影至原旧端query正交补。用反映当前状态的梯度、不读VALID；这是有限次非线性近似修正，不能称精确可行投影或保证新—新/VALID排名安全。

query_normmatched_pair4沿原query_uniform方向，只把每步投影累计差norm匹配pair4保存TRAIN轨迹，区分附加修正的方向/大小。若需要放大记录scale；对照依赖候选训练轨迹，非独立部署算法。同norm不等于同Adam/梯度轨迹。无需保存普通330MB快照。

三个父状态全部九模型提交后统一旧TRAIN260498、旧VALID86832、新实体VALID2786、TRAIN-only raw/filtered、三新方向。17/29复用107缓存，43复用110teacher/gold索引，分母各自父状态；新VALID仍原完整评分。记录已见新—新残差均方/最大及变化、旧端score残差、correction费用、TRAIN和VALID取舍。简单的已知约束修正不称原创。无TEST/硬超时/独审/SHA/归档。
