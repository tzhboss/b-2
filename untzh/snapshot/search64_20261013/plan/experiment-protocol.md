# 064 TRAIN语义干扰与更新几何

沿用063 DistMult200/FB-CKGE 0→1/seed17、前10epoch缓存/后10epoch370步同tape、旧E/R冻结。15组：按关系在newTRAIN出现过的新增答案排除或软权重(.1)，128/full候选；关系频数/新实体端点频数逆平方根权重；只从没有新TRAIN答案的旧查询中抽保护样本或对暴露查询软降权；梯度切向/径向/逐坐标非冲突；语义与几何组合；postAdam .2/.3及语义postAdam .1。原始/逆关系IDs分别统计。

缓存候选身份及oldq/gold复用；所有语义来源仅TRAIN、不读VALID标签。各损失初态8batchRMS匹配rho .1、逐步限幅 .1（absolute原lambda），32×128合法候选；postAdam位移norm比例单列为不同优化规则。相同学习host/样本tape/Adam状态。

一次完成全部训练才读已暴露VALID；raw/filtered MRR/Hits与三个新方向、固定旧资格掉榜。筛选及Pareto旧TRAIN评价沿用063，完整输出所有新臂VALID。历史普通/uniform/replay复用非独立重复。准备/训练/评价成本分别记录，共享prefix历史费用不算免费部署。单seed开发探索，多重尝试不代表独立验证/创新；无TEST/硬超时/CPU审计/SHA/归档，新产物只在/data/lc/untzh/search64_20261013。
