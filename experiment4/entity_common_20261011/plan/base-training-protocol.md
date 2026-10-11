# 共同旧模型训练注册协议

本阶段仅准备共同旧图底座，不等于已启动或完成所有文献方法的增量比较。复用共同数据预检：FB_CKGE snapshot0 TRAIN111642事实，7505实体、474包含逆关系的关系。训练使用原事实及逆方向，不读旧TEST、新快照TRAIN/VALID/TEST三元组。

三seed17/29/43分别训练TransE200、DistMult200、TransE180（保留SAGE原起始维数，不能共享不兼容维数）。200epoch上限，每job2400秒、batch1024、10负例；学习率TransE1e-4、DistMult.007。TransE复用作者归一化L1距离及先取负距离均值再margin8 hinge；DistMult复用已用softplus正负损失和L2=.0005，称共同训练实现而非文献原训练复现。参数200维底座可直接共享给兼容增量算法；180维供SAGE后续增长。

每5epoch及epoch1评估固定oldVALID1024条方向query（selection seed124）raw strict MRR，选首次最高者，不用旧排名保护或新知识TEST选择。为数据一致，每架构/seed使用完全同一VALID采样。注册固定不进行此次base阶段调参。新初始化及方法预算仍需另外冻结，不能依据新TEST追参数。

每job单3090两CPU线程，现场空闲GPU0..6，9个任务排队；不用无关资源进程。记录训练loss、epoch、选点、用时、峰值显存、参数数与输入SHA。64个oldVALID全候选评分独立NumPy复算：TransE绝对误差≤1e-12，DistMult≤2e-4；排名必须精确相同，失败原样留存，不降低门槛。通过后才允许把基模型供适配调用；未审权重不作正式比较结论。

主要产物在远程artifacts/base。CLI/coordinator与可复用模型训练分别放scripts/src。canonical experiments/KGE-ENTITY-COMMON-BASE-001/experiment.yaml为唯一状态；终端收据是执行证据。原文献原生结果、统一重算和失败记录均不覆盖。共同新增任务主after评分固定11258候选，但本stage只训练7505旧实体，未知实体不能提前参与训练负例。
