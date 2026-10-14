# 087：第二次实体增量snapshot1→2

固定seed17/DistMult200，父模型为052普通新实体学习epoch20（snapshot0→1，全部3753新行已训练），不使用077最佳候选作父模型。历史TRAIN为snapshot0+1，新TRAIN为snapshot2涉及新增实体的事实；无新增实体的事实若存在，则只加入TRAIN过滤truth并报告数量，不对冻结参数声称学习。初始已知11258实体扩至13134，关系映射仍474；旧11258实体和关系逐位冻结，仅训练新增1876行。

新增初始化沿用normal std .12，但本轮新GPU生成器固定seed1087，并明确不同于0→1历史初态。每增量Adam fresh .007、20epoch、batch1024、10合法负例、L2 .0005。共享前10epoch普通prefix、学习tape及16384旧合格TRAIN答案cohort、32答案×128合法新候选抽样。旧资格用父模型在旧候选池TRAIN过滤Top10，绝不跨增量复用187470分母。

固定六臂：none/absolute/mass/blend/absolute_normmatched/absolute075，配方完全沿用080。同一新初态8batchRMS校准rho .1，逐步cap，blend各半。检验保护信号的跨增量行为，而不是继续筛时点阈值。准备缓存只生成一次，所有训练结束才读取snapshot0/1/2 VALID。

旧VALID取snapshot0+1固定随机1024方向查询；新VALID取snapshot2涉及新实体的固定1024方向查询。TRAIN-only过滤来自全部0/1/2 TRAIN，不读TEST。评价所有六臂完整旧TRAIN、旧VALID、新VALID raw/filtered MRR/Hits/Top10固定资格掉榜及三个新方向。此阶段VALID未参与前述配方选优，但旧VALID和部分父模型历史资产已暴露，不能叫独立数据集或多seed确认。

费用记录本次TRAIN读取/teacher资格/cohort/tape/prefix/校准/各臂训练/评价，历史父模型生成费用非零。不是最近邻原生复现或安全证明。资产/data/lc/untzh/increment87_20261014，无硬超时、CPU审计、SHA/归档。

实际准备：原37214方向事实中21566涉及新实体，15648两端都是旧实体；本轮只研究新实体学习子任务，不能代表snapshot2完整新增事实学习。全部1876新实体均有TRAIN支撑。每epoch22批、20epoch440步、共享prefix220步、各臂后220步。首版错误套用旧740步假定，在准备完成后断言失败、尚未训练；旧log/代码保留，改按实际batch数循环并只读复用初态/train/tape。协议此前740措辞已据真实数据纠正，不是延预算或增加epoch。
