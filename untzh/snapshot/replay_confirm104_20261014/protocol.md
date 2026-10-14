# 104同第二增量Replay确认：启动前固定两个seed各四组

普通底座与此前完全相同：seed29/17各自父模型、新增1876行初态/20epoch440步学习tape，11258旧E/474R冻结，仅21566新实体方向事实学习。每步额外32历史TRAIN方向事实、每事实同host10负例，标准随机单端腐化，候选含全部13134实体，过滤old+全部snapshot2 TRAIN真例（含未学旧—旧事实），不使用VALID/TEST truths。

每seed固定rng1040000+seed，一次向量化生成全部440步Replay tape并缓存，各臂共用；历史池是260498方向TRAIN全体，非Top10限定，非严格在线固定buffer。真实旧正例与负采样loss用现有base.learning，旧正例/旧端点项的梯度因冻结为0，保护作用来自含新实体的合法负例。

- replay_start0_w03：从初态440步常规Replay，loss系数固定.3（已有历史Replay探索值），没有共享普通prefix。
- replay_late_w03：复用10epoch普通prefix，后220步同.3，用于同干预时点比较。
- replay_late_rho01：同prefix、后220步Replay附加梯度每个非零步归一到host norm .1；这是强度诊断适配控制，非作者原生Replay。
- anchor_replay_rho01：同上一归一化Replay梯度和joint Adam，附加094/095累计差查询空间约束。用于比较约束更新在相同Replay信号上是否有用。

每seed四臂共用Replay缓存；两个seed八个模型全部保存后统一评价旧TRAIN260498、旧VALID86832和新实体VALID2786，teacher TRAIN/完整VALID直接复用087/088/102。raw/filtered TRAIN-only、旧MRR/Top10掉榜、新三方向；没有旧1024评分重复或完整TRAIN teacher重建。

模型全保存后才读本轮VALID；102已暴露，属于开发确认，不能称未见TEST。新随机只用于必需Replay负采样的一次缓存，不生成新cohort/普通影子/SVD。成本含Replay tape/所有追加训练及评价、历史空间/影子费用另计。不同采样候选数和逐步归一化改变算力/强度，报告实际gradientratio，不能把同.3当同实际保护力度。

无硬超时/独立CPU审计/SHA/归档。Replay是已知基线、约束是已有候选组合，不做原创性或完整snapshot2任务声明。
