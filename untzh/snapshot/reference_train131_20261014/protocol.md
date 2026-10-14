# 131 固定130初始化的终态和保护比较

不再拟合或选择初始化。130两个父（FB17 DistMult、WN ComplEx）×TRAIN-only/history-reference×constrained/normmatched_random/reverse三方向：12普通20epoch轨迹+各自后10epoch答案anchor，共24训练配置。WN TRAIN-only约束无激活，实质是ridge缩幅，对应方向控制不当独立算法。zero/half自身普通及同保护结果只读复用。

所有24终态提交后统一完整旧TRAIN/旧VALID、新VALID评价（FB新2786/WN新3720），另按128群组报告校准与留出旧指标。历史参考含额外旧gold/rank监督，校准不混留出；TRAIN-only单列。新VALID不用来拟合、选初值、选择动作或改规则。

Adam .007、旧E/R冻结、既存tape/cohort/query空间与保护选择固定，普通轨迹缓存prefix含optimizer供保护续训，在线普通shadow匹配自身终态且计费。答案anchor：32旧TRAIN查询×完整合法新候选，保护梯度norm rho .1；jointAdam后累计相对自身普通轨迹位移投到已见跨新旧TRAIN查询空间正交补。FB1876/WN4093新候选。不能保证双新/未见排名安全。只读已有130初态eligible排名，新增prefix/endpoint阶段配对分解。

以同范数随机/反向及强fixedzero/half同保护控制判断机制，不以初态下降替代终态有效。报告完整和留出旧MRR/掉榜、raw/filtered、新三方向、恢复与新损伤、训练/初始化/影子/评价费用。开发数据已暴露，非独立未见确认；无TEST/硬超时/独审/SHA/归档。
