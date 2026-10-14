# 076 固定候选方案时点稳健性/公平对照补齐

候选从075 relation_w1.1_rho.15固定，不继续用本轮VALID改参数：relation等权保护p∝relation资格答案数^-1，旧头新gold学习weight1.1，rho .15/32×128；candidate所有超参固定。另matcheduniform同weight/rho、simpleuniform原weight1/rho.1、learningonlyweight1.1。各保护开启时点epoch0/5共8新组，20epoch统一终态。newE仅可训练/oldE,R冻结、seed17/同tape；epoch5来自真实普通前185步Adam状态，0从initial。它是开启时点稳健性，不是跨seed/未见数据确认。

只读复用075shared-cache全部187470oldTRAIN资格/q/candidates，不重构大cache；每条件在自身初态8batch校准，全部学习/保护步骤真实执行。所有8训练完成才VALID，筛选/非支配TRAIN沿063。fixed protocol，不以075接近门槛的rho.2重新称通过。

补齐onset10关系candidate及精确uniform同weight/rho、无weight relationrho.1/globaluniformrho.1的全TRAIN公平比较；旧75/74模型/VALID数组/已有TRAIN metrics复用，不重训/不重复已有评价。补评新结果单独写076onset10-controls，不改历史75资产。检查所有新方向/旧MRR/掉榜/成本，不宣称全面支配每一控制。

无TEST/新seed/独立CPU审计/SHA/归档/硬超时。产物/data/lc/untzh/confirm76_20261014，单seed暴露VALID开发后续有限条件验证，独立确认仍缺。
