# 123 TRAIN初值与简单保护组合

121已完成18普通终态训练、部分评价正在进行。继续对预定义neighbor_mean/query_mean两个有TRAIN语义的初值各加simple absolute及answer anchor保护；三父43/17/29共12新配置。不是从VALID选一个最佳初值，更不宣称未见确认。零/缩尺度/方向shuffle/flip不作为此阶段组合；对应121普通、116低随机普通/保护只读控制。

121 initial/prefix10含Adam/ordinary endpoint复用、不重训前缀；source87/88/110负采样tape/cohort/selections/q/teacher过滤复用。simple absolute为32×128固定合法候选/RMS8batch校准/.1 cap；answer anchor为同32×完整1876合法候选、当前保护梯度norm rho .1、jointAdam后累计相对自身ordinary在线影子的已见新TRAIN跨旧端查询空间正交补约束。space旧E/R不变只读复用。与自身ordinary匹配判断保护附加收益，与公平116低随机同种保护判断初值价值。影子不是独立方法。

十二新模型提交后评价完整旧TRAIN/旧VALID/新VALID，raw/filtered及三新方向、旧MRR/Top10drops都报告，历史费用非零。若组合只保TRAIN而VALID或新方向变差，不宣布有效。所有VALID为开发已暴露，FB只新实体子任务，非完整新增事实。无TEST/硬超时/独审/SHA/归档，无新风险树/收益网络/路由调参，不据中间结果改配方。
