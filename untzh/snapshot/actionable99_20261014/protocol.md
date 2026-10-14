# 099缓存诊断，启动前固定

不训练、不重新计算神经完整排名。读取094/097/098同一088查询rank，配对tree和uniform_tree_norm/random_tree_norm/random_pool；TRAIN按旧阶段、cohort/非cohort、tree选中答案/查询分层，VALID仅报告现存结果，不参与选规则。

几何定义：固定旧TRAIN查询q、各新实体固定跨新旧TRAIN空间的正交补P_j，计算||P_j q||²/||q||²，作为允许方向的一阶可干预程度。仅用既有cohort16384及其128合法固定候选；查询向量/空间/初始前10epoch参数复用。缓存完整128候选fraction、prefix10最大竞争候选的fraction与均值，后续不用重算。无随机重抽、future标签或TEST。

固定分桶[0,.25),[.25,.5),[.5,.75),[.75,.9),[.9,1]，报告tree相对同累计差norm uniform的TRAIN救回/损害。几何和rank关联为开发诊断，不能当独立泛化或实际Adam收益预测。若几何变化小或无关系，不强行训练学习器或追阈值。

记录numpy配对时间、GPU几何和缓存时间/bytes、数值基本检查；无硬超时/独立CPU审计/SHA/归档。远程独立新目录/data/lc/untzh/actionable99_20261014，历史资产只读。
