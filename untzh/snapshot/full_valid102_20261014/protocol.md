# 102扩大VALID覆盖：不训练、只追加未评分查询

全部所选模型已保存终态，才开始此次完整VALID评价。新查询直接来自作者FB_CKGE：snapshot0+1的全部旧VALID；snapshot2的全部含新实体VALID（2786），不混未训练的新增旧—旧子任务。

seed29固定12模型：普通、简单absolute/.075、强fullabsolute、anchor_uniform、答案树、risk×geometry、geometry-only、query tree/random/uniform、tree强度uniform控制。seed17固定6模型：普通、absolute/.075、fullabsolute、normalized abs128、anchor_uniform。不会按本轮指标增删模型或重调参数。

按原seed124两次permutation验证原1024old/new选择，原1024teacher和模型rank逐条复用，仅评分其余行。每seed只计算一次新增oldVALID teacher（仅原11258旧候选），再按同teacher资格计算旧Top10掉榜。共享查询/余集一次保存，所有模型TRAIN-only raw/filtered、三新方向保持/MRR/Hits1/Hits10一致。旧TRAINrank与此前指标不重算。

记录全部旧VALID与新实体VALID范围、源新VALID旧—旧排除数、每seed资格、重复源行情况若存在照原作者分布报告。旧E/R正常冻结基本检查，不独立CPU审计。基础加载/teacher追加/每模型追加评价费用单独记录，历史成本不当免费。

这是扩大已暴露开发VALID，不能称独立未见TEST，后续所有使用这轮评价仍应标开发。无训练、重建SVD、影子、cohort、学习tape或teacher TRAIN全量；无TEST/硬超时/独审/SHA/归档。不因更多查询自动把已知方法当原创。
