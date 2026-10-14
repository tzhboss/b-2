# 074 保护覆盖、分布与样本方差

DistMult200/FB-CKGE0→1/seed17/oldE,R冻结/058prefix10/后370步同学习tape。12配置：既有16384cohort均匀batch32/128；完整187470TRAINfiltered资格答案均匀batch32/128/512、query等权batch32/128、relation等权batch32/128、boundarytop20%(81.25%target+18.75%uniform)batch32/128；globalold-only语义batch128。

query sampling每答案概率1/同query答案数、relation同理；不借采样分布改资格分母。所有答案预备128固定随机合法新候选；cohort身份候选逐位复用058，避免换candidate造成cohort32结果差异。新global候选seed17抽样/排TRAIN正例，q可缓存。每配置按自己相同batch/分布初态8步RMS校准rho .1，逐步cap .1；所以batch扩增是成本/方差变量，并非等候选评分预算。所有candidate生成/存储费用记录，不作为算法创新。

全部12训练完后VALID，筛选/非支配TRAIN沿063；全报结果，固定旧资格和三个新方向。候选池coverage/differentqueries统计。无TEST/独立CPU审计/SHA/归档/硬超时，产物/data/lc/untzh/coverage74_20261014，单seed开发搜索。
