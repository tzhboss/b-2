# 新增实体共同任务：已审计结果与失败状态

99/108个注册方法/种子任务通过审计。运行、排队与失败均逐seed记录；失败不填成绩。

after均为11258候选，旧TEST74428／新TEST12406方向查询，rank=1+count(score>gold)，无过滤。当前是已暴露的开发比较。

| 方法 | 模型/维数 | 审计seed数 | 旧MRR | 新MRR | 旧Top10掉榜% | 新Hits10% | 增量秒 | 状态17/29/43 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| LKGE | TransE/200 | 3/3 | 0.15561 | 0.14995 | 8.43386 | 27.27175 | 112.98619 | 通过/通过/通过 |
| IncDE | TransE/200 | 3/3 | 0.15485 | 0.14125 | 9.52942 | 25.65963 | 561.01521 | 通过/通过/通过 |
| FastKGE | TransE/200 | 3/3 | 0.13759 | 0.14946 | 13.65470 | 28.25515 | 108.92178 | 通过/通过/通过 |
| STARK | TransE/200 | 3/3 | 0.13563 | 0.15205 | 14.84426 | 28.36531 | 315.51096 | 通过/通过/通过 |
| SAGE | TransE/210 | 3/3 | 0.15120 | 0.15615 | 14.14197 | 29.81890 | 657.59482 | 通过/通过/通过 |
| KGEditor | EXBert/768 | 0/3 | — | — | — | — | — | 运行/运行/运行 |
| RandomExpansion | DistMult/200 | 3/3 | 0.15295 | 0.00547 | 12.87999 | 0.99952 | 2.40664 | 通过/通过/通过 |
| RandomExpansion | TransE/200 | 3/3 | 0.15762 | 0.04574 | 0.00000 | 9.57601 | 3.33721 | 通过/通过/通过 |
| NewEntityOnly | DistMult/200 | 3/3 | 0.16792 | 0.14117 | 2.59145 | 25.66231 | 25.40331 | 通过/通过/通过 |
| NewEntityOnly | TransE/200 | 3/3 | 0.14579 | 0.14666 | 8.33692 | 27.47864 | 62.64763 | 通过/通过/通过 |
| FullFinetune | DistMult/200 | 3/3 | 0.12630 | 0.09655 | 31.20603 | 18.97200 | 157.61415 | 通过/通过/通过 |
| FullFinetune | TransE/200 | 3/3 | 0.16379 | 0.12914 | 21.72415 | 23.77882 | 318.58145 | 通过/通过/通过 |
| Replay | DistMult/200 | 3/3 | 0.13110 | 0.11446 | 28.80157 | 21.64007 | 53.82696 | 通过/通过/通过 |
| Replay | TransE/200 | 3/3 | 0.16144 | 0.14380 | 22.15681 | 26.37433 | 102.95800 | 通过/通过/通过 |
| DirectNovel | DistMult/200 | 3/3 | 0.01461 | 0.09138 | 92.45015 | 19.43952 | 64.48918 | 通过/通过/通过 |
| DirectNovelBounded | DistMult/200 | 3/3 | 0.01461 | 0.09138 | 92.45015 | 19.43952 | 72.26852 | 通过/通过/通过 |
| LearnedProposal | DistMult/200 | 3/3 | 0.16792 | 0.14117 | 2.59145 | 25.66231 | 48.32123 | 通过/通过/通过 |
| LearnedProposalQPPred | DistMult/200 | 3/3 | 0.16883 | 0.13981 | 1.61312 | 25.03896 | 189.31378 | 通过/通过/通过 |
| LearnedProposalQPPredEuclideanR2 | DistMult/200 | 3/3 | 0.16883 | 0.13981 | 1.61312 | 25.03896 | 168.64476 | 通过/通过/通过 |
| LearnedProposalQPTrue | DistMult/200 | 3/3 | 0.16815 | 0.14081 | 2.36074 | 25.42050 | 66.32412 | 通过/通过/通过 |
| LearnedProposalRank | DistMult/200 | 3/3 | 0.12348 | 0.08654 | 30.37235 | 17.40018 | 44.47452 | 通过/通过/通过 |
| LearnedProposalRankQPPred | DistMult/200 | 3/3 | 0.14942 | 0.07626 | 14.22419 | 14.61121 | 372.09956 | 通过/通过/通过 |
| LearnedProposalRankQPTrue | DistMult/200 | 3/3 | 0.14292 | 0.07911 | 18.69885 | 15.43339 | 132.30388 | 通过/通过/通过 |
| LearnedProposalTaskQPPred | DistMult/200 | 3/3 | 0.16841 | 0.14077 | 1.87018 | 25.22167 | 390.72140 | 通过/通过/通过 |
| QPPredRankNovel | DistMult/200 | 3/3 | 0.00138 | 0.02971 | 99.73874 | 8.39379 | 661.61496 | 通过/通过/通过 |
| QPPredRankNovelBounded | DistMult/200 | 3/3 | 0.01501 | 0.05984 | 90.88056 | 13.51228 | 833.46924 | 通过/通过/通过 |
| QPTrueRankNovel | DistMult/200 | 3/3 | 0.00132 | 0.02964 | 99.77282 | 8.29975 | 633.81061 | 通过/通过/通过 |
| QPTrueRankNovelBounded | DistMult/200 | 3/3 | 0.01473 | 0.05934 | 90.77819 | 13.33495 | 845.37381 | 通过/通过/通过 |
| RankTruncatedNovel | DistMult/200 | 0/3 | — | — | — | — | — | 失败留档/失败留档/失败留档 |
| RankTruncatedNovelBounded | DistMult/200 | 3/3 | 0.00526 | 0.01474 | 98.32034 | 2.50685 | 69.82147 | 通过/通过/通过 |
| RidgeNovel | DistMult/200 | 0/3 | — | — | — | — | — | 失败留档/失败留档/失败留档 |
| RidgeNovelBounded | DistMult/200 | 3/3 | 0.00263 | 0.00896 | 99.56897 | 1.17416 | 122.45531 | 通过/通过/通过 |
| StrictNullNovel | DistMult/200 | 3/3 | 0.00178 | 0.00174 | 100.00000 | 0.16927 | 67.22976 | 通过/通过/通过 |
| StrictNullNovelBounded | DistMult/200 | 3/3 | 0.03396 | 0.00463 | 79.00379 | 0.81412 | 72.80713 | 通过/通过/通过 |
| SupportTrainNovel | DistMult/200 | 3/3 | 0.01461 | 0.09138 | 92.45015 | 19.43952 | 64.89523 | 通过/通过/通过 |
| SupportTrainNovelBounded | DistMult/200 | 3/3 | 0.01461 | 0.09138 | 92.45015 | 19.43952 | 70.87483 | 通过/通过/通过 |

数值仅取已通过审计seed的均值；不足3seed的行不能作为完整三seed比较。各seed状态、标准差、同池掉榜、位移与阶段成本在table.json。旧掉榜分母为原7505池Top10成功查询，同池辅助分母为11258初始化池成功查询。

Rank／QP采用同值初始化的float64共享实体连续更新，单轮新TRAIN扫遍；基线SGD有不同轮次和成本，不能据此宣布等成本胜利。Bounded是公开的稳定性适配，半径只依赖旧底座及随机扩展初态；它不是原论文公式或已确认创新。

原始Rank、Ridge的数值不稳定及QP审计错误保留；单列原始与稳定版本，不能将共同共享实体适配失败外推为原论文独立编辑失败。QP被接受时只保护注册TRAIN集合的当前边界；初始化损失、回退及未保护TEST无硬保证。

KGEditor使用干净语言模型底座、原生元编辑器和累积在线编辑，另计底座／meta成本，描述文本及预训练知识属于额外信息。尚未通过完整审计时不填最终成绩。

LearnedProposal各行复用已审计NewEntityOnly训练权重，以固定三seed消融SVD、TRUE／Prediction与最近投影。增量秒包含复用SGD阶段加投影阶段，JSON另列阶段耗时；复用不等于新重训，也不是新算法确认。所有候选仍需真实旧／新排名和预设新知识效用门槛。JSON还按“新查询头／旧答案、旧查询头／新答案、两端均新”拆分新知识效果，保留全部12406查询。

FastKGE／STARK旧E/R不变仍可能旧答案掉榜，支持参数保持不足以保证新增候选竞争下的排名保持；不能概括所有旧方法仅保护分数。LearnedProposalQPPred相对同底座NewEntityOnly降低旧Top10损失，同时牺牲少量新MRR；不支持全面优于所有文献基线或原论文的结论。

实现边界：STARK消费距离列；IncDE排序文件被作者loader覆盖、degree分支复用betweenness；SAGE旧producer元数据FAIL仍保留。完整来源适配及审计限制沿用已注册协议。审计核查全查询身份/聚合与固定完整候选评分行，不等于独立重训所有优化器。

## 独立VALID效用复算

以下是固定开发集门槛，非最终留出认证：三seed平均新MRR至少保留98%，平均旧Top10损失最多为源模型75%，且每seed旧损失不恶化。18个任务全量VALID复算通过；审计通过不等于科学收益门槛通过。

| 方案 | 固定开发门槛 |
|---|---|
| LearnedProposal | NO_GO_FOR_THIS_FIXED_COMBINATION |
| LearnedProposalQPTrue | NO_GO_FOR_THIS_FIXED_COMBINATION |
| LearnedProposalQPPred | PASS_DEVELOPMENT_UTILITY_SCREEN |
| LearnedProposalRank | NO_GO_FOR_THIS_FIXED_COMBINATION |
| LearnedProposalRankQPTrue | NO_GO_FOR_THIS_FIXED_COMBINATION |
| LearnedProposalRankQPPred | NO_GO_FOR_THIS_FIXED_COMBINATION |

## 新知识检索方向的代价

总体新MRR保留约99%不能代表每个方向都保留99%。以下按同一批完整新TEST查询拆分；这些已暴露开发结果不作为未见留出认证，也不回写已冻结的VALID门槛。

| 检索方向 | 指标 | 无QP | TRUE＋Prediction QP | 保留比例 |
|---|---|---:|---:|---:|
| 新头实体查询／旧答案 | raw_mrr | 0.25074 | 0.25165 | 100.36% |
| 新头实体查询／旧答案 | hits10 | 0.44782 | 0.45065 | 100.63% |
| 旧头实体查询／新答案 | raw_mrr | 0.03319 | 0.02984 | 89.93% |
| 旧头实体查询／新答案 | hits10 | 0.07546 | 0.06055 | 80.23% |
| 两端均新 | raw_mrr | 0.13296 | 0.13013 | 97.87% |
| 两端均新 | hits10 | 0.20492 | 0.19672 | 96.00% |

旧头实体查询／新答案方向MRR下降约10%、Hits10下降约20%；当前QP方案仍存在新增答案检索代价。该方向是后续优化需要解决的问题，不能仅据总体VALID门槛通过就宣称新知识写入已无明显代价。

KGEditor17号底座原float32 CPU排名审计失败留档：一个近同分候选使rank240/241不一致。独立注册的016复审采用CPU float64，64查询/480320分数在原1e-4容差内且排名全相同；GPU float32重新加载分数误差0。017用同一权重的独立底座目录继续，后续元训练／在线／最终审计门槛不变。008失败及所有继续尝试可在JSON的attempt_history核对。

## 任务方向投影的固定验证

019比较欧氏与TRAIN新答案评分方向加权的TRUE＋Prediction投影，各三seed。020独立复算要求整体新MRR至少保留98%、旧掉榜最多为源模型75%、每seed旧损伤不恶化，并要求新答案检索方向MRR至少保留98%、Hits10至少保留95%；加权方案还须不劣于欧氏控制。未完成时不能认定优化有效。

| 方法 | 固定方向验证结论 |
|---|---|
| LearnedProposalQPPredEuclideanR2 | NO_GO_FOR_THIS_FIXED_COMBINATION |
| LearnedProposalTaskQPPred | NO_GO_FOR_THIS_FIXED_COMBINATION |

## TRAIN保护范围诊断

021在16757条旧头／新答案TRAIN事实中检查到三seed平均67.44%落在初态保持的原第10名边界上。冻结旧E/R且保持严格边界时，这些新答案仍至少有10个旧候选分数更高；两类端点、三seed全部旧候选独立评分均验证rank下界11。该结果只解释一个充分的TRAIN阻碍子集，不是VALID/TEST全损伤因果归因。

固定加权候选未通过020门槛，保留负结果；不扩大系数或挑选seed。共享向量构造例显示，调整一个旧竞争候选可以保住旧真答案第10名、让新答案进入前10，同时保持未受影响关系的预测集合；这只是构造可行性，真实旧参数改动仍须重算所有共享head/tail保护查询。细节见plan/review/task-projection-disposition.md。