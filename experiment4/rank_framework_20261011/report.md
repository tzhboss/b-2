# 统一 Rank 评价实测报告

已按无过滤、严格大于、Top10重算已有检查点和更新向量。所有方法使用同一指标定义；保留各自的模型、学习单元和保护查询集合。本轮无重新训练或QP求解，完成审计后停止。

同模型、同400编辑请求的Rank/QP对照可以直接作配对比较。持续学习/语言模型编辑的行用于检验其旧知识排名是否保持，不能据跨模型表格宣称统一任务SOTA。旧Top10掉出率以最初进入Top10的查询观察为分母，SafeSucc以完整编辑请求/批次为分母，两者不是同一个数。

| 方法 | 模型/数据 | 旧Top10数 | 掉出Top10 | 排名变化P95 | 新查询Hits10 | SafeSucc |
|---|---|---:|---:|---:|---:|---:|
| KGEditor | EXBert/FB15K237 | 12110 | 1.26% | 2 | 97.31% | 85.87% |
| KGEditor | EXBert/WN18RR | 3320 | 0.51% | 1 | 56.87% | 29.88% |
| IncDE | TransE/ENTITY | 38608 | 9.75% | 8 | 37.07% | — |
| IncDE | TransE/RELATION | 36393 | 21.27% | 102 | 29.59% | — |
| IncDE | TransE/FACT | 34534 | 10.68% | 13 | 26.34% | — |
| IncDE | TransE/HYBRID | 36255 | 10.12% | 10 | 26.48% | — |
| FastKGE | TransE/FB_CKGE | 35864 | 0.00% | 0 | 28.64% | — |
| STARK | TransE/FB_CKGE | 36576 | 0.00% | 0 | 29.25% | — |
| LKGE validation-selection revision | TransE/FACT | 35523 | 15.25% | 29 | 28.36% | — |
| SAGE（探索，原完整性FAIL） | TransE/FACT | 34850 | 13.65% | 20 | 26.47% | — |
| Direct | distmult/FB15k237 | 1084676 | 0.51% | 0 | 100.00% | 32.50% |
| Rank (fraction .125) | distmult/FB15k237 | 1084676 | 0.04% | 0 | 96.00% | 48.25% |
| Support TRAIN adaptation | distmult/FB15k237 | 1084676 | 0.08% | 0 | 93.25% | 54.00% |
| QP TRUE+Prediction + Rank | distmult/FB15k237 | 1084676 | 0.03% | 0 | 96.00% | 57.75% |
| QP TRUE-only + Rank | distmult/FB15k237 | 1084676 | 0.03% | 0 | 96.00% | 58.00% |
| Support all-split literal | distmult/FB15k237 | 1084676 | 0.09% | 0 | 0.00% | 0.00% |
| Support all-split exclude target | distmult/FB15k237 | 1084676 | 0.10% | 0 | 91.25% | 54.00% |
| StrictNull | distmult/FB15k237 | 1084676 | 0.00% | 0 | 3.00% | 3.00% |
| Direct | complex/FB15k237 | 1088369 | 0.53% | 0 | 100.00% | 30.25% |
| Rank (fraction .125) | complex/FB15k237 | 1088369 | 0.04% | 0 | 96.00% | 49.00% |
| Support TRAIN adaptation | complex/FB15k237 | 1088369 | 0.07% | 0 | 93.50% | 53.50% |
| QP TRUE+Prediction + Rank | complex/FB15k237 | 1088369 | 0.03% | 0 | 96.00% | 59.75% |
| QP TRUE-only + Rank | complex/FB15k237 | 1088369 | 0.03% | 0 | 96.00% | 59.75% |
| Support all-split literal | complex/FB15k237 | 1088369 | 0.07% | 0 | 0.00% | 0.00% |
| Support all-split exclude target | complex/FB15k237 | 1088369 | 0.09% | 0 | 91.25% | 53.25% |
| StrictNull | complex/FB15k237 | 1088369 | 0.00% | 0 | 3.00% | 3.00% |

排名变化P95统计所有旧Top10答案的有符号 after-before，可能为0即使一部分答案损坏。完整表另有仅退后样本的P95和最大位移。持续学习的新查询Hits10是最终快照TEST原始排名；KGEditor是暴露edit_test；Rank/QP是同400VALID请求，因此这些新查询率的难度不同。

## 候选扩张诊断

以下固定最终权重，将每个历史候选集合扩到最终实体全集。旧答案自身分数逐条不变；排名变化仅来自加入的竞争候选。这不等于同候选条件下的权重遗忘。

| 方法/数据 | 旧候选下Top10数 | 扩候选后掉出 | 排名变化P95 |
|---|---:|---:|---:|
| IncDE/ENTITY | 37733 | 25.39% | 44 |
| IncDE/RELATION | 32414 | 0.03% | 0 |
| IncDE/FACT | 35102 | 1.53% | 1 |
| IncDE/HYBRID | 35700 | 12.24% | 9 |
| FastKGE/FB_CKGE | 35864 | 11.10% | 15 |
| STARK/FB_CKGE | 36576 | 11.56% | 11 |
| LKGE validation-selection revision/FACT | 34814 | 0.68% | 0 |
| SAGE/FACT | 34005 | 0.63% | 0 |

## 本轮能得出的结论

这些方法可以统一计算排名保护指标。结果不支持笼统断言“过去的方法都没有保护排名”：FastKGE/STARK在固定旧候选集合下确实保留全部旧排名；候选扩张带来的排名损伤是另一个可直接观察的条件。IncDE/LKGE等在固定候选下出现旧Top10掉出，说明其已有保护在这些观察上并不保证排名完全保持。
distmult同400请求：新编辑Hits10均96.00%，SafeSucc由Rank的48.25%到QP TRUE+Prediction的57.75%（增加9.50个百分点）；逐旧知识观察掉出率由0.0428%到0.0338%。TRUE-only的SafeSucc为58.00%，因此此轮未显示增加Prediction约束比TRUE-only更有效。
complex同400请求：新编辑Hits10均96.00%，SafeSucc由Rank的49.00%到QP TRUE+Prediction的59.75%（增加10.75个百分点）；逐旧知识观察掉出率由0.0442%到0.0327%。TRUE-only的SafeSucc为59.75%，因此此轮未显示增加Prediction约束比TRUE-only更有效。

## 分数与排名

共同表包含正例分数变化除以编辑前候选分数标准差，及绝对标准化漂移≤0.001的旧Top10子集掉出率。阈值仅为描述性分箱。分数稳定同时排名损坏可证明这些观察上分数稳定不足以保护排名，不能据此把所有算法目标概括成“只保护分数”。

## 审计与边界

持续检查点：全部query重评分及向量/标量指标交叉核查；每个模型独立NumPy全候选复算960个状态/query，float64分数误差≤1e−12且排名精确一致。
KGEditor：完整native批次文本/条件重建，所有before/after原native排序排名、Top3及logit（≤1e−4）核验；另算strict排名并CPU/GPU交叉检查。保留native共享embedding恢复限制，八个locality探针反复使用，不是完整独立L-Test；edit_test参与检查点选择。
Rank/QP：原更新向量/检查点hash绑定，全部受保护事实排名重算并核对原400例损伤/成功率；抽样全候选标量复算。继承原QP证书，无新QP求解。仅冻结query单候选列、同关系TRAIN原Top10按顺序cap5000，不能称全局实体更新保护。all-split Support额外使用标签。Ridge原失败保留；SAGE仅探索。
所有输入SHA在前后绑定，逐query产物留在远程artifacts且不进入Git。当前表是事后描述性评价，无新留出、新颖性或统计独立性声明。

通过审核作业：12/12；保留失败：[]。
