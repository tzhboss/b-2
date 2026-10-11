# 同一Rank协议的实测对照

两模型、各400个已暴露VALID请求、公开256维权重。无过滤全实体候选、strict >排名、Top10；同关系TRAIN原Top10真事实、顺序上限5000。SafeSucc＝目标进入Top10且此保护范围内无一旧正确答案掉出。这里只更新候选尾列，head查询冻结。

复用034/036/037已审的5600个旧更新向量，在远程全部重新评价，旧目标排名和损伤字段逐例完全一致；新增800个StrictNull更新已核验。Ridge的800个更新保留，但独立向量门槛失败，不进入下面通过核验的性能表。没有训练、QP重新拟合、数据重选或参数搜索。

| 方法 | DistMult目标成功 | DistMult SafeSucc | ComplEx目标成功 | ComplEx SafeSucc |
|---|---:|---:|---:|---:|
| Direct | 100.00% | 32.50% | 100.00% | 30.25% |
| Rank(.125) | 96.00% | 48.25% | 96.00% | 49.00% |
| Support TRAIN适配 | 93.25% | 54.00% | 93.50% | 53.50% |
| TRUE+Prediction QP＋Rank | 96.00% | 57.75% | 96.00% | 59.75% |
| TRUE-only QP＋Rank | 96.00% | 58.00% | 96.00% | 59.75% |
| Support all-split：目标可选 | 0.00% | 0.00% | 0.00% | 0.00% |
| Support all-split：排除目标 | 91.25% | 54.00% | 91.25% | 53.25% |
| StrictNull | 3.00% | 3.00% | 3.00% | 3.00% |

TRUE+Prediction QP＋Rank相对Rank的SafeSucc提高9.50/10.75个百分点；配对新增安全成功38/43例，原Rank安全成功丢失0例。目标成功率均为96%。这些是已知队列上的描述性比较，不是最终盲测、SOTA或全局安全结论。

TRUE-only QP的SafeSucc为58.00%/59.75%，预测约束没有额外SafeSucc收益。TRUE+Prediction的外部旧事实损失/次为0.045/0.070，TRUE-only为0.1475/0.175，Rank为0.075/0.0925；这里外部VALID的部分查询已被TRAIN查询键覆盖，不能一概称未见查询泛化。

Support TRAIN为适配；all-split是明确目标函数和目标处理约定的重建，含额外VALID/TEST标签。不能称精确复现作者最强完整方法或公平部署胜利。目标可选约定的0%结果保留，不据此称原论文有bug。

StrictNull两模型旧事实无损率100%，目标成功率仅3%；它显示保护与可编辑性的取舍。Ridge初次审计在两个模型都失败；续审保持相对向量误差<1e-5、驻点残差<1e-7门槛不变，收集全部失败，不放宽门槛、不重新求解/选择生产结果。初次canonical仍为failed_retained，续审仅提升通过的八个方法。详细失败见results/KGE-RANK-COMP-001-AUDIT-V2/ridge-failure-disposition.json。

IncDE/FastKGE/STARK/LKGE和KGEditor可以使用同样的raw排名与旧Top10保留指标，但原始架构、增量输入和训练信息与该单次编辑不同；它们此前的MRR/filtered排名结果未混入此表。直接同任务适配尚未完成，详见compatibility.md。

之前041的真实共享head/tail、三目标、全关系合同仍是负结果：当前预测保护QP＋Rank的全局Joint SafeSucc两模型均2.5%。这与本表保护范围不同，不能用本表局部提升替代共享模型安全结论。

执行位置：用户experiment4/rank_comparison_20261010。两CPU进程、各2个BLAS线程；本轮是向量重放与数值审计，未训练神经编辑器，不需要GPU。每模型生产重放耗时：distmult 74.46s, complex 83.06s；该时间不是历史QP求解/训练成本，不能用于跨方法效率胜负。

数值核查使用独立SVD/最小二乘、所有声明保护行的单列排名代数和抽样直接全候选重评分；不是独立科研同行评审。完整轻量审计与输入SHA绑定见results/目录。
