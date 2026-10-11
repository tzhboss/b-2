# 知识图谱嵌入编辑：远程复核与文献基线实测对照

**2026-10-10 · experiment4 · 结果复用及独立审计**

## 1. 结论与本次完成事项

可以复用已有数据、权重和结果做有边界的对照。当前已有 Rank-Level Locality 的部分复现、同400案例的QP比较，以及远程64维共享参数三目标实验。它们对应不同协议，下面分别报告。30篇综述条目并非30个同任务可替换编辑器；本次没有把它们全部重训或拼成统一SOTA表。

本次在 /data/lc/tzh/experiment4/comparison20261010 新建隔离目录，重放第24轮保存结果的哈希与汇总审计。30个checkpoint哈希、18个编辑文件核对通过，32行汇总成功重建；三个审计/汇总文件与旧版逐字节一致，18个原编辑文件未改变。没有新增训练、模型推理或官方TEST数据。

另对本地037两模型共1600个已保存Support更新执行独立SVD、支持行选择、完整排名及字节绑定审核，共49,360项记录检查，0差异。数值正确性PASS不代表方法性能通过，也不是独立科研评审或原创性认证。

## 2. 原论文数值与公开权重部分复现

论文来源：https://arxiv.org/html/2609.12116v1；公开代码：https://github.com/pa0lai/kge-rank-locality 。下面读取既有final_reproduction_audit.json，不是本次新跑1000个TEST案例。每模型重建1000个案例；作者原逐例清单不可得，两个队列不同，差值不能当作配对胜负。历史TEST已暴露。

| 模型 | 方法 | 论文SafeSucc | 重建队列SafeSucc |
|---|---|---|---|
| distmult | Direct | 23.00% | 23.80% |
| distmult | ValidationBestRank | 32.80% | 33.40% |
| distmult | StrictNull | 1.40% | 1.70% |
| distmult | ValidationBestRidge | 未提供主表对应值 | 1.50% |
| complex | Direct | 23.20% | 21.90% |
| complex | ValidationBestRank | 34.70% | 32.20% |
| complex | StrictNull | 1.30% | 1.70% |
| complex | ValidationBestRidge | 未提供主表对应值 | 1.50% |

Support主表原论文SafeSucc为DistMult36.3%、ComplEx37.7%；它不是此重建TEST表已精确复现的方法。

## 3. 同权重、同400个VALID请求的固定比较

034的两个模型使用公开256维权重，单候选尾列更新；原head查询向量冻结。所有400次包括拒绝和Rank回退。SafeSucc要求目标进入raw Top10且规定范围内的TRAIN原正确事实无一掉榜。VALID已暴露，不能称最终盲测。

| 模型 | 方法 | 成功率 | SafeSucc | 外部损伤/次 |
|---|---|---|---|---|
| distmult | Direct | 100.00% | 32.50% | 0.9675 |
| distmult | Rank(.125) | 96.00% | 48.25% | 0.0750 |
| distmult | Support TRAIN适配 | 93.25% | 54.00% | 0.1450 |
| distmult | TRUE+Prediction QP及Rank回退 | 96.00% | 57.75% | 0.0450 |
| complex | Direct | 100.00% | 30.25% | 1.1850 |
| complex | Rank(.125) | 96.00% | 49.00% | 0.0925 |
| complex | Support TRAIN适配 | 93.50% | 53.50% | 0.2350 |
| complex | TRUE+Prediction QP及Rank回退 | 96.00% | 59.75% | 0.0700 |

QP相对Rank的SafeSucc提升9.50/10.75个百分点，结论限定于这个固定队列。TRAIN支持适配不是原作者all-split Support，不能由此称战胜原论文最强完整系统。

## 4. 036机制消融：收益归属必须修正

| 模型 | 方法 | SafeSucc | 外部损伤/次 |
|---|---|---|---|
| distmult | Rank(.125) | 48.25% | 0.0750 |
| distmult | TRUE-only QP及Rank回退 | 58.00% | 0.1475 |
| distmult | TRUE+Prediction QP及Rank回退 | 57.75% | 0.0450 |
| complex | Rank(.125) | 49.00% | 0.0925 |
| complex | TRUE-only QP及Rank回退 | 59.75% | 0.1750 |
| complex | TRUE+Prediction QP及Rank回退 | 59.75% | 0.0700 |

加入预测集合约束的额外SafeSucc贡献未通过固定门槛：DistMult -0.25个百分点，ComplEx 0.00个百分点。其主要支持证据是减少已覆盖查询的旧事实损伤。没有TRAIN查询键的外部事实桶中，Rank原本损失为0；当前低事件数不能支持未知查询泛化改善。

本次复用了现有TRUE-only与TRUE+Prediction两臂。Prediction-only QP尚未实测；不能称完整三臂消融已经完成。

## 5. 037 all-split Support：声明约定的重建

冻结同034400请求，按原论文软目标 min(q^Tδ-Δ)^2+||Cδ||²，最多256个最高原始范数same-tail支持行。原作者目标排除、重复/并列处理、数值内核与原队列未完全公开，因此保留两种事先声明的约定，不挑表现较好的版本冒充原作。

| 模型 | 目标处理约定 | 成功率 | SafeSucc | 外部损伤/次 |
|---|---|---|---|---|
| distmult | 目标保持可选 | 0.00% | 0.00% | 0.1325 |
| distmult | 排除全部请求事实 | 91.25% | 54.00% | 0.1600 |
| complex | 目标保持可选 | 0.00% | 0.00% | 0.1850 |
| complex | 排除全部请求事实 | 91.25% | 53.25% | 0.2475 |

在这个队列，软目标的目标处理约定显著影响结果。目标本身被支持项惩罚时，提升与保护要求可能冲突；这不证明原论文实现错误。排除请求事实的重建SafeSucc为54.00%/53.25%，低于固定QP的57.75%/59.75%，属于同队列、不同实际信息输入的描述性比较，不是精确原作者最强方法战胜或公平部署结论。

037显式读取TEST事实标签作为优化支持输入，没有新计算TEST评价分数/排名。外部VALID标签也可进入支持矩阵，因此其外部损伤不能当作整个未参与优化的留出评价。

## 6. 远程 experiment4：64维共享Head/Tail三目标

远程六个模型：DistMult/ComplEx各3训练种子401/402/403，25epochs，64维；每种81次尝试，B=5.2，成本为逐实体L2范数之和。三个目标可能涉及不同实体，不等于同一tail向量的三请求联合编辑。该表使用跨关系TRAIN-heldout损伤，不是034的SafeSucc。

| 模型 | 方法 | 三目标成功 | 跨关系损伤/次 | L2,1/次 |
|---|---|---|---|---|
| DistMult | Direct | 80/81 | 0.432 | 1.739 |
| DistMult | Soft3_LocalGuard | 80/81 | 0.321 | 1.876 |
| DistMult | Soft3_AllGuard | 71/81 | 0.654 | 2.234 |
| DistMult | Rank125_ReleasedKernel | 68/81 | 0.827 | 2.281 |
| DistMult | Rank1of64_ReleasedKernel | 73/81 | 1.309 | 2.088 |
| DistMult | AdaptiveSoft_LocalGuard | 80/81 | 0.395 | 1.933 |
| DistMult | AdaptiveSoft_AllGuard | 80/81 | 0.765 | 3.014 |
| DistMult | AdaptiveRank_ReleasedKernel | 73/81 | 0.877 | 2.702 |
| ComplEx | Direct | 78/81 | 0.148 | 1.919 |
| ComplEx | Soft3_LocalGuard | 78/81 | 0.123 | 2.042 |
| ComplEx | Soft3_AllGuard | 71/81 | 0.420 | 2.295 |
| ComplEx | Rank125_ReleasedKernel | 68/81 | 0.444 | 2.301 |
| ComplEx | Rank1of64_ReleasedKernel | 73/81 | 0.593 | 2.331 |
| ComplEx | AdaptiveSoft_LocalGuard | 78/81 | 0.160 | 2.090 |
| ComplEx | AdaptiveSoft_AllGuard | 78/81 | 0.556 | 2.907 |
| ComplEx | AdaptiveRank_ReleasedKernel | 73/81 | 0.519 | 2.817 |

LocalGuard相对Direct跨关系损伤差的查询聚类探索性95%区间：DistMult [-0.2564, 0.0128]；ComplEx [-0.0897, 0.0385]，均含零。此协议下尚未支持显著超越Direct。作者Rank数值内核用于本地多目标扩展，不能用此结果否定原论文完整方法。

## 7. 30篇综述条目的复现路由

method_routing.csv逐条保留原综述的标题、来源、代码链接和协议分类，并增加本轮可比性判断。链接“存在于原综述”不等于本次验证所有仓库可运行。

- Rank-Level Locality：优先直接对照；已有Direct/Rank/StrictNull/Ridge部分复现，Support采用显式约定重建。

- KGEditor与KBS2026：LM-based KGE，需其架构、EDIT/ADD数据、训练编辑器及相同rank-damage指标，不能把本文SafeSucc与其论文数字硬比。

- AIR、IncDE、FastKGE、SIT、SAGE、DebiasedKGE、STARK等：多快照动态/持续KGE，先建立快照、实体增长、MRR/Hits/遗忘和成本协议，再谈复现。

- GEM、OGD、AlphaEdit等：约束思想与创新性审计前作，不能直接作为浅层KGE同任务SOTA数值。

本次只直接网页核对Rank论文/仓库与KGEditor出版页；其余30条元数据继承原范围综述，不能称本次全部独立全文核验。

## 8. 去留判断与下一步

GO：形成有出处、分协议、可复核的实验对照材料。尚无全SOTA战胜、新算法优先权或投稿GO。037正确性门槛通过，性能为描述性结果；目标处理差异应作为复现限制保留。

若继续方法研发，先补Prediction-only机制臂和更严格的未知查询损伤评价；再单独注册一个共同tail向量的三请求任务，并对实际共享Head/Tail实体表重新评分。此前的冻结尾列证书不能自动扩展为全共享模型安全保证。当前不扩大模型、参数网格或新的TEST评价来追求正结果。

## 9. 可复核文件与执行边界

本地对照目录：C:/Users/11500/Documents/ChatGPT/b会文件夹/kge-remote-comparison。远程重放目录：/data/lc/tzh/experiment4/comparison20261010 。remote-evidence/replay_manifest.json记录逐字节相同及原始文件不变；comparison_results.csv提供所有分面结果，method_routing.csv提供30篇路由。

037审计原件保存在rank-locality-followup30/artifacts/{distmult,complex}-all-support400-audit/audit.json；原034/036结果与失败历史保留。本报告生成器和manifest.json记录所消费文件SHA-256，PDF原综述转换版保留原文字，不把本轮新结果静默写入原综述。
