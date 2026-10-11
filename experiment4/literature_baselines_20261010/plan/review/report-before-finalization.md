# 远程文献基线复现报告（持续更新）

更新时间：2026-10-10T13:36:34.069287+00:00。实验、数据、模型及完整日志主要位于 `/data/lc/tzh/experiment4/literature_baselines_20261010`，文献来源 `/data/lc/tzh/kge_editing_literature_20261010`。IncDE、FastKGE/STARK、LKGE修订对照及候选集合诊断均已归档；SAGE原完整性失败保留。KGEditor WN18RR已完整20epochs并获带恢复限制的数值审计证据，FB15k237仍运行。本轮尚未整体完成。状态以远程canonical现场为准。

## 1. 已完成且审计通过：IncDE原生四划分

每格“实测/论文”仅作单种子与论文表值的描述性对照。原生单种子3407、TransE200、5快照、200epoch上限及patience5早停；valid.txt选检查点，使用历史查询快照候选/累计过滤和double-argsort。不是多种子均值或统一当前实体集合的排名。

| 划分 | MRR实测/论文 | H1实测/论文 | H3实测 | H10实测/论文 | 原生训练计时 |
|---|---:|---:|---:|---:|---:|
| ENTITY | 0.253/0.253 | 0.150/0.151 | 0.295 | 0.448/0.448 | 2626.0s |
| RELATION | 0.198/0.199 | 0.110/0.111 | 0.234 | 0.367/0.370 | 2073.7s |
| FACT | 0.219/0.216 | 0.129/0.128 | 0.252 | 0.395/0.391 | 2550.5s |
| HYBRID | 0.225/0.224 | 0.132/0.131 | 0.264 | 0.401/0.401 | 2483.2s |

四路原生数字接近论文Table2；ENTITY H1比表值低.001，RELATION MRR/H1/H10低.001/.001/.003；FACT高.003/.001/.004，HYBRID高.001/.001/.000。单种子变化不能证明改善。论文primary DOI10.1609/aaai.v38i8.28722，原表与来源核验存 `papers/incde-reference/` 和 `tables/incde-paper-targets.json`。

审计约779万项文件/数组一致性断言不等于独立实验次数。实际四个训练运行，各19逐查询排名、5最佳检查点；合计3840条抽样独立重推理全匹配。所有已登记输入、源代码、选中epoch文件和产物hash均绑定，76份排名文件指标独立复算。未全量重推理前期检查点；4pre-FWT/运行只核查查询/范围/过滤/指标，未重建初始RNG；未舍入验证指标选择无法完整独立认证。`results/KGE-LIT-001`为PASS_WITH_LIMITATIONS。

## 2. 已完成且审计通过：FastKGE / STARK同划分

| 方法 | MRR | H1 | H3 | H10 | 原生训练计时 | 总进程时间 |
|---|---:|---:|---:|---:|---:|---:|
| FastKGE | .223 | .131 | .256 | .402 | 449.15s | 577.59s |
| STARK | .226 | .134 | .261 | .408 | 1315.27s | 1462.41s |

FB_CKGE原始划分逐字节相同，seed3407，完整五快照和作者200epoch上限/早停。FastKGE接近论文Table3的.223/.131/.257/.405；论文时间405s来自3090Ti/五次均值，与本次3090单次时间不同。STARK实测比FastKGE MRR高.003、H10高.006，训练计时约2.93倍；不是因果消融或多种子显著性结论。

独立审计361/504项均0失败，各19排名/5best核查，各120抽样排名全部精确重推理匹配。STARK仅公开实现复现：源代码读结构距离列而不是PageRank，完整论文未取得。`results/KGE-LIT-002`为PASS_WITH_LIMITATIONS。

## 3. SAGE：数值通过，原producer完整性失败保留

默认FACT seed55，完整维度180/210/250/270/290，五快照。实测MRR.220/H1.134/H3.249/H10.387，原生训练1510.42s，总进程1779.57s。825项科学输入/产物检查0失败，29排名/5best/2结果JSON覆盖，120抽样排名CPU重推理全部匹配。

总体仍为FAIL_RETAINED：原仓库的 `provenance/preparation_manifest.json` 注册后发生9项元数据leaf变更，注册时的原始元数据与失败receipt均保留。实际运行worktree所有300项及外部2wheel均匹配master，代码/数据/配置不变；变更是3个已冻结观测patch源码hash说明、2导出默认值及4HTTP记录字段，不被所查native代码读取。证据支持SUPPORTED_NONCONSUMED_METADATA_DRIFT，属于带限制的数值证据，不是干净通过。

SAGE与IncDE FACT三元组多重集合一致，行顺序/种子/维度/训练不同；目前不能将.220与.219的微小差异称方法收益。公开实现预设维数，未确认其完整论文动态蒸馏机制；单种子不等于论文五种子均值。`results/KGE-LIT-003`完整保存限定判定和元数据差异。

## 4. Pons动机固定候选集合诊断：独立派生数值与来源审计通过

所有四路都在同一个最终检查点、相同查询和分数上比较五臂，无新训练/调参。约496208条查询的原生排名全量精确重放通过。表中未舍入按查询数加权；旧候选double-argsort和strict-greater本次均一致，未出现得分tie。

| 划分 | 原生历史候选/历史filter MRR | 最终候选/同历史filter MRR | 相对变化 | 最终候选/最终filter严格rank MRR | 查询数 |
|---|---:|---:|---:|---:|---:|
| ENTITY | 0.253196 | 0.184402 | -27.17% | 0.199010 | 124054 |
| RELATION | 0.198343 | 0.198136 | -0.10% | 0.198136 | 124052 |
| FACT | 0.218612 | 0.215244 | -1.54% | 0.235914 | 124050 |
| HYBRID | 0.224777 | 0.186430 | -17.06% | 0.201988 | 124052 |

ENTITY/HYBRID的竞争实体影响明显；RELATION影响小；FACT更新最终过滤集后数值可上升，说明候选与过滤变化不能混为单一效果。最终filter使用后续已知真三元组，仅是评价协议对照，非新增模型信息或预测收益。本结果是独立实现的Pons动机诊断，不能称其八数据集全复现或新发现。独立CPU派生审计约1094万项一致性检查0失败，不等于独立实验次数；查询/过滤/必要rank恒等式、指标和全部来源hash通过。未另行独立重算全部最终候选分数，保留这一限制。`results/KGE-LIT-005`为PASS_DERIVED_NUMERICS_WITH_LIMITATIONS。完整五臂逐例文件存 `runs/candidate-diagnostic/`。KGE-LIT-005旧prefix误多选pre-FWT的机械assert失败和协调器启动失败均保留；科学protocol未改。

## 5. KGEditor部分完成；LKGE修订对照完成

KGEditor固定公开作者源码/HF模型和数据修订，单种子0、完整20epochs。WN18RR已终端exit0、14900 optimizer steps，源输入未变，总进程4065.36s；FB15k237仍在GPU4运行，不能将父KGE-LIT-004写为完成。

| WN18RR暴露评估 | 实测 | 论文表值 |
|---|---:|---:|
| Succ@1 | .307229 | .833 |
| Succ@3 | .414458 | .844 |
| ER_roc | .868610 | .991 |
| RK@3 | .967693 | .956 |
| RK_roc | .182034 | .256 |

明显未达到论文成功率。完整公开实现数值运行与原论文指标匹配分别判定；不能因得到更弱的数值就称强baseline已被击败。上述是native exposed evaluation：edit_test既用于选best也用于最终评估。WN830条edit、3320次locality重复暴露仅832个unique stable行，不是完整5003个L-Test；FB仍为drop_last3086/3087。workers4替代作者32，原RMSprop(centered)保持。ER_roc/RK_roc是排名变化率，不是ROC AUC。来源34文件逐hash核验通过。

WN CPU v2实际100804项检查0失败；额外epoch-step-score日志绑定通过。固定16批/160例（32 edit + 128 locality）/320 before-after记录，独立CUDA重推理rank、Top3、targetlogit全部精确匹配，最大logit误差0；原参数/输入hash及版本号保持。限制：**必须重放作者CPU共享存储恢复**。训练后检查点ex_model与ex_learner的embedding在646653项不同，最大差.01770751；构造时`.weight.data`共享存储，Lightning root load依次加载ex_model、ex_learner，后者覆盖模型embedding和tied decoder，故最终effective模型不等于原ex_model序列化prefix。分离storage而严格恢复两组prefix的v2独立重推理实际FAIL（320记录中至少一项不匹配、最大logit差.23060799），完整失败保留。v3只机械匹配原生恢复顺序，不修改任何训练/最终导出/样本/容差。支持该固定公开加载协议的数值复现，不支持无恢复副作用的基础模型保留。

首次CPU审计的epoch断言FAIL也保留：PL1.3.1在dump_checkpoint中为恢复用途给epoch/global_step加1，WN原checkpoint raw epoch19实际对应训练epoch18；原检查器误按现代/零基epoch关系解释。已保存实际框架源码/hash、diff、版本化registration及机械fixture，v2修正解释；不能删除原失败。独立代码侧审因额度未完成，主控接管复核，另补selected epoch/step/score绑定；不声称完整独立代码复核已通过。

WN轻量证据已归档 `results/KGE-LIT-004/kgeditor-e-wn18rr-s0`，状态PASS_NATIVE_NUMERICS_WITH_RESTORE_LIMITATIONS，论文数值NOT_MATCHED；**父实验仍running，FB尚未审计**。最新审计等待器为 `KGE-LIT-004-AUDIT-V3`，选择绑定为 `KGE-LIT-004-SELECTION-V3`；原v1/v2状态和失败保留。FB终端后自动跑CPU v2、同160例的alias-aware v3 replay及选择绑定，最后仍需主控核验后提升整体。没有新的训练、模型网格或后续实验。

经典LKGE修订对照已完成并提升 `results/KGE-LIT-006`：FACT seed55、TransE200、五快照、200epoch上限/patience3，实测MRR.225/H1.135/H3.258/H10.406，原生训练1215.35s、总进程1331.97s。698项独立源码/来源/指标/检查点检查0失败，120条CPU抽样重推理全部精确匹配。明确采用valid.txt选best，区别于原版test.txt选best；ID和过滤集也随valid加载而改变。单种子、历史候选/过滤；未独立复算验证集排名、完整优化器训练或pre-FWT初始化。审计器由主控在代理额度耗尽后完成，没有另一个完成的代码复核。不能把该结果与不同协议/维数/种子的原生结果直接排序为方法收益。

## 6. 对当前浅层KGE编辑方向的实际威胁

KGEditor是直接PLM-KGE编辑强基线；持续KGE方法是遗忘、增量效率和表示适应的重要参照，但不能与固定列DistMult/ComplEx的Top10 SafeSucc直接同表。Pons已说明竞争实体会损害排名，即使旧query/旧正例得分不动也不足以保留预测。AlphaEdit/AlphaEdit+已有保留子空间投影与冲突处理；浅层KGE适配须明确归属，不能伪称作者原生LLM复现。

KBS2026 DOI10.1016/j.knosys.2026.116976出版元数据/摘要已确认结构局部性和低秩编辑机制，但完整论文与作者代码仍未获得；403/429和获取限制保留，未获得不等于不存在。现有有限固定列QP/Rank fallback尚未证明共享head/tail实体安全、新颖性或投稿GO。后续方法比较必须固定同模型/数据/候选/过滤/tie/信息集，并把暴露集与独立留出区分。来源核验与scope见 `plan/threat-review.md`。

## 7. 文件与使用

`repos/`固定作者源；`data/`与`models/`远程原始资源；`runs/`完整日志、检查点和逐例导出；`results/`审计和已审结果/带限制证据；`experiments/`实时canonical；`configs/`冻结协议与job清单；`tables/`派生对照表；`plan/`协议和文献威胁边界。`scripts/status_snapshot.py`是只读状态摘要工具。不把旧PID或本文缓存时间当作实时状态。
