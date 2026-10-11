# 远程文献基线复现报告（本轮完成，按用户要求先停）

完成时间：2026-10-10T14:25:26.131731+00:00。本轮选定的10路完整训练及审计均已终端；结果、失败记录和比较限制已归档，当前注册训练/审计无活动进程，不启动下一轮。数据、模型、训练、完整检查点及主要产物位于 `/data/lc/tzh/experiment4/literature_baselines_20261010`，文献来源 `/data/lc/tzh/kge_editing_literature_20261010`；本地仅保留报告与轻量证据副本。

IncDE四划分和FastKGE接近论文表值；STARK为公开实现结果，LKGE为验证集选点修订对照。SAGE原完整性FAIL保留。KGEditor两路均完整20epochs，固定公开实现数值得到带恢复限制的重推理支持；WN18RR成功率明显低于论文，两路逐项差异见表，不认证为论文性能复现。本轮完成的是选定文献的实验与审计收尾，尚未得到可声称优于原论文的新方法或投稿GO。

资源使用以实际进程收据为证：2026-10-10T11:49:23Z有7个已注册训练进程重叠，GPU0至6各一路（IncDE四路、FastKGE、STARK、SAGE）。这证明并行执行；不声称所有显卡每时刻均100%利用率。末期只剩原定KGEditor长任务，未为填满显卡而新增实验。证据见 `tables/parallel-execution-evidence.json`。

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

## 5. KGEditor两路完成；LKGE修订对照完成

KGEditor固定公开作者源码、HF模型和数据修订，seed0、完整20epochs。两路exit0，登记科学输入未变。论文指标仅作描述性对照；原论文训练预算的具体参数来自固定作者shell，而非声称正文完整声明了所有预算。

| 数据集 | Succ@1实测/论文 | Succ@3实测/论文 | ER_roc实测/论文 | RK@3实测/论文 | RK_roc实测/论文 | optimizer steps | 总进程时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FB15k237 | 0.779326/0.866 | 0.958198/0.986 | 0.998379/0.999 | 0.975592/0.874 | 0.799254/0.635 | 30860 | 7664.13s |
| WN18RR | 0.307229/0.833 | 0.414458/0.844 | 0.868610/0.991 | 0.967693/0.956 | 0.182034/0.256 | 14900 | 4065.36s |

两路与论文表值的描述性差异如上，WN18RR成功率明显低于论文。不能用较弱实测结果宣布强baseline已被击败，也不后验改变模型、训练预算或参数追分。论文Time .226/.232为编辑时间列，不能与这里完整训练进程耗时直接算加速。

这是native exposed evaluation：edit_test同时用于选择best和最终评估，不是独立heldout。workers4替代作者32，原RMSprop(centered)保持；ER_roc/RK_roc是排名变化率，不是ROC AUC。来源34个文件的原blob/LFS哈希核验通过。
FB15k237实际评估3086条edit、12344次locality暴露，只有3088个unique stable行；不能将重复暴露等同完整unique L-Test。
WN18RR实际评估830条edit、3320次locality暴露，只有832个unique stable行；不能将重复暴露等同完整unique L-Test。

CPU v2两路共431751项检查0失败；两路selected epoch/step/monitor score绑定通过。每路固定16批、160例（32 edit+128 locality）、320条before/after记录，两路合计320例/640记录，独立CUDA重推理rank和Top3全部精确匹配。
FB15k237最大target-logit误差0，序列化两组embedding最大差0.283800036；参数hash/version及输入未变。
WN18RR最大target-logit误差0，序列化两组embedding最大差0.0177075099；参数hash/version及输入未变。

**恢复限制必须保留：**作者CPU构造时`.weight.data`共享存储，Lightning root load先加载ex_model、再加载ex_learner，后者覆盖基础embedding和tied decoder；最终effective模型不等于原ex_model序列化prefix。v3仅按源码复放该原生恢复顺序，支持此加载协议的固定数值，不能证明无恢复副作用的基础模型保留。WN两embedding有646653项不同，最大差.01770751；严格分离storage、各自恢复prefix的v2重推理实际FAIL，最大logit差.23060799，失败完整保留。没有修改训练、原最终导出、采样或容差。

首次WN CPU审计的epoch断言FAIL保留：PL1.3.1 dump_checkpoint为恢复用途将epoch/global_step加1；raw epoch19实际对应训练epoch18。v2按实际框架源码修正解释，源码/hash/diff/fixture/版本化登记已归档。独立代码侧审因额度未完成，主控接管复核并补选择绑定，不声称完整独立代码复核通过，也未重建整个优化器轨迹或独立全量验证前向。

`results/KGE-LIT-004`父canonical为completed_audited_qualified，审计PASS_NATIVE_NUMERICS_WITH_RESTORE_LIMITATIONS。两路CPU/重推理/选择绑定与修正依据已归档。先前WN单路summary中的parent_complete:false及当时FB仍运行说明是冻结历史快照；当前整体完成状态由父summary及canonical给出，不改写历史。原v1/v2失败和superseded状态保留。

经典LKGE修订对照已完成并提升 `results/KGE-LIT-006`：FACT seed55、TransE200、五快照、200epoch上限/patience3，实测MRR.225/H1.135/H3.258/H10.406，原生训练1215.35s、总进程1331.97s。698项独立源码/来源/指标/检查点检查0失败，120条CPU抽样重推理全部精确匹配。明确采用valid.txt选best，区别于原版test.txt选best；ID和过滤集也随valid加载而改变。单种子、历史候选/过滤；未独立复算验证集排名、完整优化器训练或pre-FWT初始化。审计器由主控在代理额度耗尽后完成，没有另一个完成的代码复核。不能把该结果与不同协议/维数/种子的原生结果直接排序为方法收益。


## 6. 对当前浅层KGE编辑方向的实际威胁

KGEditor是直接PLM-KGE编辑强基线；持续KGE方法是遗忘、增量效率和表示适应的重要参照，但不能与固定列DistMult/ComplEx的Top10 SafeSucc直接同表。Pons已说明竞争实体会损害排名，即使旧query/旧正例得分不动也不足以保留预测。AlphaEdit/AlphaEdit+已有保留子空间投影与冲突处理；浅层KGE适配须明确归属，不能伪称作者原生LLM复现。

KBS2026 DOI10.1016/j.knosys.2026.116976出版元数据/摘要已确认结构局部性和低秩编辑机制，但完整论文与作者代码仍未获得；403/429和获取限制保留，未获得不等于不存在。现有有限固定列QP/Rank fallback尚未证明共享head/tail实体安全、新颖性或投稿GO。后续方法比较必须固定同模型/数据/候选/过滤/tie/信息集，并把暴露集与独立留出区分。来源核验与scope见 `plan/threat-review.md`。

## 7. 文件与使用

`repos/`固定作者源；`data/`与`models/`远程原始资源；`runs/`完整日志、检查点和逐例导出；`results/`审计和已审结果/带限制证据；`experiments/`实时canonical；`configs/`冻结协议与job清单；`tables/`派生对照表；`plan/`协议和文献威胁边界。`scripts/status_snapshot.py`是只读状态摘要工具。不把旧PID或本文缓存时间当作实时状态。

## 8. 本轮交付和停止点

`tables/continual_kge_real_results.json/.csv`有8行真实持续KGE结果；`tables/kgeditor_exposed_real_results.json/.csv`有2行暴露编辑结果。`tables/literature-baseline-evidence-status.json`为完成时canonical/hash快照；`plan/review/completion-requirements-audit-final.json`核查10路训练收据、终端状态、归档字节及无活动进程。GPU峰值未测，表中为null，不填0。候选集合诊断另列第4节，既不当新方法，也不混入持续KGE或PLM编辑排名。

本轮到此停止。未启动下一轮、额外种子、模型网格或新数据实验；未来实验、下载和模型继续优先在远程 `/data/lc/tzh/experiment4`。当前结果支持可靠基线准备和协议风险判断，不自动满足“做出可相比原论文或优化方法”的更长期研究目标。
