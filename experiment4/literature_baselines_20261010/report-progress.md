# 远程文献基线复现：阶段报告

更新时间：2026-10-10T12:29:20.699632+00:00。主执行/下载/模型/产物目录：`/data/lc/tzh/experiment4/literature_baselines_20261010`。本轮仍在执行；只有以下已审计结果可以引用，其余以 canonical 现场状态为准。

## 已完成的同划分对照

| 方法 | MRR | Hits@1 | Hits@3 | Hits@10 | 原生训练时间 | 总进程时间 |
|---|---:|---:|---:|---:|---:|---:|
| FastKGE | 0.223 | 0.131 | 0.256 | 0.402 | 449.15s | 577.59s |
| STARK | 0.226 | 0.134 | 0.261 | 0.408 | 1315.27s | 1462.41s |

两者采用逐字节相同的 FB_CKGE 五快照、seed3407、维度200、完整200epoch上限及原生早停，原生各快照指标先round4、按查询数加权后round3。训练配置/早停不同，表内时间是作者训练循环计时，不是纯GPU算子耗时。

独立审计：FastKGE361检查、STARK504检查均0失败，全部19逐查询导出/5最佳检查点/完整日志核查，每方法15post-train文件各8例，共240/240抽样排名精确重推理匹配。抽样不能认证所有得分；4pre-FWT文件核查排名范围/查询/指标，未复放其初始RNG。源代码数学部分与作者版AST一致，观测导出启用/关闭不改变排名。主控另核对冻结注册清单、运行命令、配置、原始输入和源哈希。

FastKGE论文Table3的FB-CKGE为MRR.223/H1.131/H3.257/H10.405、训练405s（3090Ti、五次运行平均）。本次单次3090结果MRR/H1相同、H3低.001、H10低.003；支持该作者默认单种子复现接近论文表值，不能称已重现五种子均值。STARK本次MRR比FastKGE高.003，H10高.006，原生训练耗时约2.93倍。STARK公开代码读结构文件的距离列而非PageRank，且未获得完整论文；只称公开实现执行，不称所有论文公式/性能结论已复现。

Canonical：`experiments/KGE-LIT-002/experiment.yaml`已为completed_audited/PASS_WITH_LIMITATIONS，轻量证据在`results/KGE-LIT-002/`。早期检查器未恢复历史LoRA A/B缓冲而失败，失败日志原样保留；修复明确恢复并逐tensor比对所有保存缓冲和前一最佳检查点谱系，没有忽略失败来强行通过。

## IncDE 已完成的单划分审计（整批尚未提升）

| 划分 | MRR | Hits@1 | Hits@3 | Hits@10 | 独立重推理 |
|---|---:|---:|---:|---:|---:|
| RELATION | 0.198 | 0.110 | 0.234 | 0.367 | 960 |
| FACT | 0.219 | 0.129 | 0.252 | 0.395 | 960 |
| HYBRID | 0.225 | 0.132 | 0.264 | 0.401 | 960 |

保持原生历史快照候选/过滤和double-argsort；每划分单种子3407。其余划分在完成后再独立审计，KGE-LIT-001未整体提升。RELATION的固定候选扩张诊断已终端成功：124052查询原生排名全量精确重放，MRR从0.198342893变为0.198136328。该诊断尚待独立派生结果审计，不据单划分作总体结论。
## 本轮其余执行范围

- IncDE四个作者主要划分：ENTITY/RELATION/FACT/HYBRID，完整五快照训练，canonical KGE-LIT-001。
- SAGE V1.0默认FACT：完整预设维数180/210/250/270/290，canonical KGE-LIT-003。源码未发现论文全部动态蒸馏/尺度拟合机制，保持原版并标注。
- KGEditor两个EDIT数据集：E-FB15k237、E-WN18RR，作者HF固定权重/数据修订，完整20epochs编辑器训练及暴露评估，canonical KGE-LIT-004。下载34文件逐哈希通过；源/token映射通过。作者edit_test用于检查点选择；FBdrop_last仅3086/3087编辑，WN830/830。locality统计是重复batch曝光而非unique全L-Test。workers4替代32改变worker局部采样权重，不能称逐数字同协议。
- Pons动机候选扩张诊断：固定同IncDE最终检查点、五种候选/过滤/tie评估，不训练、不调参，canonical KGE-LIT-005；等待IncDE独立审计PASS再运行。此为独立实现诊断，不是Pons全部八数据集作者复现。

## 对当前编辑方法的含义

这些持续KGE结果不与现有固定列DistMult/ComplEx的SafeSucc放在同一排名表。IncDE/SAGE的FACT三元组集合相同但顺序/种子/维度/训练不同；FastKGE/STARK另一个完全相同的划分组。仅数据同源不足以合并协议。

KGEditor是直接PLM-KGE编辑前作；AlphaEdit/AlphaEdit+已经覆盖保留子空间投影与冲突处理，KBS2026结构感知编辑的出版元数据/摘要已确认，但完整方法/代码尚未取得。不能把本轮成功训练或这些scope差异写成新颖性、SOTA或投稿GO。细节见plan/threat-review.md、plan/kgeditor-native.md和plan/selection-and-comparison.md。
