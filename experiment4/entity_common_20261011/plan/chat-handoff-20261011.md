# KGE实验新聊天交接提示词

请接手当前知识图谱编辑（KGE）项目，先读取项目AGENTS.md，并现场核查远程状态。不要重启金融或语音任务。以下是2026-10-11交接快照，不是未来实时状态。

## 用户当前要求与优先级

1. 收尾当前新增实体第041轮：继续观察原任务，完成检索评价、独立审计、三种子门槛复算、结果表更新和归档。不要因观察超时而重新启动。
2. 用户最新明确授权：“按这个做一个统一的测试一下”，指把LKGE、IncDE、FastKGE、STARK、SAGE、KGEditor、Rank系列和我们的QP＋Rank放入统一单次编辑测试。当前仅完成可比性和接口调查，尚未编写或注册这项新测试，也未启动训练。需要接着实施，不能将累计新增实体适配的坏结果当作单次编辑比较。
3. 长期目标是实际改善新知识学习与旧排名保护的取舍，未完成。原思路文件必须读取：C:/Users/11500/.codex/attachments/d93a5eba-6d3b-4577-bce5-d763b274ab61/pasted-text-1.txt。已知机制组合不能直接声称创新或SOTA。

## 远程环境

- 所有后续实验、下载、模型和主产物优先放/data/lc/tzh/experiment4。
- 连接：ssh -p 30966 -o BatchMode=yes -o ConnectTimeout=35 remote30340。复用旧别名覆盖端口，不修改全局SSH配置。
- 当前新增实体项目：/data/lc/tzh/experiment4/entity_common_20261011。
- 通用Python：/data/lc/tzh/experiment4/literature_baselines_20261010/environment/downloads/bin/python。KGEditor另用environment/kgeditor38/bin/python。
- 7张RTX3090、32CPU、约200GiB内存；只用现场空闲资源，不能结束无关进程。设置OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2。
- HAMI的GPU总览把多个设备内存堆在GPU0，不能据此把GPU1–5当空闲。用nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader及UUID核查。驱动PID与容器PID映射尚未认证。
- 容器/proc/uptime与进程start ticks观察到不同时间原点，不能直接相减估算时长；训练预算自身使用time.monotonic()。
- 远程没有rg，可用Python/grep；本地有rg。

## 当前确实运行的第041轮

- ID：KGE-ENTITY-COMMON-FEASIBLE-TRAIN-041。
- canonical：experiments/KGE-ENTITY-COMMON-FEASIBLE-TRAIN-041/experiment.yaml。
- config：configs/feasible-train41.json；协议plan/feasible-train41-protocol.md。
- 从已审计Full/Scoped冻结QP终态开始，冻结旧实体和关系参数，保留原保护集合/半径/间隔；每个新增实体直接优化其关联新TRAIN损失，完整候选池选择hard negatives。只提交实际排名保护通过且局部损失不增的更新，否则原样回退。局部TRAIN改善不能当检索收益。
- Full/Scoped×seed17/29/43共6任务，各3753实体一个扫遍，预算10800秒/训练任务，不自动重试或延长。自动流程run→evaluate→audit→六任务汇总。
- 交接时训练总进度55.9%，尚无任务完成或进入评价。
- coordinator PID801733，start ticks104568393。
- full17 PID801736/GPU0：2045/3753；scoped17 PID801737/GPU1：2015。
- full29 PID801738/GPU2：2136；scoped29 PID801739/GPU3：2255。
- full43 PID801740/GPU4：2058；scoped43 PID801741/GPU5：2075。
- 六个进程及coordinator身份刚核验一致。上述PID/进度必须重新查，不能当永久状态。
- 只读监控：在远程项目目录执行上述通用Python scripts/inspect_feasible_train41_progress.py。已修复阶段切换后训练进度归零问题。
- 原始产物artifacts/incremental/<job>；日志logs/feasible-train41；最终results/KGE-ENTITY-COMMON-FEASIBLE-TRAIN-041。
- 六任务全部审计成功后，要求三seed总体新VALID MRR≥源模型98%；旧Top10损失均值≤源模型75%且每seed不恶化；旧头/新答案方向MRR≥98%、Hits10≥95%。只为固定开发集门槛，非留出认证。
- 审计独立重建3753步/全部训练正例和hard negatives/7506标量损失，以及完整终态保护集合和固定VALID。共同TEST仅64完整评分行独立CPU核查加全部查询身份/汇总，不称全TEST CPU重推理。
- 不改正在运行的科学源码、配置、选择或门槛。若工程/审计失败，保留原失败和权重，另行版本化修复。
- 统一表tables/common-comparison-v5已有43方法行、129任务槽：117通过、6历史失败、6本轮运行。完整资格后运行scripts/assemble_feasible_common_table.py --root . --output results/<新的receipt文件名>.json，核验旧41行及v4 SHA不变。
- 最新远程Git HEAD faf6aba，前两提交a975c15/52c9571。只提交轻量证据，不提交权重/NPZ/日志/环境/压缩包，不push。

## 已完成的文献与共同任务

- 作者协议复现目录/data/lc/tzh/experiment4/literature_baselines_20261010；总报告report.md。IncDE四划分及FastKGE接近论文；STARK仅公开实现，LKGE验证集选模修订；SAGE原producer元数据FAIL保留。KGEditor两数据集训练完成，WN18RR成功率明显低于论文，并有原生恢复共享存储限制，不能称完全性能复现。
- 文献清单/data/lc/tzh/kge_editing_literature_20261010。
- 统一新增实体任务FB_CKGE快照0→1，7505旧+3753新=11258实体，74428旧/12406新TEST方向查询，无过滤strict rank=1+count(score>gold)。seed17/29/43，已暴露开发数据。
- LKGE/IncDE/FastKGE/STARK/SAGE/KGEditor均完成三seed共同任务评价及相应审计。旧Top10掉榜分别8.43%/9.53%/13.65%/14.84%/14.14%；KGEditor累计适配崩溃不能外推作者单次编辑。
- NewEntityOnly DistMult新MRR .14117、旧掉榜2.59145%；预测保护QP新MRR .13981、旧掉榜1.61312%，但旧头/新答案MRR保留约89.93%、Hits10约80.23%，尚无全面方法胜利。
- 013 TRUE-only、019/020任务加权投影、027/030/032共享联合更新、037/038容量豁免均保留负结论，不原样再跑。039 TRAIN-only冲突诊断已完成审计：大量阻碍来自冻结旧候选容量，不是只换新增行更新顺序就能解除。040/040R2只是预检，不替代041结果。
- AIR和AlphaEdit没有完整作者任务复现；不能把所有文献都写成已复现。

## 新授权的统一单次编辑测试（待实施）

每条请求从同一对应底座恢复，学习一条新事实，评价后恢复原模型；不同请求不累积。统一事实ID、候选池、查询集合、strict排名和指标。明确是已有实体加事实还是新增实体加事实；本次讨论参照Rank单次已有实体事实晋级，不能悄悄改回批量新增实体。

保留各方法核心机制，使用相同数量的VALID调参机会与预设选择规则；记录底座/meta训练及在线时间、旧数据访问、可改参数、真实共享head/tail影响。模型架构不同可同表标明，参数兼容者另做同底座对照；不通过阉割蒸馏/LoRA/维数机制获得整齐表格。不支持项、退化或失败必须显式披露，不能随意填0或宣布赢。

指标至少：目标Top10成功率、旧Top10答案掉榜比例、发生任一损伤的编辑比例、成功且零损伤SafeSucc、损伤数均值/P95及时间。两种掉榜分母绝不能混淆。优化只能读取编辑事实与旧TRAIN；测试只评价。先核查各核心机制在单事实输入下的有效性，预检通过后冻结完整协议并运行。

已可复用代码：src/common_incremental.py（LKGE/FastKGE/STARK桥），src/native_extended.py（IncDE/SAGE），src/lm_editor_common_v2.py与src/lm_editor_bridge_v2.py（KGEditor原生meta和在线delta）。已有artifacts/base的transe/distmult/sage-base三个seed，artifacts/lm-editor-meta中KGEditor meta三个seed；旧源码import模块名冲突需进程隔离。模型接口和旧参数范围不能假设都支持现成单次编辑。

过去同400请求Rank与QP对照在/data/lc/tzh/experiment4/rank_comparison_20261010，本地kge-rank-unified-20261010/report.md。Rank目标成功96%，SafeSucc48.25%/49%；QP＋Rank目标成功96%，SafeSucc57.75%/59.75%，依次DistMult/ComplEx。这是已暴露VALID、冻结head候选尾列编辑，不能当完整共享模型全局保证，也没有全部文献基线。

## 本地与交付

本地根C:/Users/11500/Documents/ChatGPT/b会文件夹。
新增实体副本kge-entity-common-20261011；文献副本kge-literature-baselines-20261010；单次Rank旧报告kge-rank-unified-20261010；更早共享模型证据rank-locality-followup30（这里旧KGE-RANK-041与当前KGE-ENTITY-COMMON-FEASIBLE-TRAIN-041不同）。本地主要是报告和轻量证据，远程权重/NPZ/日志/canonical为权威。本地镜像不一定已含最新远程运行结果。

请先核查041实时状态并读取以上文件，然后落实新授权的统一单次编辑测试。不要只重复进度，不要把训练完成或审计通过当科学优化成功。
