# DASFAA 2027｜第24轮：30966 远程 experiment4 强基线审计

日期：2026-10-09。**已执行完成**；目前只是实验基线和负向结果，**没有复现/战胜完整SOTA系统，也不能确认可发表研究空白**。

## 1. 实验执行与数据协议

远程 root@192.168.3.1:30966 的 /data/lc/tzh/experiment4。使用7张3090中的部分GPU训练，NumPy CPU评估。上传 FB15k-237 官方 TRAIN/VALID/字典和 phase22 锁定的三种子各27个EVAL查询，**未上传、读取或评价官方 TEST**。TRAIN guard保护事实与独立TRAIN heldout事实不重叠；VALID只指定每查询三个编辑目标。模型共享实体向量同时用于head和tail，不采用冻结head简化评价。

真实独立训练：DistMult、ComplEx，各64维，各401/402/403共3个训练种子，每个25epochs，Adam lr 0.007，batch4096，4个均匀负尾/正样本。保存5/10/15/20/25的完整模型checkpoint和哈希；六个模型均成功完成；TEST未用于训练或调参。每模型27例编辑，共162个模型-案例评估事件；同种子在两模型族使用一致query ID，跨训练种子查询可能重叠，绝不视为162个独立数据集。

作者数值对照：从 Rank-Level Locality 作者发布的 MIT 代码原样引入 ProtectionSubspace 与 rank_truncated_delta；但用于我们三目标共享参数协议，**不是作者完整1000例权重/系统/cohort复现**。其他本地方法是Direct、针对有关候选tail的Soft(LocalGuard)、对每个编辑实体采用全部TRAIN guard的Soft(AllGuard)，及两种guard代理选参 Adaptive。所有方法执行相同旧答案位保护与三个VALID新目标进入Top10的任务。成本 C(ΔE)=逐实体欧氏范数之和L2,1，预算B=3.5/5.2，超限/编辑失败均拒绝整次更新，拒绝计入成功率分母且损伤0。

## 2. 模型拟合检查

固定512条VALID样本（仅用TRAIN+VALID做filtered）的MRR，3种子均值：

| 模型 | 第5轮 | 第25轮 |
|---|---:|---:|
| DistMult 64d | 0.2654 | 0.2906 |
| ComplEx 64d | 0.2949 | 0.3064 |

这不是完整官方TEST MRR，也不是领域SOTA权重。VALID兼作编辑目标来源，不是完全独立于本轮研究设计的最终验证集。

## 3. 三目标编辑、跨关系保护：第25轮权重，B=5.2

下表跨关系损伤是独立TRAIN-heldout编辑前正确Top10事实，之后因编辑掉出Top10的事件数，按**81次编辑尝试**均值计算。

| 方法 | DistMult成功 | DistMult跨关系损伤/次 | DistMult实际L2,1/次 | ComplEx成功 | ComplEx跨关系损伤/次 | ComplEx实际L2,1/次 |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 80/81 | 0.432 | 1.739 | 78/81 | 0.148 | 1.919 |
| Soft3 LocalGuard | 80/81 | **0.321** | 1.876 | 78/81 | **0.123** | 2.042 |
| AdaptiveSoft LocalGuard | 80/81 | 0.395 | 1.933 | 78/81 | 0.160 | 2.090 |
| Soft3 AllGuard | 71/81 | 0.654 | 2.234 | 71/81 | 0.420 | 2.295 |
| RankTruncated 原作者内核 frac=0.125 | 68/81 | 0.827 | 2.281 | 68/81 | 0.444 | 2.301 |
| RankTruncated 原作者内核 frac=1/64 | 73/81 | 1.309 | 2.088 | 73/81 | 0.593 | 2.331 |
| AdaptiveSoft AllGuard | 80/81 | 0.765 | 3.014 | 78/81 | 0.556 | 2.907 |
| AdaptiveRank 作者内核＋本地调参 | 73/81 | 0.877 | 2.702 | 73/81 | 0.519 | 2.817 |

DistMult原始可信TRAIN跨关系留出事件暴露1757条/次组，ComplEx为1779。模型基线预测不同导致被纳入保护的事实集合不完全一致，**不同模型之间不能只比较绝对掉榜量就认定ComplEx更擅长防遗忘**。

所有方法的逐例预算约束、完整三目标编辑成功记录、模型和查询ID一致性均通过自动核对。v1/v2 Direct在全部相同案例上逐例完全一致。

## 4. 配对统计以及重要负结果

按各模型的78个不同(h,r)查询聚类，对81次编辑计算探索性Bootstrap 95%区间，B=5.2：

- DistMult LocalGuard Soft3相对Direct跨关系差 -0.1111/次，区间 [-0.2564,+0.0128]，包括零（10次改善、5次恶化、66次相同）。
- ComplEx LocalGuard Soft3相对Direct差 -0.0247，区间 [-0.0897,+0.0385]，包括零（4次改善、2次恶化、75次相同）。
- DistMult AllGuard Soft3相对Direct差 +0.2222，区间 [+0.0128,+0.4167]，且三目标成功从80降至71。
- ComplEx AllGuard Soft3相对Direct差 +0.2716，区间 [+0.0641,+0.4808]，成功从78降至71。

**关键发现**：在这个共享参数、多目标、TRAIN保护集合的任务下，将所有guard方向无差别施加给所有被编辑实体，可能比仅保护与该尾实体相关的旧知识更差。作者Rank内核的这一局部多目标改造版本未超过Direct或LocalGuard。**这并不意味着原作者完整SOTA系统表现差，也未证明LocalGuard显著优于Direct**。AllGuard相比Direct的损伤更高，尽管失败更多已使损伤均值受到向下偏置，是值得研究的负向信号。

同训练epoch下固定Soft3 LocalGuard往往比按TRAIN guard分数漂移自适应选λ更好，提示代理目标最小化与真实heldout Top10排名保留之间存在不一致。

## 5. 训练epoch消融（v1 LocalGuard同协议，B=5.2）

| 模型/epochs | Direct成功 | Direct损伤/次 | Soft3成功 | Soft3损伤/次 | Adaptive成功 |
|---|---:|---:|---:|---:|---:|
| DistMult 5 | 78/81 | 1.025 | 76/81 | 0.716 | 78/81 |
| DistMult 25 | 80/81 | 0.432 | 80/81 | 0.321 | 80/81 |
| ComplEx 5 | 80/81 | 0.593 | 80/81 | 0.444 | 80/81 |
| ComplEx 25 | 78/81 | 0.148 | 78/81 | 0.123 | 78/81 |

虽然查询ID相同，随着训练轮次增加，原始模型Top10正确heldout事实集合也变化；DistMult检查次数为1718→1757，ComplEx为1747→1779。因此下降不能完全解释为模型越强越少遗忘；应将原先正确的**同一事实交集**固定后再回测。

## 6. 可发表性判断和下一步

研究问题值得继续审计：多目标共享KGE编辑时，**保护事实的实体相关性、约束的局部性及真实排名损伤**，可能比统一全局正则强度更关键。但还不是已证实的原创research gap，也没有统计稳健地战胜Direct。

后续必要：作者完整预训练模型及cohort或KGEditor正式强基线、同实际成本matched预算（不是仅同硬上限）、保护标签数量/事实完全匹配、TEST盲测以及第二知识图谱/ComplEx更强训练。WN18RR按当前三新目标＋两旧目标仅有13条语法可行查询，不能假装已经完成跨KG泛化。

## 7. 文件和复现

远程路径：/data/lc/tzh/experiment4

- data：FB15k-237 TRAIN/VALID/字典、phase22锁定案例，未包含TEST。
- code/train_kge_baselines.py：真实训练。
- code/eval_paired_kge_edit.py：LocalGuard和5/25epoch对照。
- code/eval_author_rank_kernel_v2.py：AllGuard＋作者RankTruncated数值内核。
- code/vendor/kge_rank_locality + LICENSE：作者MIT源码。
- code/audit_remote_phase24.py 和 audit_rank_kernel_v2.py：实际执行的对齐和预算审计。
- checkpoints：DistMult/ComplEx 3种子 × 5阶段的本机checkpoint，SHA256在训练记录和逐编辑JSON中。
- logs：逐训练、逐编辑命令日志。
- results：全部逐例输出以及两个epoch审计和Rank核审计JSON/CSV。

重跑示例（单种子）：

~~~bash
cd /data/lc/tzh/experiment4
CUDA_VISIBLE_DEVICES=1 /data/conda/envs/test/bin/python code/train_kge_baselines.py --dataset data --out checkpoints --kind ComplEx --seed 401 --dim 64 --epochs 25
OPENBLAS_NUM_THREADS=1 /data/conda/envs/test/bin/python code/eval_paired_kge_edit.py --kind ComplEx --seed 401 --epoch 25
OPENBLAS_NUM_THREADS=1 /data/conda/envs/test/bin/python code/eval_author_rank_kernel_v2.py --kind ComplEx --seed 401 --epoch 25
/data/conda/envs/test/bin/python code/audit_remote_phase24.py --epoch 25
/data/conda/envs/test/bin/python code/audit_rank_kernel_v2.py
~~~

**最终：实验已完成且具有复现材料，但尚不足以宣称新方法超过SOTA。**
