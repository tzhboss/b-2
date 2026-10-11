# 新增实体受控比较：首批原生适配及训练对照

本注册覆盖 LKGE、FastKGE、STARK 的作者原生模型/loss 在共同旧检查点上的增量适配，以及 TransE200/DistMult200 的随机扩展、仅新实体更新、全部参数微调、等量旧 TRAIN replay。它是完整比较的首批任务；IncDE、SAGE、KGEditor、Rank/QP 尚需各自适配与注册，不能将本批完成视为整个比较结束。

数据固定为已经审计的 FB_CKGE 0→1，3753 真新实体，11258 完整候选；三seed 17/29/43。每参数空间使用完全相同有效扩展初始化。原生桥保留作者模型、loss、层数、rank 和指定默认/运行参数；共同训练调度器替代原多快照全量数据加载器，只向模型提供两份 TRAIN 图。此适配不是作者原协议指标复现。负采样统一保留原生 50% head/tail 腐化、无过滤、上界排除最后实体 ID 的分布；使用共同 Torch 随机流。新旧底座已经审计冻结，不按本批结果更换。

FastKGE排序特征按作者TRAIN图度中心性及逐行传播距离重建。STARK当前源的排序reader消费第二列，即distance，第三列PageRank不影响当前执行；本桥重新生成相同已消费distance，不修改reader。相应源文件及特征消费语义须出现在报告，不能据此声称论文意图的PageRank机制已正确复现。LoRA实体padding与作者gather置换保留；用逆置换放置共同初始化，防止ID/初始化不一致。LKGE保留importance regularization/MAE，两者权重来自配置，embedding transfer关闭；训练图不含VALID/TEST。

一个固定配置/方法，不做网格。每任务上限200 epoch，batch1024/10 negatives，2400秒训练预算，一GPU两CPU线程。模型选取只依据seed124选出的旧VALID1024及新VALID1024，在完整11258候选上的raw MRR均值，各占1/2；epoch1及每5epoch验证，首次最高选中。随机扩展无需训练，仅保留初态。Replay每epoch从旧TRAIN无放回取与增量directed TRAIN等量，FullFinetune使用全部已见TRAIN；报告各自数据访问成本，不能声称训练成本相同。全部参数微调仅表示所有参数可训练，其TRAIN包括新旧全部事实。

检查点固定后才读取两个TEST split，均为原文件行序的双向查询；不使用过滤排名，rank=1+count(scores>gold)。旧主Top10损失以旧7505候选成功集合为分母、after11258候选；同扩展池辅助指标独立注明。报告全旧/新MRR、Hits1/10、rank displacement、旧参数位移、模型/训练参数数、实际用时/显存。原生预测与有效导出全局评分需在预检/每次选模验证；任何失败保留，不原地覆盖。

独立审计用NumPy重算64个完整11258候选行，精确排名对照及整表汇总/查询ID/输入哈希核验；只认证这部分推理重放和汇总，不声称独立重新训练全部任务或所有分数逐项重算。仅通过审计的轻量结果可进入results。本数据已暴露，是开发比较，不是新最终留出。
