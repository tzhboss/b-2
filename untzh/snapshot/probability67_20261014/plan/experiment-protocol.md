# 067 概率质量保护与学习损失

沿用seed17/DistMult200/FB-CKGE0→1/旧E,R冻结/prefix10/后370步同正例tape和Adam初态。12配置：原学习host+mass_old/mass_gold/mass_guard/mass_sample；contrastive及fullsoftmax host各none/absolute/semantic/mass_old。

contrastive把同10负例/正例改为11分数CE；fullsoftmax为完整11258候选CE，屏蔽TRAIN其他真答案、保留当前gold；L2同原学习。prefix前10epoch仍为原学习，故是阶段切换，不称从初态完整CE原生实验。对比改变host的matched none控制及历史ordinary/uniform/replay。

mass_old=softplus(logsumexp合法新候选−旧池logZ)，旧池logZ按每答案屏蔽其他TRAIN真答案、保留当前gold；mass_gold参考gold分数；mass_guard另加.1 gold参考项；mass_sample用固定128合法新候选估计new exp总量，估计log并非无偏。R/oldE固定使旧logZ可缓存，所有参照只TRAIN。semantic为064关系共现候选权重.1。

32保护答案，初态8batch依对应host RMS校准rho .1，逐步cap .1；保护full/sample候选费用分别记录。全模型训练完后VALID；筛选及非支配TRAIN评价沿063。没有TEST、独立确认、创新主张、硬超时、CPU审计、SHA/归档。新资产/data/lc/untzh/probability67_20261014；单seed暴露开发集，多重尝试须后续确认。
