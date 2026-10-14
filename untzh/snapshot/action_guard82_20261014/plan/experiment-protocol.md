# 082：实际Adam拟议干预的TRAIN代价控制

新的开发假设，不冒充已有有效方法。seed17/DistMult200/FB-CKGE0→1，复用058 cohort/prefix10、052学习tape、077旧teacher logZ。32旧答案×128合法新候选、初态8batch RMS .1及每步cap不变。两种保护方向：absolute和50/50 blend。

每步先拟议普通Adam更新（包括当前动量），再依次检查保护梯度乘数1/.5/.25/.125。三个新方向各固定抽256个TRAIN三元组，负样本复用该三元组在第一epoch的学习tape，所有臂共同使用。若各方向探针损失不超过普通拟议更新的1.001倍，且同一32旧答案的标准化softplus竞争压力小于普通拟议更新，接受第一个最大强度；否则执行普通梯度。探针和候选评估仅TRAIN、不读未来/VALID标签。保护实施仍用选择的梯度调用原Adam，不改变历史状态。

实际Adam提案由独立临时参数和相同optimizer状态计算，不更新真实optimizer；不是Euclidean梯度内积保证。一步TRAIN探针不损伤不证明终态MRR安全或未知知识安全。候选选择可能过拟合固定TRAIN探针，需要实际VALID与完整旧TRAIN评价。

固定两个决策配置absolute_guard/blend_guard；各追加一个乘数恒为该配置实际平均乘数的控制，使用共同tape/初态/抽样/校准。后者是诊断控制，系数来自已运行TRAIN决策日志、不是VALID选择。历史none/uniform/blend的科学终态只读复用；先核对共享校准与来源。所有四臂训练完成后才读VALID，并对四臂全部评价完整旧TRAIN。

记录每强度接受次数、各方向实际探针代价、旧竞争proxy收益、拟议次数和耗时、实际梯度比例及完整终态raw/filtered质量。若决策不能比同平均强度控制更好，不称风险感知收益。新产物/data/lc/untzh/action_guard82_20261014，无固定超时/审计/SHA/归档。
