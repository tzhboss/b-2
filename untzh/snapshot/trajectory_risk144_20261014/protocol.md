# 144：剩余训练段的Adam矩预测与经验不确定性，固定P1证伪

对象固定原全事实父FB17 DistMult200（1→2新实体子任务）/WN真实ComplEx100（0→1完整新TRAIN），复用121/122 zero初态/prefix10、112/087学习tape、124旧rank/gold/querynorm、136 TRAINcohort及完整旧VALID prefix特征。133历史VALID监督保护和140内部参考新父不混入本轮比较。无重训父/前缀/新训练终态，TEST不读。

创新资格未确认，本轮检验一个预测假设：prefix10上第11epoch全部原batch的学习梯度分布可以近似剩余训练段。梯度只来自TRAIN、旧E/R冻结、每个batch仅做prefix forward/backward、不提交到原模型。一轮FB22/WN11 batch梯度的均值与逐元素二阶矩估计Adam未来一阶/二阶矩；按原betas/lr/eps/step外推剩余220/110步。

固定经验情景：g_b均值误差按1/sqrt(H)缩放，构造g_mean+(g_b-g_mean)/sqrt(H)，二阶矩同凸组合E[g²]+(g_b²-E[g²])/sqrt(H)。共同batch情景保持所有新实体/候选间相关性。不额外随机生成向量或假设候选独立。使用E[m]/sqrt(E[v])近似Adam期望并假设梯度平稳，既非真实训练轨迹也非已校准贝叶斯后验；不能称安全概率。所有预测阵列先保存，两父全部预测固定后再加载真实普通/固定Anchor终态标签。

预测方法预先固定：当前完整rank/最大合法gap、136 sampled128压力/原rank边界、首batch真实Adam一步位移乘完整剩余H、平均Adam矩外推、经验情景掉榜比例/平均rank；不改H、方差权重、batch集合或阈值。119为一个真实提案、主要一epoch特征/学习器，本轮为全batch矩、完整剩余长度及共同不确定性；124是候选生成前独立高斯初态风险，二者不同。

全候选filtered rank只需冻结旧candidate rank+合法新candidate超越数；current重建必须与136一致。纯TRAIN参考及VALID gold参考诊断分开，VALID输入用于离线风险评价，不用于训练或动作选择，不宣称未知gold发现。完整扫描与采样压力成本不同，因此完整current-rank/gap也作公平上限对照，历史gold/rank成本非零。

标签分开：prefix安全查询的后段新Top10掉榜；prefix已掉榜查询的普通自行恢复；固定Anchor相对普通的额外救回/新害。AP与1/5/10%召回，事件0→AP=null；经验情景比例用Brier与恒零风险作检查，不因AP提高宣称校准。额外救回仅关联预测，近Top10边界固定分数作为可恢复性诊断，不当保护必要性。报告forecast rank误差、情景宽度、norm偏差、费用。

GO候选预测至少需在VALID新增/恢复目标上相对完整current压力或单提案有实质信号、费用合理，并区分波动是否额外贡献；否则NO-GO该平稳矩外推，不微调。即使GO也不是完整自适应增益，P2需另有动作/强固定/随机控制，P3不启动。模型/cache留远程，无硬超时/独立审计/SHA/归档/push。
