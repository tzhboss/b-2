# 119 低尺度FB与实际ComplEx的下一段风险迁移

无训练、无拟合新识别器。固定三个FB父状态17/29/43低尺度初始化与ComplEx17标准/低尺度两初值；旧TRAIN固定cohort和原Top10旧VALID参考群体分别评价，不按结果换窗口。预测epoch10→20新掉榜，复用117/118各阶段rank、initial/prefix、teacher/gold、TRAINtruths。VALID参考对象的旧答案、此前rank已知，属于既有事实预警，不是未知答案发现；VALID不用于训练或选择规则。

058固定current-tree与new-event-tree，历史训练目标为下一epoch，本轮直接迁移到后十epoch，未经校准。和边界、32/128候选压力、Adam线性前瞻、固定随机比较。current树仅原15个当前特征+epoch_fraction .5；new-event树另需5个下一步特征，使用prefix optimizer复制态真实Adam提案，每父/初值仅一次缓存，按本模型每epoch批数作线性外推而非未来轨迹。查询向量按DistMult/ComplEx实际公式。128新候选/64旧候选尺度一次抽样缓存；同父标准/低初值复用同样候选，过滤已见TRAIN正例。特征中不做全候选评分。

safe AP仅epoch10仍Top10群体；5/10/20%预算同时报告全参考池与只选择当前安全对象两种政策，后者需要此前完整rank，费用沿用117/118不能视为零或完全轻量部署。既存当前完整rank分数仅额外诊断控制，同样有全评分成本。所有预测保存后才读终态标签；历史已暴露VALID，是开发迁移证据而非未见确认。无事件则AP/recall null；查询共享实体，不当独立事件。报告特征/抽样/预测成本，历史资产费用非零。无TEST/硬超时/独审/SHA/归档。
