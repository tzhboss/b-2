# 089 第二增量风险识别—保护联动

固定seed29、088父状态/初态、学习tape、10epoch prefix含Adam、16384旧TRAIN资格答案及128合法新候选、teacher/尺度/特征统计，全部只读复用。20epoch总训练，新增四臂只做后220步。冻结旧11258E及474R，仅更新1876新行。任务仍是snapshot2新实体子任务，新增旧—旧事实仅truth过滤。

四新臂transfer_tree/pressure128/pressure32/random_pool，每epoch开始按自身当前模型重算风险并取top5%池，每步26个池内＋6个全池补充且不重复答案，保护仍32×128 absolute softplus。随机池按固定epoch种子产生。tree为058原模型直接迁移，不拟合本轮学习器、不用未来标签。pressure32仅用32候选评分选池，但保护仍128候选。

所有臂使用088同一初态8batch RMS absolute系数、逐步梯度范数上限.1，同host/Adam及初始化。系数不重新按风险臂选优；记录实际梯度强度，范数上限不是同Adam位移或安全保证。复用088 none/absolute/absolute075模型指标，复用控制非独立重复。

所有四臂完成才评价088固定1024oldVALID/1024newVALID及完整oldTRAIN，同TRAIN-only filtering和固定214095/416Top10资格，raw/filtered均保留。主比较对均匀absolute，而非仅none；同时检查三个新方向、旧MRR与旧Top10掉榜。以匹配随机池排除集中采样本身效果；弱保护absolute075排除单纯强度解释。不细调top比例/阈值，不读TEST。

本轮是在已暴露开发结果基础上的机制探索。目标是检验跨阶段识别的动作价值，不复活058–060旧结果/不宣称识别即保护有效。特征只依赖当前模型/TRAIN，完整未来标签不进入路由。准备复用费用与历史生成费用分别记录，新检测/训练/评价真实计时。无硬超时、独立CPU审计、SHA或归档。
