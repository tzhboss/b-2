# 106在线普通影子实现验证与成本，固定两个seed×两个候选

保留094/095 fullabsolute和105 fullReplay_anchor的所有学习/保护/Adam/查询空间公式、普通prefix10和后220步tape，只把CPU逐步anchors读取改为一个独立GPU普通学习影子：每步按同tape更新普通影子，引用其当前新行作为累计差参考，不保存330176000字节逐步快照。普通影子计算仍是实际额外计算，不当免费或删除学习底座。

四实现执行完检查普通影子终态与原ordinary逐位相同、保护终态E/R与既有对应模型逐位相同；只有完全一致才复用102/105完整VALID指标，无重复评分，不称独立科学确认。若有差异，记录后按新模型追加必要评价，不放宽检查后冒用旧指标。

记录输入/空间/必要fullReplay cache加载、在线影子setup、包含双路径计算的训练墙时，理论消除逐步anchor tensor bytes。未在同场重跑旧实现及生成普通快照，不把历史异时5.9–6.1秒生成+2秒训练与本轮直接作可靠速度提升比。

输入仍复用所有teacher/cohort/negative tapes，fullReplay cache现成；不重随机/SVD/影子快照。实施参数等价基本核对是复用评价所必需，不独立CPU审计/SHA/归档。无TEST/硬超时；在线影子仅当前已见TRAIN梯度，无future label。
