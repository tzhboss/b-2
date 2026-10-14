# 141：140同父固定保护控制与参考收益诊断

WN_CKGE0→1真实ComplEx100，直接复用140的90%旧TRAIN拟合父、zero初态、prefix10/Adam及ordinary终态，112的完整新TRAIN/学习tape、旧TRAIN内部划分固定，不重训父/前缀。新目录唯一运行，已生成模型不静默重跑。

三臂none/fixed_anchor/fixed_replay，后段110步。共享完整可用旧TRAIN（含父未拟合内部参考）的信息权限；Anchor依照固定118构造规则，从原父Top10资格旧TRAIN采样16384 cohort、每步32固定选择，不改变penalty或rho.1。Replay复用112事实选择/合法候选mask/分母，用140旧E/R重算query向量一次。Anchor累计相对同ordinary shadow位移投影到新TRAIN跨旧query空间正交补；Replay不投影。三臂都计算学习、两保护梯度、shadow、完整投影操作包络，未选者丢弃。none需与140普通终态参数相同，非独立重复。

所有三终态保存后才首次在该父状态下加载官方old/new VALID作评价，VALID不作为训练/选择输入，不读TEST；仍是已开发暴露数据。报告完整旧TRAIN（另分父拟合/内部参考）、完整旧VALID/原128半群组、新VALID3720 raw/filtered/MRR/Hits/三方向、Top10掉榜和漂移、阶段新掉榜/救回、成本与动作比例。旧参数冻结，Replay非原生全参数方法。

内部参考诊断：固定prefix压力/旧rank/负归一gold特征，事后比较各固定保护相对ordinary的额外救回、新损伤和RR变化；报告参考及VALID上的关联AP，标签是动作收益而非未来损伤。分数不拟合、不用于本轮选择，无保护必要性或泛化校准主张。原参考中只有1591/6696父Top10，不隐藏资格选择及其偏差。

GO下一动作设计仅当内部参考的额外救回有非退化信号且固定控制存在合理新旧取舍；NO-GO直接恢复饱和实体gate或把高AP当端到端贡献。141本身只是控制，不构成自适应方法；P3暂不启动。费用计入一次必要新父query/space构造、全操作包络和全部评价，140父/数据准备历史费另列。无硬超时/独立审计/SHA/归档，不自动push。
