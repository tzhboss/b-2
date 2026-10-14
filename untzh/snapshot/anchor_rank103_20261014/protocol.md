# 103累计差空间约束×排名信号，启动前固定六配置

两个开发seed29/17，相同FB_CKGE snapshot1→2新实体子任务、前10epoch普通prefix/后220步tape、旧E/R冻结、jointAdam、rho .1每个非零保护梯度。普通影子与查询空间分别读各自094/095/092缓存，不重SVD/影子/cohort/teacher TRAIN。

保护动作固定累计相对同一步普通影子的差投影到各新实体跨新旧TRAIN正例查询空间正交补。只换保护信号：090/091既有violation_full（压所有正gap）或budget_full（允许10−oldrank个新越位，保留最大gap、压剩余小正gap），公式/阈值不变，不当原创容量方法。

seed29固定violation/budget×uniform/tree四组。uniform同088固定selections；tree每epoch自身当前top5%风险池26+6，复用058树，首prefix features核对。seed17固定两个uniform组，用087自己的cohort/teacher/gold/空间/影子。无当前future标签和新收益学习器。

六个模型全部保存后才统一评价：旧TRAIN260498方向全体、102同作者旧VALID86832、新实体VALID2786，TRAIN-only raw/filtered、oldMRR/Top10drops/三新方向。每seed资格独立、teacher TRAIN与完整VALID均缓存复用。控制读102相同完整VALID指标；不混以前1024结果或把VALID扩展当未见TEST。

主要比较anchor fullabsolute、简单absolute与普通，以及两seed固定uniform确认；tree是否优于uniform要同时检查VALID、学新方向、实际非零步/梯度/累计差norm。非零步数不同，不把同rho称为同累计力度。若有候选再作强度或父状态确认，不追阈值。

记录准备加载、检测、训练/非零步、完整统一评价，历史影子/空间/拟合成本非零。无需独立CPU审计、SHA、归档或硬超时，无TEST。保留失败并据现场判定，不因观察超时重启。
