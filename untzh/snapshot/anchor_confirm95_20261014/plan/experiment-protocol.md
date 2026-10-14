# 095固定094配方seed17确认

在启动前锁定配方；历史seed17 VALID已暴露，这是开发阶段跨父模型确认，不是未见TEST。

- FB持续知识图谱snapshot1→2新实体子任务；DistMult200，旧11258实体/474方向关系冻结，仅学习21566含新实体的方向TRAIN事实，15648新增旧—旧事实仅truth过滤。
- 只读复用087 initial/prefix10含Adam/learning tape/cohort/protection schedule/teacher及评价查询；不重新抽样。20epoch440步，后220步保护。
- 普通影子从相同prefix沿既有tape生成一次，每步新实体行缓存，末态必须与087 none逐位一致。其时间、330176000字节缓存计入成本。
- 使用seed17旧E及R、同一新TRAIN重建查询空间，不能复用seed29向量。归一化double SVD，相对cutoff1e-10，每新实体单独处理，两个方向保留各自R。
- 两组anchor_null/anchor_normmatched固定094公式。每步32旧TRAIN答案×完整1876合法新候选fullabsolute；非零保护梯度norm=.1×host norm，joint Adam持久状态。
- anchor_null把相对同一步普通影子的累计参数差投影到上述空间正交补；normmatched把自身累计参数差按自身投影后的全行norm缩幅。两者不保证跨轨迹相同最终累计差norm，不能把此控制称为严格同总强度。
- 两模型全部保存后统一评价既有oldTRAIN全体与old/new VALID各1024；TRAIN-only filtering，raw/filtered并列。资格214239/419独立于seed29。
- 主要比较普通、087简单absolute/absolute075和091固定fullabsolute/rank信号：新MRR/三方向保持、旧MRR/原Top10掉榜、raw行为；不以整体均值掩盖方向损失，不因单seed上涨宣布成功。
- 单步记录相对普通的跨新旧TRAIN正例分数最大/平均残差、累计差norm、冻结基本检查。不是VALID/双新端/负例/排名安全保证。
- 成本分别记录准备、空间、普通影子、各保护训练与投影、统一评价；历史底座另计。无TEST、硬超时、独立CPU审计、SHA或归档。
- 此轮未加入风险路由；只有动作获得重复支持后才检验识别增益。正交约束相关已知方法，不以模块数量或SVD作为原创性声明。
