# 092 新TRAIN查询空间上的保护附加位移

第二增量seed29，088initial/prefix10含Adam/tape/cohort/TRAINtruths/teacher只读复用。旧11258E/474R冻结，1876新增行，共20epoch440步，新臂只训练后220步，uniform32旧TRAIN答案/完整1876合法新候选fullabsolute，非零保护梯度norm rho.1。仍为新实体子任务，新增旧—旧事实只truth过滤。

每新实体收集全部新TRAIN跨新旧方向事实对应的固定old-neighbor×各自R查询向量，身份按(neighbor,r)去重，q归一化后double SVD，固定relative cutoff1e-10，缓存正交基。仅TRAIN，无VALID/TEST或future标签；人工逆边R不替换。记录rank/自由度/真实时间/内存，不做独立审计。

共同当前状态Adam joint-gradient提案与host-only提案的差作为附加保护位移，三个新臂persistent optimizer只记录host梯度：post_raw直接追加；post_null在每新实体TRAIN查询空间正交补上投影；post_normmatched将未投影位移缩放至本状态投影后全新行norm。按各臂自身状态匹配规则，不称共轨迹或逐实体等幅。090joint fullabsolute复用为原优化器状态控制，088none/absolute/absolute075复用非独立重复。

FP32实际追加位移对这些TRAIN正例分数变化逐步记录。线性保持只涉及固定旧端点且已见TRAIN事实，SVD截断及FP32舍入披露；不保护双新端交叉项、未见VALID分数、候选排序或后续所有host更新。主目标是终态学新—保旧取舍，不能以分数保持本身宣布效用。

三个模型全提交后复用088固定1024+1024VALID、全oldTRAIN及214095/416资格，raw/filtered/三方向/旧MRR与Top10共同评价。已有开发VALID暴露，未知测试不读。无硬超时、CPU独审、SHA门槛或归档；新准备/提案/投影/训练/评价与历史生成成本分列。相关正交子空间思想已存在，本轮不宣称新颖性/安全保证。
