# 060：风险识别后的干预协调

058/059显示风险预测准确不保证选择保护收益；旧头→新gold额外损失仍存在。复用058 cohort、prefix10与059使用的tree_current，不生成标签或参照。

固定seed17，共享前10epoch普通学习与Adam状态。后10epoch以uniform或tree_current选择，保护仍为32答案×128合法新增候选softplus，lambda与.1梯度限幅不变。新增2×2控制：Euclidean或基于host拟议Adam二阶矩的对角度量下，将保护梯度对旧头→新gold学习梯度的负内积分量投影移除。统一/风险均应用，不把这种已有梯度协调思路作为创新。

投影只是当前一阶方向控制，Adam最终位移及排名均不获保证。记录实际触发步数与投影后的内积；若投影无作用应报告。所有新模型完成后一次VALID开发评价，先报告全部候选；只对TRAIN-only filtered三新方向MRR≥普通.98且旧VALID掉榜≤5的候选追加完整旧TRAIN评价。既有ordinary/uniform/current_all128控制直接复用。此筛选不改058/059或历史raw门槛，不是未见确认。

新代码/日志/模型/结果在 /data/lc/untzh/risk60_20261013；无硬超时、独立审计、归档或重复资格。成本单列训练/选择/VALID与必要完整TRAIN评价。
