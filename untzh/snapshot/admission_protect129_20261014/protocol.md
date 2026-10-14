# 129 接入尺度决策＋保护的完整流程比较

128历史参考校准规则/分组/尺度选择全部只读，不据本轮未来排名修改。五父与共同absolute/answer-anchor保护：比较固定std .03、固定half .015、固定zero和128自适应尺度选择后的保护。FB四个参考输出来自116/126，WN来自118/127；缓存复用不计新训练或独立重复。

仅WN DistMult低尺度.03的absolute/answer-anchor/full-replay-joint历史缺失，需要三新后段模型。115 low初态/prefix含Adam/ordinary、112 tape/cohort/query空间/replay缓存复用，不重训前缀。所有三终态提交后完整评价，规范同127，额外Replay不作为FB共同两臂对比的一部分。训练影子同115自身普通终态，成本含在线shadow。

按128旧VALID(h,r)校准/评价划分，旧评价只用独立group一半；校准不混入heldout。新VALID完整且不用于尺度选择。历史参考提供了额外旧gold/rank，不当TRAIN-only策略；已暴露开发数据，回溯流程比较非未见确认。保护梯度仍只用TRAIN，尺度选择来自旧历史参考初态概率，未来训练终态不作选择输入。

若自适应没有共同超过最强固定尺度＋同保护，不能用对弱标准.12的收益宣称风险决策必要。记录raw/filtered、旧MRR/Top10、新三方向、历史费用，当前不宣称原创或全局安全。无TEST/硬超时/独审/SHA/归档。三新配置之外30个固定保护评价与10个alias为缓存统计，不计独立模型。
