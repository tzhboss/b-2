# 098风险路由收益与累计差强度控制

固定097 seed29底座/前10epoch普通prefix/后220步tape/32旧答案×1876合法新候选/fullabsolute/rho.1/joint Adam/累计差查询空间投影。两新臂uniform_tree_norm、random_tree_norm保留各自投影后的方向，每步累计差norm缩放至097anchor_tree同一步实际累计差norm。uniform固定088均匀selections，random直接复用097已保存随机池selections，不重新抽样、不检测。

目标长度仅来自既有TRAIN轨迹；无future标签。目的检验路由相对均匀/随机分配是否超出累计差力度影响，允许缩小与放大，记录scale、FP32长度误差和TRAIN分数残差。不是部署方法，也不等同完整Adam路径或有效梯度预算完全匹配。

复用092空间/094普通影子/088所有底座，不重SVD/影子/teacher。两模型全保存后同088固定TRAIN全体与old/new VALID1024统一评价；历史VALID已暴露，无TEST/硬超时/独审/SHA/归档。

主要比较同norm下tree、uniform、random的新MRR/三个方向、旧MRR/Top10掉榜。收益只有TRAIN或细小VALID事件差时保留不确定，不以一个标量把多个取舍合并称成功。历史特征拟合/影子/空间成本另计，计入本轮加载/训练/统一评价。
