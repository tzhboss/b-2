# 120 低尺度底座的固定前瞻路由联动

FB seed43/17/29，116 low std .03底座固定，不训练前缀、不重抽负采样、不重新评分teacher。119的旧TRAIN cohort预测及epoch10当前rank只读复用。每父四配置：uniform_safe、pressure_safe、tree_safe、random_safe；只对epoch10仍安全对象路由。风险池为当前安全对象前5%，随机池同大小。每步风险/随机池26+安全池6，不重复；uniform_safe从安全池32。池固定不按终态反馈改变，采样一次缓存。四臂同rho .1当前保护梯度norm、同full1876合法新增候选absolute信号、同Adam和累计ordinary-anchor正交补投影。持久Adam含joint梯度状态，影子仅ordinary学习；不把当前安全筛选当零费轻量部署。

116 none/absolute/anchor_null/query_uniform只读复用作历史控制，非独立重复。12新模型全部提交后评价完整TRAIN260498/旧VALID86832/新VALID2786，三方向分别报告。路由只用TRAIN，119 VALID性能已暴露开发证据，不称独立科学确认。比较random集中池排除仅集中效应、uniform_safe排除仅跳过已损伤事实效应；同梯度rho不意味着实际位移/新学习代价一致，实际norm另报。空间只保已见跨新旧TRAIN正例分数相对ordinary，不承诺新—新/VALID排名安全。

历史119检测（采样、特征、树/当前rank费用）非零；训练时间包含在线ordinary影子及投影，评价与准备分开。无TEST/硬超时/独审/SHA/归档。不据单seed结果中途改配方。
