# 105完整负候选Replay强对照，启动前固定四配置

seed29/17各两组：full_replay_joint、full_replay_anchor；固定088/087普通prefix10含Adam，后220步host原tape、旧E/R冻结，附加梯度每非零步rho.1，joint Adam。anchor组追加094/095累计差空间约束，joint组不约束。相同事实与损失，不训练识别网络。

读取104已缓存Replay positives后220步：每步32个历史TRAIN事实、全260498历史池均匀采样；不限定teacher Top10，不生成新随机抽样。对每事实分别均匀合法头腐化/尾腐化，各1/2，完整13134候选、只过滤已见全部TRAIN truths，和104原先两种腐化的负采样期望相对应。

旧实体/关系冻结，旧正例、旧候选负例和旧positive正则项对可学新行的梯度为0，可从保护梯度计算中省略，只计算1876新候选分数；每方向分母仍为完整合法候选数（13134−该方向已知TRAIN正例数）。这给出数学上的完整负候选Replay梯度，不声称与随机Adam轨迹或FP32逐位等价；分数用FP64，E梯度回传FP32。完整头侧truth另由同TRAIN构建，不用VALID/TEST truths。

一次缓存qt/qh/合法新候选mask/完整分母，之后两臂复用。历史正例采样、学习tape、teacher、空间、普通影子不重建。完整池比10410负例更费算力，记录全成本，不把省略常数梯度称为额外原创方法。

四模型全部保存后统一旧TRAIN260498/旧VALID86832/新实体VALID2786、TRAIN-only raw/filtered、MRR/Top10掉榜/三方向。完整VALID teacher102直接复用、TRAIN teacher087/088复用；控制102指标与104稀疏Replay供比较。VALID全部已暴露，属于开发确认。

无TEST/硬超时/独立CPU审计/SHA/归档。已知Replay全池方差变化与候选预算不公平之处明确报告；如果普通完整Replay已解释收益，不把空间约束或风险模块包装成功。
