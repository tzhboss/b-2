# 147：146失败规则的实际Adam/投影代价诊断

对象为原FB17/43/29 DistMult200、121zero/prefix10含Adam、原各自tape与第11epoch22batch、146真实gates及固定query抽样。复用144/145保存的这22个TRAIN学习梯度，不推进host/不训练新终态、不加载VALID/TEST。旧E/R冻结。

历史060已经试过单方向Euclidean/Adam度量保护梯度投影，142即时提案选择失败。本轮不提出或训练这样的规则：只在每父同一个固定prefix状态，对每个batch生成普通/fullAnchor/fullReplay/forecastAnchor/currentAnchor/randomMatched实际Adam一步提案并按146累计相对普通queryspace投影，所有提案彼此独立、每次从同prefix Adam恢复。

输入只TRAIN。新TRAIN三方向损失包括原负例及原regularizer；按该batch方向比例加权应重建base.learning梯度（固定首batch精度检查，非独审）。诊断：原保护梯度与三方向梯度cos、归一化alpha、原/投影后相对ordinary实际位移、三方向loss的真实提案差与一阶内积预测。固定参考probe为cohort排序后前256个参考（不重采样），其gold/oldrank来自已有TRAIN reference缓存，用完整合法新候选评估rank/softgap变化。

区分作用：原梯度SGD式预测与实际Adam/投影线性预测，两者符号/AP与真loss增大标签比较，无正例AP=null。实际提案是已发生的即时反事实代价，不是未来终态效用预测；probe数据与提交batch相同，不能代证VALID/长期泛化。局部配对差具有同状态一步解释，不能用它断言146全轨迹失败原因已经识别。

GO只限诊断：若三个父前瞻实际新增TRAIN损失均高于匹配随机、且能被TR可观察的实际位移线性代价识别，则保留“更新度量/投影改变冲突、代价需按实际位移判断”的解释；否则NO-GO共同的早期局部代价解释。无论何种结果，NO-GO直接恢复146触发或060/142规则，不改阈值/rho/池，不P3。报告所有状态/提案计数和费用，132个以上虚拟Adam提案不算新训练配置。
