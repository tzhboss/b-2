# 149：固定保护的反事实矩外推，先检验收益预测

不同于144仅普通训练风险、147单步代价，本轮把固定Anchor/Replay的TRAIN梯度加入同平稳Adam矩模型，检验终态相对普通的旧查询额外救回/排名收益；不是84–86真实rollout或142即时代价rule的修补。只原FB17/43/29全事实zero父、121prefix10/Adam/各自tape、第11epoch全部22batch及146固定保护条件。

固定prefix不提交训练更新。普通梯度复用144/145；fullAnchor和fullReplay原32query抽样/负例各22batch，在固定prefix求gp，按原rho.1逐batchnorm得到gl+alpha*gp的均值/二阶矩。按原Adambetas/lr/eps/step外推220步；Anchor每次相对同公式ordinary位移投原TRAIN queryspace，Replay不投影。旧E/R冻结。此平稳近似已知普通风险过量，不调H/rho/方差或加网络。

三父的TRAINcohort/完整旧VALID原Top10资格预测全部保存之后，才加载146普通/固定Anchor/Replay排名作为标签。既存终态已在开发中暴露，本次是固定新公式的回溯诊断，不是独立前向确认；VALID gold仅用于离线未知答案之外的参考评分，不用于生成保护梯度/训练动作。

预测RR增益、rank decrease、预测救回指示；对照current rank/current gap/ordinary forecast rank/旧rank边界。目标为普通掉榜且动作救回、动作新增Top10害、RR真实改善/误差/符号，分TRAIN/VALID。没有事件APnull，不把严重风险当真实需要保护。

预设GO仅限下一收益诊断：两个动作的预测RRgain分数在三个父各自VALID extra-rescue AP都严格高于current rank和ordinary forecast rank；若任何不通过则NO-GO本平稳动作矩模型，不调H/强度/公式，不据零风险TRAIN声称泛化。普通forecast row应与144/145均值完全相同作为实现复用检查。即使GO也未提交自适应更新、未证明新旧联合收益/P3。

零新父/训练终态，66旧学习梯度复用、132保护梯度新增、1980矩迭代/3父×3轨迹×220虚拟时刻（非真实训练步）。日志/summary/成本镜像，模型/cache留远程；无TEST/硬超时/独审/SHA/归档/自动push。
