# 093 joint-Adam状态下的查询空间位移比较

编写时远程连接短暂超时，随后恢复，092完整summary已读取并镜像，原进程现场gone，无093既有目录/进程。代码上传后compile并唯一启动，不重跑092。此句不替代实时启动/状态证据。

固定088 seed29第二增量20epoch440步/10epochprefix Adam/学习tape/cohort/teacher/合法候选，旧11258E/474R冻结、新1876行更新，uniform32旧TRAIN答案、full1876合法新池absolute、nonzero保护gradientnorm rho.1。复用092 doubleSVD query-space缓存，不再准备空间。

两新臂共同persistent optimizer保留joint gl+alpha gp的一二阶moments；同状态host-only Adam提案仅作反事实端点，不写persistent state。joint提案减host提案定义附加保护位移：joint_null按per-new-row已见跨端TRAIN查询空间正交补投影；joint_normmatched按该当前状态投影后的全新行norm缩放未投影位移。控制不代表同轨迹同Adam位移。下一步host提案仍受过去joint moments影响，所以TRAIN分数近似不变仅对当前附加修正成立，不是累计不伤新学习的保证。

复用092host-state-only三个控制及090jointfullabsolute；区别是optimizer历史权限，所有组都必须报告新旧质量，不能以动量更强归因方向新颖。已知正交子空间思想有关，不称原创/全局安全。

两模型全提交后同088固定1024oldVALID/1024newVALID及全260498oldTRAIN评价，214095/416原资格，raw/filtered/新三方向/旧MRR与掉榜一起报告。所有开发VALID已暴露。没有future标签、TEST、硬超时、独审、SHA或归档。准备缓存与投影/提案费用显式记录。
