# 118 WN_CKGE真正ComplEx评分比较

ComplEx100 complex维=200实参数，与112 DistMult200实体/关系参数量相同。评分Re(<h,r,conj(t)>): tail-query=(hr*rr-hi*ri,hr*ri+hi*rr)，head-query=(tr*rr+ti*ri,ti*rr-tr*ri)。父旧E/R固定TRAIN50epoch、Adam .007、std .12、softplus正负均值+.0005同口径正则，无VALID早停/选模。112 parent-learning/new-learning negatives直接读取复用，不重新随机；数据WN0→1旧24567/新4093/22R，全部新增TRAIN11160方向参与。

先做实际TRAIN样例评分与两端梯度公式必要核对，并记录已训练非零R虚部/非对称分数；不能把DistMult参数重命名ComplEx。新初值同1087随机张量，standard std .12/low=.25*standard=.03。两种prefix10各只训练一次。七增量配置standard_none/low_none/simpleabsolute/answeranchor/queryanchor/fullReplayjoint/fullReplayanchor；后110步，各保护rho .1，simple8batch RMS+.1 cap、旧E/R冻结。模型相关TRAIN teacher/cohort/q-space各生成一次，新空间按固定端角色用tail/head query，不能只乘旧邻居×R；SVD double cutoff1e-10。

Replay历史正例及合法头/尾mask、完整分母复用112，qt/qh按ComplEx父重建一次；完整4093新候选梯度、旧常数省略。在线普通参考相对各自low_none，严格终态全等，无逐步快照。标准与低初值普通对照避免混算初始化收益。

七模型全部保存后统一完整旧TRAIN66960/旧VALID22320/newVALID3720，TRAIN-only raw/filtered、三个新方向和掉榜。新父资格重新计算不可复用DistMult分母；旧rank缓存+只新增竞争评分，用ordinary/query首中尾必要原评分核对，失败则原完整评分。保存标准与low的expanded/prefix/none阶段rank用于后续风险诊断，不重训。首次缓存/父训练/影子/评价费用真实计入；单seed开发模型迁移非作者原生复现/原创/未见TEST确认。

无TEST/硬超时/CPU独审/SHA/归档，不扫新参数追正。
