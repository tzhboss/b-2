# 057：MEOR适配与replay比较

远程目录：`/data/lc/untzh/meor_replay57_20261013`。仅准备复用→训练→评价→汇总；无固定超时、独立CPU审计、SHA冻结或归档。不会重跑055，也不覆盖056。

## 公式与适配边界

原文：[Matched Excess-Outranker Regularization for Candidate-Set Interference in Continual Knowledge Graph Embedding](https://arxiv.org/html/2608.24273v2)，项目已有原文HTML逐式核对。公式12–16：

`z=(s(q,e)-s(q,a))/tau`；`U=log(mean(exp(5*relu(z))))/5`；`old_mean=mean_j(U(old_reference_j))`；`MEOR=mean_occurrence(relu(U(new)-old_mean)^2)`；总目标为host+lambda*MEOR。

尺度detach；候选排除累计TRAIN正例；每个多重集保留重复；先平均旧参照再hinge。beta5、4参照、最多256尺度样本来自实验段。原文只命名score-blind gradient-balance，未给出可执行lambda选择细节。

本项目degree分箱、fallback池、每步32个旧TRAIN filtered Top10资格答案、初态前8批RMS rho=.1、DistMult200、Adam .007、20epoch等是明确的适配。旧E/R固定，使q/gold/scale/reference_mean可精确复用。只读复用`signals56_prepare_optimization_20261013`全部740批紧凑缓存和052学习/保护tape；大小mtime参数做普通缓存失效检查，无SHA门槛。

## 四个新训练臂

1. `replay`：复现项目052的weight1 sampled softplus replay，用同初态、学习/保护tape。实际新训练，并检查端点是否与历史052逐位相同。
2. `meor_stepcap`：仅新事实host，加完整候选MEOR及逐步限幅。
3. `absolute_stepcap`：同上host、同完整候选和保护答案，加softplus绝对保护及同限幅规则。隔离信号差异。
4. `replay_plus_meor_stepcap`：与臂1完全同host/保护tape，再加MEOR及限幅。直接检验叠加MEOR相对项目replay是否有益。

原文Replay指同host下lambda=0；项目052 sampled replay具有额外负候选压分，不能互换命名。共同冻结旧参数时，旧正例项本身无可训练新参数梯度；本次保留项目已有目标，不声称作者host原生复现。原056 fixed-MEOR/absolute和历史replay .1/.3/1只作已有参照，不重跑，不算独立重复。

限幅是本项目适配，不是作者方法或原创贡献：各臂分别按共同初态前8批新行host梯度与保护梯度RMS比rho=.1定lambda，固定该上界；每步`alpha=min(lambda,.1*||g_host,new||/||g_protect,new||)`，detach alpha。保护梯度为零时alpha=0；host梯度为零时保护关闭；非有限直接失败。实际Adam输入为两新行梯度之和，旧行置零。此规则控制Euclidean梯度贡献，不保证Adam位移比例或排名安全。记录逐步未限幅比例/实际比例/lambda/alpha/激活/loss/cosine。

## 评价和结论规则（训练前确定）

末态为主，不做中间VALID选点，不读TEST。保留同TRAIN-only filtered和raw合同、全部三方向MRR/Hits1/Hits10、旧MRR和固定A资格Top10掉榜：oldVALID filtered432、oldTRAIN filtered187470。所有新模型写完后才评价已暴露VALID。

主要配对：MEOR vs ordinary（历史复用）；absolute vs ordinary（历史复用）；MEOR vs absolute（新配对）；replay+MEOR vs replay（新配对）。质量判据沿用原阈值：新整体和每方向MRR至少基线98%，每方向Hits10至少95%，旧VALID MRR至少98%，旧VALID掉榜至少减少25%，旧TRAIN掉榜不得更多。全部判据同时满足才称本固定开发点有效取舍；少数掉榜计数和单seed不作独立确认。成本单列实际calibration/train/evaluate和历史cache/tape成本，不能把缓存费用记零或共享双参照准备当单方法部署成本。

用已保存rank做5000次(h,r)簇配对bootstrap，报告新整体/三方向MRR、旧VALID MRR/掉榜率的描述性95%区间；不是独立审计、跨seed验证或同时推断。不因结果补权重/网格。失败保留原样；只有长时间无进展且进程计算活动支持卡住才结束。
