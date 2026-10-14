# 075 分层保护取舍与公平控制

同seed17/DistMult200/FB-CKGE0→1/oldE,R冻结/058prefix10/后370步同tape。复用074完整资格187470答案/候选生成固定seed与旧cohortcandidate，最终保存shared-cache便于后续只读复用。每步32×128候选。

21固定配置：relation等权power1×旧头新gold学习weight1/1.1/1.2×rho .1/.15/.2；relationfreq逆幂power.5/.75×weight1×rho.1/.15/.2；globaluniform power0×weight1/1.1/1.2×rho.15/.2。概率p∝relation答案数^(-power)。相同host负样本tape/L2；学习weight无保护参照见073，额外uniform控制排除纯增强保护解释。各分布/host按初态8batchRMS校准lambda，逐步rho cap。

21模型全部训练完后VALID，筛选/非支配TRAIN沿063。所有参数点/失败全报，固定原门槛，开发选择不是独立确认。不把分层采样视为原创，需之后明确可用机制与确认。新产物/data/lc/untzh/stratified75_20261014，无TEST/硬超时/CPU独审/SHA/归档，成本prepare/cache写/训练/评价总量可查。
