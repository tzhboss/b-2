# 077 保护分数尺度归一化与损失组合

seed17/DistMult200/FB-CKGE0→1/oldE,R冻结/058prefix10/后370同学习tape。只读075完整pool/q/candidate缓存。12配置：cohort/globaluniform/globalrelation各scores/qnorm、scores/teacheroldscorestd的softplus；cohort absolute/mass_sample按各梯度同rho.1尺度组合mass份额.25/.5/.75；globalrelation qnorm/std各learningweight1.1；cohort linearmean(score)对照。

std由teacher旧entity协方差qᵀCov(Eold)q计算，不扫描每query全候选；qnorm/std detached，oldQ固定。mass_sample旧logZ仅cohort重新一次计算并缓存，pool估计定义沿067。每component初态8batchRMS校准rho.1、每步cap.1，组合两个cap后的方向按份额相加，norm上界仍.1。32×128候选，关系/查询分布同075，所有source/TR规则固定。

全部12训练完成后VALID，筛选/非支配TRAIN沿063，失败全报，单seed多重开发非独立确认，无原创主张。costs含归一化/缓存计算和训练评分；无TEST/硬超时/独立CPU审计/SHA/归档。产物/data/lc/untzh/normalized77_20261014。
