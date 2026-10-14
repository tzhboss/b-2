# 072 差分学习率与旧模型自适应

DistMult200/FB-CKGE0→1/seed17；复用058普通prefix10/Adam新行历史、后370步同tape。新E lr .007，开放旧E lr .000007，R lr .00007；旧/R独立fresh Adam，未开放旧行保持逐位不变。权限R-only/64old+R/allold+R，64按newTRAIN端点count选。新旧参数学习率区别于062全部.0007。

15组：三个权限×original/mixed.5 host×none/replay，加三个权限original+klmass。replay为32旧TRAIN合格答案完整11258候选CE，屏蔽其他TRAIN正例，权重.1计入host；klmass=当前旧池分布对teacher旧池KL+新候选概率质量项，teacher前快照old E/R，当前old query/gold全重算；initial8batchRMS全参数norm校准rho .1、逐步cap .1。没有继续使用固定旧q作为当前query。

oldmodel可动，因此旧VALID资格固定原teacher A，不按新旧模型重新选分母；当前实际评分E/R全部动态。原冻结ordinary/uniform/replay作为上下文控制；同权限none/replay对照区分更新权限和旧约束效果。旧学习CE不是作者TADG原生复现、不是原创方法。

所有15模型完成再读已暴露VALID，筛选及非支配全TRAIN同063。全部配置/失败报告，成本train含旧约束/在线teacher评分，prefix历史费用归058。无TEST/硬超时/CPU独审/SHA/归档，产物/data/lc/untzh/adaptation72_20261014。单seed开发，未见独立确认仍缺。
