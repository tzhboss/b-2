# 063 多方向保护机制筛选

DistMult200 FB-CKGE 0→1，seed17；旧E/R冻结，前10epoch与Adam状态复用058，后10epoch同052学习tape共370步。TRAIN-only资格16384答案/128合法新候选缓存；完整3753池依法过滤已知TRAIN新正例，固定旧query/gold/scale复用。无TEST，训练全完成后才读暴露VALID。

15配置见experiment63.py ARMS：absolute重复一致性参照；相对gold的softplus gap（128/full）；margin平方hinge全池；37次拟议Adam更新外推margin；Top10剩余容量边界softplus（含MRR软guard消融）；风险路由margin；梯度冲突折减风险分数路由absolute；实体级梯度投影absolute/gap；Adam学习更新后独立保护absolute/margin；旧头→新gold学习重加权1.5及其absolute组合。分属损失、路由、优化器、学习配置，不冒称一个原创完整方法。

32答案/步，路由臂26来自top20%+6均匀。除postAdam外，同初态8batch各损失RMS校准rho .1、逐步cap .1；absolute复用058lambda。精确零hinge不以虚拟norm伪造校准。postAdam保护位移限制为实际Adam学习位移norm的.1，单列为不同更新规则。学习方向重加权有单独无保护对照。

完整保存全部VALID raw/filtered MRR/Hits1/Hits10、三方向与固定旧Top10掉榜。筛选规则：新三方向相对普通filtered MRR比≥.98、旧VALID掉榜≤4、旧VALIDMRR≥uniform的.99。仅筛中且VALID三目标不被其他筛中组支配者全旧TRAIN评价；NA未计算。比较历史普通、058uniform、057replay；复用不是独立重复。最终候选还须检查旧TRAIN损伤/旧MRR及所有方向，不能只看总体新MRR。

保护信号与选择成本计入训练，准备/评价单列，prefix历史成本归属058；固定一个seed，无跨seed方差，多次开发搜索可能过拟合，独立确认尚缺。environment记录GPU/Torch。无硬超时/独立CPU审计/SHA/归档；新资产仅/data/lc/untzh/search63_20261013。
