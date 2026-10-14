# 062：少量旧行解冻机制比较

目的：检验更新权限是否限制学新—保旧取舍，并隔离学习率、正则权重和蒸馏投影。开发探索，无原创或作者原生复现主张。

DistMult200、FB-CKGE 0→1；7505旧/3753新实体，关系冻结。旧TRAIN223284/新TRAIN37214方向记录。seed17，同052学习tape，前10epoch及Adam状态复用058，后10epoch共370步。选新TRAIN端点频数最高64旧行，ID破同频；其余旧行逐位冻结。新参数Adam状态step370保留，旧行单独Adam从step0开始。

8新配置：.0007下冻结普通/冻结absolute/解冻普通/解冻absolute/解冻均匀TAR/解冻熵TAR/解冻熵TAR+DGGP；.007下解冻普通。复用058的.007冻结普通及uniform作为历史匹配对照。学习损失、负样本、批次相同，不扫参数。原学习率对照检查降学习率混淆。

absolute同058的32答案×128合法新候选softplus、同lambda和rho=.1梯度限幅；解冻时重算当前查询向量，保留候选身份。TAR为64可更新旧行到旧模型的加权平方漂移均值，权重1或061缓存sigmoid熵；8步丢弃的TRAIN-only慢速解冻probe校准均匀TAR到学习梯度norm比例.1；两种权重及DGGP共享alpha并逐步限幅.1。这些配置均为自定适配。

DGGP：512随机旧TRAIN原始正例memory固定；每batch取与学习正例端点重叠的memory，无匹配用全512。参考梯度为当前与旧模型正例分数Huber(delta1)；总学习+TAR梯度与参考负内积时作欧式投影。记录触发次数及内积。正例蒸馏不等于新增候选排名保护，Adam更新也不等于一阶投影保证。

所有模型训练完再读已暴露VALID，无TEST。各组记录raw与TRAIN-only filtered MRR/Hits1/Hits10、三新方向及固定旧Top10资格掉榜。开发筛选：相对历史普通的三个新方向filteredMRR比均≥.98、旧VALID掉榜≤5、旧VALIDMRR≥历史普通的.99；只筛中组进行完整旧TRAIN评价，未评价为NA。该门槛用于节省评价时间，不冒称通过全部历史质量标准。所有VALID结果完整保存，不只报筛中项。

成本：prepare/train/evaluate和总秒数，GPU/Torch版本记录environment.json；共享prefix历史费用另属058，不称端到端部署成本。固定一个seed无跨seed方差/独立确认；本轮仅一个增量、少量旧行、固定配置，不外推全量解冻或论文TADG效果。

不设耗时硬上限、无独立CPU审计、SHA收尾、归档或重复合成资格。本轮代码/结果均在/data/lc/untzh/unfreeze62_20261013；历史资产只读。
