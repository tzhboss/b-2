# 数据证据

原始已下载FB_CKGE快照0/1，来自远程literature_baselines_20261010/repos/FastKGE/data/FB_CKGE；文件必须绑定既有results/KGE-LIT-002/fastkge-fbckge-s3407/independent-audit.json的data.sha256。预检产物只描述划分、覆盖和交叉重复，不是训练性能。逐query产物放远程artifacts，尚无主结果图。

022R2的synthetic_models.npz仅是共享参数求解器的构造控制，放远程
artifacts/shared-rank-controls22r2；results/KGE-ENTITY-COMMON-SHARED-RANK-CONTROLS-022R2
仅提升独立审计和轻量汇总。没有真实图谱方法效果图；合成新rank13→8
不得进入真实方法主表，也不能用作投稿性能结论。

023R3缓存检查仅有合成正确性证据；024为真实TRAIN初始模型全候选评分、
保护成员编译及资源测量。scope.npz与128查询全候选fixtures.npz逐seed保留在
远程artifacts/shared-rank-scope24；轻量审计见对应results/summary.json。
12.6–14.6秒是保护评分成本，不是joint优化总时长或学习收益，不能填入方法
主表作为完整编辑成本。未新增真实方法效果图。

025／025R2为合成求解器控制，026为8步真实集成审计，不能绘成完整方法
收益。027使用原已审计数据和源模型，新增数组/模型/逐步状态保存在远程
artifacts/incremental/shared-graph-*；v3主表从canonical读取运行与审计状态，
只有通过全部数值审计的seed填成绩。028固定VALID方向门槛单列。图形若
后续生成，须绑定同版本table.json，区分全保护与限定保护合同，保留完整
旧/新查询、各seed差异以及源训练与范围准备成本。
