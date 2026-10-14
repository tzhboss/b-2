# 107评价优化验证完成

原SSH47317自然exit0，PID887718已退出，程序35.824秒。固定105两seed两方法×旧TRAIN/完整旧VALID共八rank数组，与既存完整评分逐条一致，零差异；seed17 joint旧VALID64行因临界new-gold分数回退原完整评分，其余无fallback。

旧E/R冻结时，teacher原候选rank固定，只增加合法新候选超过gold的数量。每seed首次gold+新truth过滤索引缓存准备7.274/7.407秒，缓存旧TRAIN27.408MB、旧VALID8.948MB。原teacher生成费用仍非零。缓存后旧TRAIN单模型3.636–3.851秒、旧VALID1.209–1.226秒；新实体VALID含可变旧查询或gold，继续原完整评分。

后续FastOld必须检查旧E/R冻结，口径strict greater、TRAIN-only filtering；不适用于旧参数可变/关系新学/时间嵌入可变。首次原64行FP64 GEMM提取gold，临界新增分数回退原评分；已保存teacher/排名基本核对为实现复用所需，非独立科学确认/CPU审计。未同场重新跑原完整评价，不宣称精确加速倍数。无新训练、TEST、硬超时、SHA、归档，goal仍active。
