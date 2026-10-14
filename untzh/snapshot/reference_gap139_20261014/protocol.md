# 139：参考查询覆盖与答案分布诊断（固定只读协议）

问题：136–138的TRAIN无冲突盲区，是旧(h,r)查询缺少曝光，还是参加父模型拟合的答案与未见答案的分数分布不同？先分辨这两种解释，再讨论动作学习。

复用124完整旧TRAIN/VALID原Top10参考gold、querynorm、oldrank；137普通/固定Anchor/Replay终态rank；136旧TRAIN cohort、完整旧VALID prefix缓存及既有queries。无新评分/优化/随机数据/父训练。FB17 DistMult为1→2新实体子任务；WN ComplEx为0→1完整新TRAIN。TRAIN truths过滤，旧参数冻结。完整旧VALID均已开发暴露，仅事后诊断，不作为动作输入或新的未见确认。

固定统计：完整旧Top10资格群体gold/querynorm的分位数、负分数比例、终态掉榜；VALID按是否在全部旧TRAIN/其资格查询/136 cohort中有相同(h,r)分层，单独区分同查询不同gold。TRAIN/VALID按相同方向relation和旧rank1..10做共有分层，以VALID层权重描述TRAIN分布差异，报告未匹配质量，不推因果或显著性。prefix已有损伤与后段新增/恢复分开。不会将正gold、查询覆盖或无掉榜定义为安全。

GO仅限：若查询已覆盖仍有损伤且同层gold分布不同，值得设计父模型训练前的旧TRAIN内部参考实验。NO-GO：无稳定对应则不创建新学习器，不扫描分数阈值。即使GO也不自动升级方法、不扩P3；与86新TRAIN留出不同，但会减少父训练事实，必须重训匹配父和控制并报告代价。

效率：CPU读取已有缓存一次，记录总时及每父加载/统计费用；历史模型、gold评分与prefix准备成本非零，本轮无新增训练配置。无TEST、硬超时、独立审计、SHA或归档。
