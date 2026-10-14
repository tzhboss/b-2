# 第十五轮：已成立时间冲突的全量缓存覆盖

本轮只读五臂各81,772条已保存的历史方向查询。九个实体名对应已读成立时间来源，均采用保守年份边界；“尚未成立的组织作为动作主体”和少量明确双边互动的客体单列。声明、意图、要求等关系中的客体，暂只记名称日期冲突，不直接当真实参与者错误。未标记的查询不能据此视为合理。

| 模型/训练 | 更新后名称日期冲突 | 参与者时间冲突 | 参与者冲突且gold排名下降 | 原raw rank=1的条件错误漂移见证 |
|---|---:|---:|---:|---:|
| TComplEx普通微调 | 22 | 15 | 13 | 6 |
| TComplEx Replay | 25 | 16 | 12 | 3 |
| TComplEx冻结历史参数 | 10 | 9 | 9 | 0 |
| TDistMult普通微调 | 29 | 12 | 9 | 4 |
| TDistMult Replay | 26 | 14 | 9 | 6 |

所有臂的对应更新前Top1冲突数均为0。TComplEx普通微调的15条参与者冲突：AAP7、Tasnim5、LNP2、Nusra成员1；10条主体、5条实际互动客体。13/42,851约0.0303%只是这一组已核对组织的冲突覆盖率，不能称总错误漂移率。保守边界、仅查Top1、有限组织集合均限制覆盖。

原raw rank=1意味着更新前没有候选超过gold；如果最终唯一Top1是时间上不可能的答案，就新增至少一个错误排名位置。因此六条可以作为条件见证：仍需接受原gold确实在查询日成立，以及组织身份/实际发生日期语义。它们均在固定随机40样本之外，不混入随机样本、不给原gold补独立支持。六条见证来源涉及Times of India、BBCMonitoring、IrishTimes、ABC、TheHindu，已从原匹配缓存取得具体StoryID/句子，详见targeted-conflict-gold-provenance-round15.json。

冻结历史参数也有9条时间冲突，支持新增候选能够引入时间不相容答案。但其旧候选排序与其他臂、起始gold质量不同，不能直接以9对15解释两机制的因果份额，也不能把较少Top1冲突当较小整体遗忘。

固定40条来源继续补充：S06的[电文镜像](https://wikileaks.jcvignoli.com/cable_09PARIS1558)将JackLang北京访问写为2009-11-05至09，[AFP图片说明转载](https://www.21news.be/jack-lang-largent-comme-systeme-privileges-impayes-et-chute-dun-personnage/)明确11月5日在北京。电文是未经官方发布链验证的镜像；AFP只用实际读到的图片说明。这些补充原gold证据，但通用“法国特使”可以代表多人，不能排除海牙另一名特使。S11的[RTP/LUSA海军合作正文](https://www.rtp.pt/noticias/pais/portugal-disponivel-para-melhorar-marinha-angolana_n28551)没有显示所需6月20日日期，不能把别日合作或长期关系当同日正例。

针对六条条件见证另查来源：[Ulster大学CAIN电视目录](https://cain.ulster.ac.uk/heathwood/static/2006.html)记录Durkan与Ahern于3月2日会谈，对应候选gold却为3月3日。目录正文已读，原视频未观看，IrishTimes原故事未对应；保持精确3月3日gold未知。另实际读到[ABC4月16日宣布结束车臣行动的正文](https://www.abc.net.au/news/2009-04-16/chechen-war-over-russia/1653598)，内容回顾1990年代战争及1999年行动启动，没有给出4月16日具体叛军暴力事件；尚未对应内部StoryID，不把历史回顾当同日gold正例，也不推断当日无其他袭击。

固定样本结论不变：9条gold有独立同日语义支持；2条最终rawTop1有同日语义支持；最终Top1明确证伪0、未知38；整张排序错误漂移已确认1条S38，其余39条尚未完整判定。合理/真正错误总体点估计仍为空。S39仅有条件漏标解释，不能填成确定合理。

缓存扫描实际0.557495秒，六条原故事元数据读取0.110005秒；均不含既有训练/评分、来源检索解释、编码及镜像费用。无新训练、神经重评分、独立审计或硬超时。本轮网页回执保存在round15-web-receipts.json。
