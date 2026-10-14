# 148：TRAIN条件答案类别与候选组可比性，单次评分探针

147说明即时TRAIN loss下降不保证长期排名效果，停止即时loss动作rule。新假设为新旧候选组分数缺乏可比性，而非继续调整保护强度或risk阈值。只用FB17原121zero普通终态；不训练新父/前缀/增量模型，不改变可训练参数或读TEST。不将评分后处理冒充已完成的风险自适应更新方法。

从旧+新增+extra全部已知TRAIN truth集合按(h,r)计数旧/新答案，去重同候选；p0为全TRAIN新答案占比。固定Beta收缩质量2，p_new(q)=(n_new(q)+2*p0)/(n_total(q)+2)，未见query回退p0；无搜索。计数/全局与随机映射先保存后读取任何VALID query/target。固定rng1480017在TRAIN键内按head为旧/新分层置换p，打乱上下文但保留两层分布。

四臂raw（既存排名）、global_group（固定p0）、context_group（TRAIN条件p）、shuffled_context（同分布错配）。令z_old/logsumexp旧候选原分数、z_new新候选原分数，delta=logit(p)+z_old-z_new；只将新候选分数加delta，旧候选分数保持。它把两个组的总softmax质量匹配p，组内排序不变；是已知分组概率归一化，不称概率已校准或原创。所有候选参与z，无评价gold-dependent过滤；实际评价只过滤TRAIN truths。

报告完整旧TRAIN/旧VALID/newVALID2786的raw/filtered MRR/Top10掉榜/漂移、三个新方向、实际delta分布/费用。old-query旧candidate logZ按独立(h,r)只算一次并缓存，模型/排名复用。新query随newE固定普通终态一次评分；raw零偏置重建必须匹配既存rank。固定126Anchor/146Replay原排名是强简单控制，不给校准模块底座功劳。

GO仅允许评分可比性候选继续：context在本父新MRR及三个方向不低于普通*.98，同时旧MRR/掉榜至少不差固定Anchor，且至少一项优于global_group和shuffled_context；否则NO-GO此组校准，不扫先验质量/温度或更换概率公式。即使GO仍需回到风险→更新动作目标，不能用评分后处理缩小目标或直接P3。一个已暴露开发父的筛选，不是方法确认。无硬超时/独审/SHA/归档/自动push，history79独立。
