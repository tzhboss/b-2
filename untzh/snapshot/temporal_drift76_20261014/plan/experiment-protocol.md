# ICEWS05-15 时间持续学习排名诊断（076）

用户授权：下载公开数据，在 `/data/lc/untzh` 做准备、训练、评价、汇总；无新方法、独立审计、SHA门槛、归档或硬超时。第一轮 seed17，固定配置，不依据 TEST 调参。这里的新目标明确要求历史测试查询诊断，故使用标准 TEST；以前 FB15k 开发实验不接续调参。

## 问题和语义

1. 固定 `(h,r,?,日期)` 与其已记录 gold，未来年份 TRAIN 更新是否使其排名变差？
2. 相同历史日期与 gold 下，把候选集固定为上一年实体，再和当前已见实体候选集比较，分解参数更新与新增候选竞争。
3. 以无更新旧 E/R/历史时间行的控制验证“只有新增行及候选空间增长”能否产生历史干扰。

ICEWS05-15 是事件发生日期数据，不是职位/状态有效区间或事实撤回数据库。2015 年不同事件不能撤回 2005 年事件。因此本轮报告固定历史事件基准的排名恶化率（benchmark operational UFR proxy），不声称语义错误已人工确认。不计算真实世界 TVR 或“合理30%/错误70%”。固定日期查询的合理状态变化和当前日期查询的合理状态变化是不同目标，不能混用分母。

## 来源和切片

源：[TKBC作者仓库](https://github.com/facebookresearch/tkbc)，ICLR2020 Tensor Decompositions for Temporal Knowledge Base Completion；其 [download_data.sh](https://github.com/facebookresearch/tkbc/blob/main/tkbc/scripts/download_data.sh) 指向 https://dl.fbaipublicfiles.com/tkbc/data.tar.gz。原始 ICEWS 数据 DOI 为 10.7910/DVN/28075（包内 LICENSE）。保留原始标准 train/valid/test 及名称和日期。2005–2015 共 **11** 片，以 TRAIN 按日历年切分，学习只访问当年 TRAIN；不存在 2005→2015 共10片的计数错误。

完整训练，完整已具备训练实体/关系/日期表示的标准 TEST 双向查询，不抽样选漂亮案例。主 cohort 在其原年就可评价，后续保持同一身份。冷启动实体、关系、无训练日期的查询单列覆盖率，不能当正确排名/零损伤。原始 exact TEST 与 TRAIN/VALID 交集计数，污染项不进主 cohort。未来实体/关系名称仅离线编号，TRAIN首次出现之前不用于候选或训练。只从 TRAIN 定义活动实体；heldout-only实体不主动学习。评价候选为截至当前已见 TRAIN 实体，另加原年/上年固定候选反事实。

## 模型和固定配置

- TDistMult：`sum E_h * R_r * T_date * E_tail`，128维实数；明确为普通DistMult的日期乘法扩展，不称任何作者原生基线。
- TComplEx：`Re <E_h,R_r,T_date,conj(E_tail)>`，64复数维（128实数参数），公式核对TKBC模型实现；本次顺序设置/初始化为诊断适配，不是TKBC/TIE原生复现。两模型各有独立逆关系向量。
- seed17；E/R高斯标准差.1；日期实部初始1（TDistMult加.01噪声，TComplEx虚部.01噪声）。Adagrad lr.1 无momentum/weightdecay；2005共同50epoch，之后每年15epoch，batch1024；完整已见候选cross entropy、N3=1e-3，无时间平滑、无新保护loss、无VALID/TEST选择或早停。
- 每架构普通finetune和replay从同一2005模型及优化器状态分叉；每片维持Adagrad状态。Replay每年从此前原始TRAIN均匀取4096事实（不足则全部）；每个新TRAIN batch额外1/4 replay方向查询，`(loss_current + .25 loss_replay)/1.25`，额外成本单列；两架构共用同年memory身份，不称等计算预算。
- TComplEx额外 frozen_history 机制控制，与相同2005起点和新TRAIN匹配；每片冻结所有旧实体/旧关系/历史日期行，仅开放新行；并非有效学习新知识的方法宣称。
- 五臂全部训练和checkpoint完成后统一读排名，无按测试表现追训练。无硬时长终止。

## 评价定义

双向查询日期保持原日期，filtered按相同 `(query_entity,directed_relation,exact_date)` 的标准TRAIN+VALID+TEST正例屏蔽。未来年份正例不进入该历史日期的filter；另外完整保存raw排名。使用strict `>` plus average-tie（实际实现会明确），真gold自身不计竞争者。

每片完整评价当前和历史主cohort，以及当前涉及新增实体的查询。保存每个查询的原年/上年/当前排名，完整rank arrays和模型、时间/实体名称映射。报告MRR、Hits@10、平均/中位rank、HRD均值/分位数、恶化/改善率、旧Top10→掉榜率。

对于从s−1到s的固定历史查询：

`HRD_total = r(theta_s,C_s) - r(theta_(s-1),C_(s-1))`

`HRD_parameter = r(theta_s,C_(s-1)) - r(theta_(s-1),C_(s-1))`

`HRD_new_candidates = r(theta_s,C_s) - r(theta_s,C_(s-1)) >= 0`

两项精确相加；这是固定已训练模型的排名分解，不是两个完全独立训练机制的因果效应。按查询分类param-only/candidate-required/mixed；正参数漂移与新增候选rank贡献另报告，不把它们强行解释成真实世界错误类型比例。还报告相对查询原年候选的累计分解。

frozen_history旧候选rank必须不变（除数值并列处理）；若旧rank变化必须先检查评分实现，不能将错误叫参数遗忘。top竞争者案例标注其TRAIN首次出现年、是否在该日期有已知gold事实；“没有记录”不等于证明现实不可能。

## 可得结论的边界

存在性可由一个完整时间序列的多个查询/更新支持，seed17不能证明普遍效应大小。大量HRD同时伴旧MRR下降支持历史基准稳定性问题；固定候选分解和freeze对照支持新增候选成分。新颖性需后续对TIE、CKGE候选协议和TKG forgetting文献系统对照，本轮不宣布独立新问题。TIE原生方法不在这轮简单诊断内。
