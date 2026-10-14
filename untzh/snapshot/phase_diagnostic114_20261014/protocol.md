# 114 WN候选扩展/前十epoch/后十epoch分解

113 prefix10→20在TRAIN cohort和旧VALID参考均没有新掉榜事件，不能估计风险AP/召回。固定读取112已有初始扩展模型、prefix10、ordinary终态与旧候选teacher，在旧TRAIN66960/旧VALID22320分别计算三个阶段原teacher Top10资格中的掉榜/救回配对；不新训练/改变方法。

仅新增评分初始扩展模型和prefix10的新候选4093列，旧gold/teacher rank/过滤缓存复用，ordinary排名既存。区别加入新候选时已有竞争与之后优化造成的转移，不直接当独立干预因果效应。全部为已暴露开发VALID诊断，不能用无晚期事件说没有风险或识别器成功。无TEST/硬超时/独审/SHA/归档。
