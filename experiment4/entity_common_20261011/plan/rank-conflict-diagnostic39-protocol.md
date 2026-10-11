# TRAIN-only保护冲突诊断039

前提：037/038完整终态及所有六seed/policy审计成功；两种容量豁免方案
均NO_GO。保留032和038门槛、失败及全部既有模型。不是新方法比较。

固定seeds17/29/43，原Full／Scoped与容量Full／Scoped各3753实体终态。
同一源NewEntityOnly初始E/R和目标终态；只读取旧/new TRAIN与已合格
scope/endpoint审计。没有VALID/TEST查询或rank读取，没有权重更新、
保护合同变化、预算改变、系数/radius/seed搜索。每seed限1200s，
三张现场空闲RTX3090并行、各CPU/BLAS两线程，目录experiment4。

每个保护查询的head、reference都是旧实体且旧E/R精确冻结。因此
参考最低分cut不随新实体替换变化。完整计算7505旧候选的strict-greater
计数o以及3753个终态新候选计数n，确认o+n<10。将单个新实体e替换为
源目标时的新计数为o+n-[current_score_e>cut]+[source_score_e>cut]。
不改变其他实体，计数>=10是该查询的完整成员保护违规。逐查询／实体
扫描所有scope，不使用working-set来代替完整资格。

区分：
- o=9时旧不可动候选已占满允许slot。若源目标跨cut，改变其他新实体
  或仅重排新实体更新不能解除该查询的此项限制。
- o<9但单实体替换仍违规：其他新候选已占用剩余slot。这是终态竞争，
  不直接证明换顺序会改善，也不代表全局可行性或最优性。
- 原终态新行不等于源目标，但源目标在整个终态scope中可行：报告为
  可复查的协调机会，不直接作第二次更新或按此选方法参数。

统计每个新实体是否有上述任一限制、query/entity pair数量、初始旧slot
直方图，并按newTRAIN affected query与旧TRAIN是否包含最低保护成员
披露。未标为TRAIN真的成员不自动称为错误知识。按预测／TRUE成员
组成记录scope类别，不据类别切换保护合同。

GPU使用原固定FP64二叉树scorer，列tile256。32条完整查询fixture／
policy／seed用NumPy独立同序二叉树重算旧、目标和当前全部候选分数、
cut、counts与全部3753个替换资格；所有fixture位值和计数必须精确一致。
完整GPU范围结构身份与旧E/R固定检查、原终态数值审计资格同时核验。
CPU独立评分仅覆盖固定fixture，不能称全GPU结果均已CPU独立重算。
合成slot controls覆盖旧占满、新候选竞争、替换自身扣除与cut并列。

原始fixture保留artifacts，轻量摘要和hash提升results；每个seed失败独立
留档，不重写原结果。诊断没有预设效用GO门槛：只能用于确定下一机制
的触发原因。任何后续实质新方案需要另注册、固定对照及原完整任务
收益门槛，不把本诊断的可行性或覆盖计数当成新方法效果或创新性。
