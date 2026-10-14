# 085：固定训练段决策规则跨seed比较

084提出并固定的local(.998逐段)/cumulative(.98相对普通影子)两规则，保持epoch10开启、absolute方向、候选乘数0/.5/1，不改预算或挑时点。复用080历史seed29/43初态、prefix10、学习tape、资格cohort与校准。每seed新TRAIN探针三个方向各256，旧cohort1024，全候选TRAIN过滤排名；探针生成seed=840000+seed，所有方法同材料。准备仅生成与模型有关的新探针mask，旧缓存不重建。

每seed六配置local/cumulative及各自恒定均值/固定随机打乱乘数控制；另只读复用080 ordinary/absolute终态。两seed配置和代码在任何本轮VALID读取前固定。各seed全部训练提交后评价该seed VALID，不根据seed29结果改seed43；对所有不同终态完整旧TRAIN排名。与同epoch/平均乘数控制比较附加效用，不把弱保护导致新学习恢复当创新。

本轮不是未见数据集、跨增量或原生近邻复现。三seed开发VALID已有暴露，两个seed此前还参与080/081。只确认固定新决策规则在不同初态/学习tape的行为，不声称统计显著或独立泛化。

成本计入普通影子、全部30分支、完整probe评分、校准来源和缓存复用。每seed资格自己的teacher，不能跨seed混分母。无TEST/硬超时/独立审计/SHA/归档。资产/data/lc/untzh/epoch_confirm85_20261014。
