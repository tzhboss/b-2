# 117 FB三父状态两初值阶段配对（无训练）

固定43/17/29，复用standard及116 low initial/prefix10/ordinary终态，原父teacher/gold/truth cache不变。仅计算初始扩展与prefix10的1876新候选分数，不重算11258旧候选神经rank；普通epoch20 rank只读。每个split按本父原Top10资格拆候选加入→前10→后10的新掉榜/救回/persistent、raw/filtered MRR，保存阶段rank用于后续预警评价。

旧TRAIN260498与旧VALID86832均做，不把两者风险轨迹混用；已见TRAIN truths过滤。17/29复用107，43复用110。不得把对同一路径的阶段观测称独立干预的因果比例。无新训练/负采样/TEST/硬超时/独审/SHA/归档；开发VALID已暴露。
