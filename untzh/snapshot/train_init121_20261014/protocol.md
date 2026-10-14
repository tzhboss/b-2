# 121 TRAIN邻居初始化与尺度/方向控制

固定FB父43/17/29，旧E/R冻结，source87/88/110 TRAIN负采样tape只读，116同随机std .03为比较底座。每父六新普通学习配置共18：zero、random_half (.015同张量)、neighbor_mean、query_mean、query_shuffle、query_flip。20epoch440步从各自初态学习，存prefix10含Adam供后续保护复用；本轮不加保护、不按VALID选配置。十八终态全部提交后才评价。

neighbor_mean取新TRAIN跨新旧事实的旧邻居单位向量均值；query_mean取对新行的DistMult正例梯度方向（旧邻居×关系）单位向量均值，角色键去重，逆向关系各自保留。均值方向归一化后逐新行乘原std .03随机向量的L2半径，排除单纯norm差；无跨旧约束或相消行回退同低随机向量。shuffle仅在有支持行间置换query方向、重新匹配原半径；flip反转受支持行query方向。zero/half是尺度控制。均属已有初始化思路，不声称原创。

新实体任务只学21566涉及新实体方向TRAIN，15648旧—旧新增事实只truth过滤，非完整snapshot2。完整旧TRAIN260498/旧VALID86832/新VALID2786，各父自己的teacher资格和已见TRAIN filtering；raw/filtered与三新方向实测。复用FastOld frozen缓存，仅新增候选分数，旧候选排名不重算。保存TRAIN扩展初态/终态，VALID初态/prefix/终态及新增/救回转移。初态收益与训练后收益分开，不称独立因果贡献。VALID为暴露开发数据，无TEST/硬超时/独审/SHA/归档。

目标是判断TRAIN邻居方向是否比公平低尺度随机更有价值，以及它是否只是更快学新而引入更大旧竞争。记录构造/训练/评价费，历史父模型/tape/teacher/cache非零；普通对照116复用非新独立重复。不据首seed结果修改后两个seed配方。
