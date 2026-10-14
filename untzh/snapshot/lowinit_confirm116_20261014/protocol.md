# 116 FB三配置固定低尺度初始化确认

115在WN低尺度普通学习新MRR .220719 vs原.209656，说明基底初始化是大部分收益来源。固定同一scale .25（std .12→.03），在43/17/29父配置各比较none/simpleabsolute/answeranchor/queryuniform，十二新配置。所有初始化只乘原已有随机新行，不重抽；每seed低init普通prefix10一次；其后220步。旧E/R、negative tape、cohort/query sampler/teacher/SVD/TRAIN truths均与110/087/088只读复用。

simpleabs8batch RMS系数只在低init prefix重新计算一次，.1 cap不变；fullabsolute为完整1876合法新候选、rho .1、joint Adam、在线普通影子累计差投影。query32互异(h,r)，RNG1010029；answer沿原selections。保护必须相对对应lowinit ordinary比较，不能把初始化收益算保护贡献。标准初值none/absolute/answer/query完整VALID控制只读复用102/108/110。

全部十二模型提交后统一完整旧TRAIN260498/旧VALID86832/new实体VALID2786、raw及TRAIN-only filtered/三方向MRR；旧parent分母不变，低init初始扩展oldVALID附加竞争单算。普通影子终态与本seed lowinit ordinary严格相同。17/29用107cache，43用110，旧E/R逐位冻结；不重复原完整旧候选rank。初值策略是已暴露开发数据上的固定假设，不称新颖性/未见泛化/严格多seed同初态因果证明。

无TEST/硬超时/独审/SHA/归档。若与WN收益不一致，保留条件结果，不扫更多σ。风险路由价值仍单独未证。
