# 122 WN两模型TRAIN邻居初始化迁移

沿用121六个固定普通学习初值、未按121结果改变方法：zero/random_half(.015)/neighbor_mean/query_mean/query_shuffle/query_flip。DistMult200与ComplEx100(200实参数)各六，共12新配置；112/118父模型只读复用不重训，112增量负采样tape同一份，std .03底座分别取115/118。每行mean/shuffle/flip与本父低随机向量norm匹配，unsupported回退原低随机；zero/half尺度控制。每配置普通20epoch220步，prefix110及optimizer缓存。

query_mean从既有query-space rows/vectors缓存构造单位方向均值，不重建SVD；ComplEx向量由118真实头/尾梯度公式区分，DistMult为neighbor×relation。neighbor_mean仅旧邻居单位向量均值、按对应角色键去重。固定seed置换支持行方向并逐行半径匹配；不额外抽负例。不称邻居初始化原创。

全部12终态提交后统一评价完整旧TRAIN66960/旧VALID22320/新VALID3720及三方向；TRAIN初态/终态，VALID初态/prefix/终态。各模型自己teacher/gold及TRAINtruth过滤、raw/filtered并存，不把不同父质量当独立初始化因果效应。现有115/118低尺度ordinary和112/118标准ordinary只读复用，非新独立重复。无VALID选父/早停/规则阈值选择，无TEST/硬超时/独审/SHA/归档；VALID已暴露开发。
