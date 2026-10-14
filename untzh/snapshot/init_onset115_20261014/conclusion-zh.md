# 115初始化比晚期保护带来更大的WN收益

原SSH11992自然exit0，PID891536退出，三新配置完成，程序17.130秒。只将112同一随机新行乘.25（std .12→.03），TRAIN tapes/旧空间/teacher全部复用；low普通prefix一次。两个保护参考分别与low ordinary或112 standard ordinary逐位一致。

|配置|新MRR|旧VALID MRR|旧VALID掉榜|
|---|---:|---:|---:|
|standard ordinary（112）|.209656079|.251263000|359|
|standard query_late（112）|.209820902|.251536534|351|
|lowinit ordinary|.220719092|.251950182|333|
|lowinit query_late|.220946310|.252121057|324|
|standard query_start0|.210022226|.251855316|340|

低尺度普通学习较标准普通新MRR提高约5.28%、旧VALID掉榜减少26，三个新方向均改善；这属于初始化带来的基底变化。lowinit query再相对自己的ordinary新MRR提高约.103%、少掉榜9，是小得多的保护附加收益。全部配置旧TRAIN掉榜0，不能宣称修复TRAIN遗忘。

低尺度初始扩展旧VALID掉榜394（原446），最终333，说明仍有大量初态竞争而不是完全安全初始化。start0比late旧VALID保护更好但用了220而非110保护步、查询抽样时间路径也不同，不能孤立归因于时点或当同累计强度控制。

三个配置都属于同WN/DistMult开发集、初始化sigma单一固定假设；不继续扫σ，不把它包装成原创或已确认端到端预警方法。116将把scale .25固定迁移FB三个配置，并在同lowinit ordinary上分别比较保护，避免贡献混算。无TEST/硬超时/独审/SHA/归档，goalactive。
