# 135 纯TRAIN查询覆盖与抽样单位

134教师边界/风险路由未共同超过query-uniform absolute。135只改变参考query池，保持zero前缀、同absolute完整新候选penalty、jointAdam rho .1与普通-relative TRAIN查询空间投影。

两父FB17/WN ComplEx×三池=6新后段配置：all_query（完整旧TRAIN不同(h,r)），qualified_query（其中至少一个原TRAIN答案教师filteredrank≤10），query_mix（16来自原134cohort去重query，16来自完整旧TRAIN query且互异）。前两池每步32无重复query。固定rng1350017一次缓存选择序列，保持预算32和后段步数；不同池不能视为共享实际样本。

仅旧TRAIN及已有TRAIN教师rank参与采样资格，无VALID输入/gold选择；q一次缓存，合法新候选用TRAIN truths的稀疏索引，避免完整query×新候选dense mask。父/tape/query空间/teacher/zero-prefix/ordinary复用。全部6终态提交后统一full旧TRAIN/VALID、新VALID、128旧半群组评价；同时只读统计完整TRAIN池、原cohort池及实际抽到query对旧VALID query的覆盖，覆盖不是保护收益或独立证据。

zero普通/答案anchor和134query-cohort absolute缓存控制；extra监督133另口径。三方向raw/filtered、旧MRR/掉榜、实际参考数量、费用。shadow匹配自身ordinary且计费。无TEST/硬超时/独审/SHA/归档；基本query抽样/Replay适配/投影不当原创，整体goalactive。
