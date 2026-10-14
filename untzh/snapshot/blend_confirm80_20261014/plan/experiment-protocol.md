# 固定双信号保护的跨种子确认080

FB-CKGE snapshot0→1，DistMult200，复用历史seed29/43初态（旧底座和新行初始化均随seed变化）。这些seed已有历史实验，本轮只是此前未用于077/078配方选优的轨迹，不是未见数据或新的独立数据集。VALID继续沿用1024+1024开发查询，不读TEST。旧E/R逐位冻结，新E Adam .007、20epoch、batch1024、10负例、L2 .0005。

固定077配方：前10epoch普通学习；后10epoch均匀抽32合格旧TRAIN答案，每答案128固定合法新候选；absolute与mass初态8batch RMS各校准rho .1、各自逐步限幅后各半混合。训练材料按seed一次生成缓存，所有臂共享同一学习tape/prefix10/保护抽样序列。mass旧teacher logZ过滤其他TRAIN答案、保留当前gold；新池排除TRAIN已知正确答案，抽样质量按合法池大小放大。

固定六臂：none、absolute、mass、blend、absolute_normmatched（本身轨迹上缩放到blend的保护norm）、absolute075。不根据新结果调配方、时点或追加权重。控制仅确认信号组合与强度解释，尚不包含新seed完整replay/MEOR复现，后续最近邻仍需补。

所有训练结束才读取VALID；每组均评价完整旧TRAIN，不按VALID筛选。保存raw及TRAIN-only filtered MRR/Hits1/Hits10、旧Top10固定资格掉榜、三个新方向对none保持比。每seed资格来自该seed旧teacher完整旧TRAIN排名，不能跨seed混用seed17资格。

主比较为blend相对absolute的学新—保旧取舍是否方向一致；同时报告相对none和全部控制，不用某一漂亮指标宣布成功。种子数2，不能凭符号一致宣布显著/跨增量泛化。成本含本次tape/teacher/cohort准备、共享prefix、校准、各臂训练、评价；历史底座训练费用另列为复用且非零。prepare→train→evaluate→summarize，无硬超时、独立CPU审计、SHA门槛或归档。

执行说明：代码按seed29→43逐个执行；每个seed内六臂全部提交后才读取该seed的VALID，而不是十二臂都提交后统一读取。seed43的六臂和配方在seed29评价前已固定，运行中没有根据seed29结果改动。原“所有训练结束”表述应按本实际范围理解，不宣称全轮统一延迟VALID。
