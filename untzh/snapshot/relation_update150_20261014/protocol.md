# 150：关系更新自由度与强固定控制，单父机制检验

上一turn147–149新实证progress，三个接口NO-GO；本轮不同假设：只训练新实体是否限制学新/保旧可达取舍。FB17 DistMult200，原1→2新实体子任务21566方向TRAIN，extra15648旧—旧只truth过滤；不把本轮说成完整FB任务。旧11258实体冻结，新1876实体及474×200关系统一可训练。共同121zero/prefix10和E-Adam历史、原087 learning tape/后220步；R原先无优化器历史，三个臂均fresh R-Adam，lr.007/原betas/eps。无父/前缀重训。

三臂relation_none/relation_answer_anchor/relation_full_replay都相同可训练权限、TRAIN、步数与学习/两保护/普通shadow计算包络。Anchor为当前R的原32cohort查询×完整新候选softplus，加已知旧TRAIN gold分数对teacher的均方锚定，固定两个项系数1，不扫描；当前关系变化时冻结R时代数空间不保证得分，本轮不投原queryspace。名称是明确扩展后的scoreAnchor，不能伪装原126投影方法。Replay原105 seed17的7040事实选择只读复用，每步32：正例softplus、合法全部13134尾/头候选负例mean、旧R相关regularizer完整计入，不把原来冻结旧参数时省去的常数项继续省略。仍oldE冻结，非作者原生全参数Replay。

每非零步保护梯度按joint newE/R norm为学习norm .1，零则0；none也计算两保护/shadow但不提交保护梯度。同rho非同累计强度，E与R各norm/旧分数漂移/费用另报。原126Anchor/146Replay/121普通只是冻结R另参数条件的机制参照，跨权限差异不归自适应方法贡献。

所有三终态保存后才读取VALID，并实算全旧TRAIN260498/旧VALID86832/新VALID2786 raw/filtered ranks、三方向、旧Top10掉榜/漂移。R已变不能套teacher旧rank+新超越数快评。纯TRAIN truth稀疏过滤索引每split缓存一次，各臂复用；固定旧R小样本零修改rank一致性检验仅确认新评分实现，不独立审计。TEST不读。

GO仅限下一风险机制研究：至少一个R保护臂相对同权限R-none新MRR及各方向>=.98、旧MRR不差/掉榜不多；相对冻结参照ordinary新MRR>=.98，并旧MRR>=冻结Anchor/掉榜<=冻结Anchor，至少一项新旧MRR较冻结Anchor更好。跨权限参照仅衡量是否出现值得研究的可达取舍，不宣称算法联合占优。否则NO-GO本关系释放条件，不改R学习率/Anchor系数/rho修补，不P3。即使GO仍无风险识别/决策独立收益，完整goalactive。

无TEST/进程硬超时/独审/SHA/归档/自动push；history79独立。新唯一目录relation_update150_20261014，观察超时只poll原句柄，不重启。缓存、父、tape与旧teacher费用非零。
