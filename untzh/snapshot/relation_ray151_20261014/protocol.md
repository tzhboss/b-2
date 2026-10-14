# 151：正关系温度的排序不变更新，单次结构检验

150 unrestricted R破坏旧排序且无联合收益，NO-GO该条件、不改R lr/保护系数。新假设为消除关系方向漂移，只学习正尺度a_r=exp(loga_r)；score=a_r*(E_h⊙R_teacher,r)·E_t。对同query所有候选乘相同正数，排序尺度可严格约掉，因此oldE冻结时旧候选间rank保留，新增候选竞争仍可能损伤旧rank。这个基本不变性/温度学习不当原创，参数化不同不是risk决策贡献。

FB17原121zero/prefix10/E-Adam/087学习tape复用，oldE/R方向冻结，新1876×200 E与474个log尺度统一可训练、共同fresh log-scale Adam/原.007参数。三臂ray_none/ray_answer_anchor/ray_full_replay同学习/两保护/ordinaryshadow包络，220步/rho.1。Anchor与Replay严格同150定义的score teacher/完整正负项，但所有得分将a放在dot之后，避免FP32关系坐标各自缩放误差干扰数学性质。没有R lr、a边界、初始化或微penalty搜索。

模型明确保存base R、relation_scale及canonical scoring说明，不冒称R未参与训练。评价时因正因子对所有候选相同可约掉：全部rank用base R和当前E；old查询的旧rank缓存因此重新合法，teacher+新超越数快评可复用，newVALID实算完整候选。核查所有scale有限且>0，固定64query已乘scale与约掉scale的rank一致性仅验证实现，不独审。

所有三个模型提交后统一完整旧TRAIN/VALID/newVALID2786 raw/filtered、三方向、Top10掉榜/漂移、触发/费用。原冻结R/150freeR是不同参数条件的机制参照，不把改变可训练自由度当公平算法收益。GO仅限后续风险机制诊断：至少一保护臂相对同权限ray_none新MRR及三方向>=.98、旧MRR/掉榜不差，并相对冻结ordinary newMRR>=.98、旧MRR>=冻结Anchor/掉榜<=546，至少一项新旧MRR较冻结Anchor更好。否则NO-GO这正尺度条件，不更换温度先验/初值/lr/rho修补，不P3。

数学性质保护的是旧候选间相对排序，不能当全部old Top10/TRUE、安全或未知gold保证。若有机械好处仍需风险→动作独立收益，完整goal不完成。无新父/前缀重训/VALID训练/TEST/硬超时/独审/SHA/归档/自动push，history79独立。
