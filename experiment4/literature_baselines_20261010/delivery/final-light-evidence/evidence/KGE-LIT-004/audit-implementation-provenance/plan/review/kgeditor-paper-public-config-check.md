# KGEditor论文与公开脚本配置复核

已读取主来源 https://arxiv.org/html/2301.10405v8 ，完整HTML保存在papers/kgeditor-reference/arxiv2301.10405v8.html，SHA256 2db43209f3b02303998062234313ac2f1592067d3d2b74d330a37cbf95edd75e。v8日期2023-12-19。论文Evaluation/Settings明确采用PT-KGE为主设置、先初始化KG embeddings，再训练editor；与author HF模型README提供的PT-KGE edit对象一致。本文未找到epoch/lr/batch/显卡等具体训练预算声明，这些值来自已固定作者脚本，不能倒写成论文明确给定的所有配置。

原作者scripts/KGEditor/KGEditor_FB15k237_edit.sh与KGEditor_WN18RR_edit.sh均指定lr5e-4、单GPUddp、max_seq_length64、edit_num2、stable_batch_size8、20epochs、KL；FB扩展64/层10,11，WN扩展16/层11。本轮保留这些设置和seed0，只把workers32替换4并重定向资源路径/输出。数据HF只发布edit_train/edit_test/stable视图，stage的dev/test均指向edit_test，属于已登记暴露评估；不能根据论文“testing”用语就称未经选择的独立heldout。

Table2及作者README交叉核对KGEditor目标：FB Succ1/3=.866/.986、ER_roc=.999、RK3=.874、RK_roc=.635；WN .833/.844/.991/.956/.256。WB实测Succ1/3=.307229/.414458明确未匹配该表。论文Time .226/.232为编辑方法的Time列，未在该取得正文中确认详细timing contract；本轮4065.36s是含初始化、完整训练、恢复与导出的全进程，不能直接与该Time列计算加速比。

两个公开HF基础checkpoint都未包含额外dense_in_ex/dense_out_ex FFN权重，所以作者架构在编辑训练中初始化这些模块；HF官方blob/LFS来源检查通过，不等于独立认证其逐字节等于论文当年全部内部权重。论文没有给出可据本次差距确认的唯一原因。共享storage恢复能解释严格prefix重推理与native export差异，尚不能解释论文成功率的全部差距。不能通过该差距否定方法、把弱复现结果当SOTA胜利或追参缩小差距。

FB最终20epoch尚待终端与审计；其已完成epoch的较高monitor不是最终论文复现结果。本轮收尾后依用户要求先停，不启动新的训练/网格/改造协议。
