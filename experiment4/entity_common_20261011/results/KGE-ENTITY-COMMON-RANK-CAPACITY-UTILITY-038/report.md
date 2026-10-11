# 固定容量豁免方案的完整收益与保护代价

旧、新VALID各固定1024条开发查询；全部旧TEST74428／新TEST12406查询参加共同评价。三个种子完整后汇总，非最终留出、等成本SOTA或新颖性认证。

| 方法 | 新VALID MRR | 旧VALID Top10掉榜% | 旧头／新答案MRR | 方向Hits10% | 固定收益结论 |
|---|---:|---:|---:|---:|---|
| CapacityFrozenFull | 0.129521 | 1.9086 | 0.024895 | 5.5788 | NO_GO_FOR_THIS_FIXED_COMBINATION |
| CapacityFrozenScoped | 0.130052 | 2.0105 | 0.025512 | 5.9275 | NO_GO_FOR_THIS_FIXED_COMBINATION |

旧TRAIN事实代价使用固定认证查询上的原Top10真答案；下面的原QP比较是额外披露，不添加或替换038收益门槛。通过相对源SGD的门槛不意味着优于原QP保护。

| 方法／seed | 原Top10 TRAIN事实数 | 源SGD掉榜数 | 原QP掉榜数 | 新方案掉榜数 | 相对原QP的MRR变化 |
|---|---:|---:|---:|---:|---:|
| CapacityFrozenFull／17 | 3210 | 113 | 2 | 73 | -0.00128731 |
| CapacityFrozenFull／29 | 3240 | 32 | 3 | 22 | -0.00044823 |
| CapacityFrozenFull／43 | 2820 | 21 | 0 | 14 | -0.00011455 |
| CapacityFrozenScoped／17 | 3210 | 113 | 2 | 80 | -0.00141567 |
| CapacityFrozenScoped／29 | 3240 | 32 | 3 | 23 | -0.00050248 |
| CapacityFrozenScoped／43 | 2820 | 21 | 0 | 15 | -0.00011778 |

豁免改变选定旧TRAIN事实与预测成员的保护合同；即使通过全部门槛，也不能称全部旧知识安全。JSON另列这些查询上的全部TRAIN事实及原来不在Top10的事实代价。原QP控制复用已有审计终态，未重新训练。

既有失败、原032负结论与037/038冻结门槛保持不变。汇总读取完整三seed终态；不从部分seed挑选胜利。源SGD、范围准备和认证成本计入增量耗时；未计时的全局split完整性检查仍明确排除。