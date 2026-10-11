# 037完整更新后的终态审计进度

六个任务均完成3753步冻结旧E/R的新实体更新及共同TEST评价。新通过的
seed43全保护方案复算3753步和67993组终态保护查询，solver审计无失败；
推断审计的720512个候选评分检查及rank fixture也通过。seed29限定保护
随后通过完整审计。此记录属于运行中阶段证据，不是三seed收益结论。

当前共同表已合格种子：

| 方法 | 已审计seed | 完整三seed |
|---|---|---|
| CapacityFrozenFull | 17、43 | 否 |
| CapacityFrozenScoped | 17、29 | 否 |

表内已审计任务数为115/123。刷新脚本逐对象核对39行非容量方法结果
不变，原效用结果及037/038冻结输入hash也不变。期间第四个审计自然
完成，因此输出文件capacity-audited-table-third-job.json记录的是实际
四个已合格任务；其名字标识刷新发起时的第三任务事件，不覆盖先前
任何结果。所有部分seed平均值不能与完整三seed控制据此宣布胜利。

剩余seed43限定保护及seed29全保护的wrapper分别为796430／796611；
现场确认CPU auditor子进程796434／796615存活，并已各复算49152组，
约为各自67985／67932组完整保护查询的72%。句柄及比例是现场快照，
后续状态必须重新读取，不能据历史PID重启。

038 coordinator783381确认存活，仍等待整个037终态。只有完整037/038
三seed审计、原门槛复算以及豁免旧TRAIN事实代价齐全后，才执行
scripts/report_rank_capacity.py的终态披露、最终共同表刷新与方法去留。
seed17参数几何和部分seed成绩不能替代此决定，也不触发参数搜索。

已执行的表刷新命令：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
../literature_baselines_20261010/environment/downloads/bin/python \
scripts/refresh_capacity_audited_table.py --root . \
--output results/capacity-audited-table-third-job.json
```

该脚本将输出路径的已存在检查放在任何表变更之前；后续刷新使用
新的明确输出路径。没有原地更改已注册worker、数学、预算、seed、
更新顺序或接受门槛。科学目标仍未完成。
