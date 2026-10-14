# untzh：2026-10-14 远程实验快照

这是原 `experiment4/` 工作线在服务器独立目录 `untzh` 中继续开展的实验记录。
本次快照截至2026-10-14 19:22（北京时间），从服务器只读复制99个目录中的1371份源码、协议、汇总和轻量证据（17097554字节，约17.1MB），原文件逐字节保留。
完整来源、大小、SHA-256和排除原因见 [上传清单](provenance/upload-manifest.json)。
这些哈希只验证快照与传输一致，不代表科学审计通过。没有重新训练或评价模型。

## 阅读入口

- [实验登记索引](experiments/index.json)：包括056–151、优化诊断、时序实验和历史核查。
- [早期063–078筛选报告](snapshot/method_search_summary_20261014/方法筛选结果.md)：仅反映该报告当时的实验范围。
- [第二增量088主线建议](snapshot/risk_increment88_20261014/主线建议.md)。
- [094累计差投影结论](snapshot/anchor_space94_20261014/conclusion-zh.md)。
- [101查询级分配协议](snapshot/query_budget101_20261014/protocol.md)及[汇总](snapshot/query_budget101_20261014/summary.json)。
- [102完整VALID扩展协议](snapshot/full_valid102_20261014/protocol.md)及[汇总](snapshot/full_valid102_20261014/summary.json)。
- [103累计差排名保护协议](snapshot/anchor_rank103_20261014/protocol.md)及[汇总](snapshot/anchor_rank103_20261014/summary.json)。
- [125生成前解析风险主线](snapshot/admission_forward125_20261014/主线更新.md)。
- [129接入保护主线](snapshot/admission_protect129_20261014/主线更新.md)。
- [133参考保护主线](snapshot/reference_space133_20261014/主线更新.md)。
- [134纯TRAIN教师边界协议](snapshot/teacher_boundary134_20261014/protocol.md)及[汇总](snapshot/teacher_boundary134_20261014/summary.json)。
- [135纯TRAIN查询覆盖协议](snapshot/query_coverage135_20261014/protocol.md)及[汇总](snapshot/query_coverage135_20261014/summary.json)。
- [143恢复顺序主线](snapshot/recovery_order143_20261014/主线更新.md)。
- [146预警保护主线](snapshot/forecast_protect146_20261014/主线更新.md)。
- [149动作预警主线](snapshot/action_forecast149_20261014/主线更新.md)。
- [150关系更新协议](snapshot/relation_update150_20261014/protocol.md)及[汇总](snapshot/relation_update150_20261014/summary.json)。
- [151正关系尺度协议](snapshot/relation_ray151_20261014/protocol.md)及[汇总](snapshot/relation_ray151_20261014/summary.json)。
- [历史79随机查询核对](snapshot/temporal_history79_20261014/随机历史查询核对进度.md)。
- [历史79错误竞争成分](snapshot/temporal_history79_20261014/错误排名成分核对.md)。
- [历史79第十轮错误漂移界限](snapshot/temporal_history79_20261014/false-rank-drift-bounds-round10.json)。
- [历史79第十九轮配对界限](snapshot/temporal_history79_20261014/paired-population-bounds-round19.json)。
- [历史79第二十二轮来源说明](snapshot/temporal_history79_20261014/round22-source-notes.md)。

## 文件与状态

`snapshot/<服务器目录>/` 是原始轻量记录，原有失败修订源码也一并保留。
`experiments/<目录名>/experiment.yaml` 是本快照的机器登记入口；其运行阶段直接引用原始
`status.json` 回执。JSON是合法YAML格式。原始回执是冻结的来源证据，不是另一个可手工维护的状态表。
没有足够信息的字段保持null，所有登记的科学有效性都保持未知。

执行完成不等于研究目标完成。102的原始回执为complete，其范围是86832条旧VALID及2786条
涉及新实体的VALID；原1024条样本复用，仅追加余集评分。这是既有开发数据的扩展，没有TEST。
历史79的中间执行回执也不能替代研究判定；目前总体“合理变化/真正错误”比例仍未知。
150和151在导出前已取得complete执行回执，其协议和汇总随本次同步保存；不能据此宣称方法已验证。
历史核查与CKGE方法结果分别保存。所有证据置于快照，未提升为正式 `results/`。

## 大文件与复现依赖

约28GB的远程目录包含模型、缓存、学习tape、逐条rank、原始数据和运行日志。
GitHub只保存轻量快照；排除文件仍在原服务器，对应路径和目录大小见上传清单。
失败日志的路径也在该清单中，原日志没有删除。GitHub不是模型与缓存的第二份备份。
逐批动作提案轨迹（proposal-records.json）留在服务器，保留汇总与分组诊断证据。

归档脚本保留了运行时的原始服务器路径。跨机器执行前必须配置数据、代码和模型位置；
不能假定克隆仓库即可重新运行全部实验。[依赖位置清单](provenance/dependency-locations.json)
记录源码中的静态服务器路径引用，包括依赖的既有 `experiment4` 代码和外部数据。
该清单不保证解析所有动态依赖。原 `experiment4/` 快照也不一定包含后来新增的全部依赖。

## 后续同步

[snapshot_tree.py](scripts/snapshot_tree.py) 是参数化的只读导出工具。
在服务器指定源目录、一个尚不存在的快照输出目录和压缩包路径：

```sh
python snapshot_tree.py --source-root "$SOURCE_ROOT" --output-root "$NEW_SNAPSHOT_DIR" --archive "$NEW_ARCHIVE_PATH"
```

下载压缩包后先验证传输哈希，再核对每个入库文件与上传清单中的哈希。
检查新增文件与排除清单后，提交到独立分支。后续同步需要再次执行此流程，目前没有设置自动同步。
