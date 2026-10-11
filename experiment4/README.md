# experiment4：知识图谱编辑实验快照

来源为远程 30966 的 `/data/lc/tzh/experiment4`，快照日期为 2026-10-11。
这是代码、配置、实验登记及轻量证据的归档。各项状态以相应的
`experiments/<ID>/experiment.yaml` 为准；归档不代表实验完成、审计通过或方法胜出。
历史失败、修订版本及作者协议与本地适配的区别均保留。

## 阅读入口

- [文献基线复现报告](literature_baselines_20261010/report.md)
- [文献基线使用边界](literature_baselines_20261010/plan/baseline-use-disposition.md)
- [Rank 方法对照](rank_comparison_20261010/report.md)
- [统一评价框架](rank_framework_20261011/report.md)
- [共同新增实体任务协议](entity_common_20261011/plan/experiment-protocol.md)
- [共同任务现有交接记录](entity_common_20261011/plan/chat-handoff-20261011.md)
- [共同任务对照表](entity_common_20261011/tables/common-comparison-v5/)
- [早期远程实验说明](README_第24轮远程实验.md)

## 保存范围与复现依赖

保存自有源码、脚本、配置、协议、登记、失败记录、审计摘要、表格和图。
原始数据、模型权重、NPZ、缓存、安装环境、日志、嵌套 Git 历史和压缩包留在服务器。
单个超过 1 MiB 的文件不纳入本次上传；逐步运行轨迹也不纳入。
完整排除路径及原因见 `provenance/upload-manifest.json`。
已有审计文件中的原始哈希与绝对路径保持原样，可能引用未上传的服务器产物；
本次上传文件的实际 SHA-256 则见 `provenance/upload-manifest.json`。

第三方作者仓库不重复整体上传。`provenance/upstream-repositories.json` 记录原 URL、
固定提交及本地补丁/新增代码；补丁在 `literature_baselines_20261010/upstream/`。
复现时先克隆所列仓库并检出固定提交，再根据清单应用补丁、复制新增代码，
按已有下载脚本和协议准备数据、权重及依赖。
环境和下载来源的轻量回执见 `provenance/environment-receipts/`。
没有重新运行训练或改写科学配置；脚本中服务器路径需要按实际环境配置。

仓库主分支的其他项目文件保留，本次实验集中位于 `experiment4/`。
