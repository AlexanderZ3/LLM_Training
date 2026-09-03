# Workflow 02：教程版本与技术事实刷新

## 何时触发

- PyTorch/CUDA/LeRobot/Transformers/TRL/PEFT API 或模型仓库发生变化；
- 公司环境版本改变；
- 教程命令运行失败且排除用户输入错误；
- 原论文、数据集许可或官方仓库状态变化。

## 刷新步骤

1. 冻结当前文档，记录失败命令、环境探针和日期。
2. `source_auditor` 对照源计划、公司固定版本文档、当前稳定版官方文档。
3. 建三列表：`公司 PyTorch 2.1 路径 / 当前公开路径 / 不支持或未知`。
4. 只修改受影响段落；不批量把全部命令升级到最新版。
5. 所有新命令标注：可执行、模板或伪代码；写清最低/已验证版本。
6. `reliability_reviewer` 检查跨周引用、旧 API、硬件 kernel 与题答一致性。
7. 运行 `scripts/validate_curriculum.ps1`（Windows）或 `python scripts/validate_curriculum.py`。
8. 在进度日志记录变更原因、官方链接、核验日期与尚未实机验证的边界。

## 版本原则

- 文档链接到 current 只说明当前事实，不保证公司环境可运行。
- 版本化链接用于公司兼容实现；不得把 current API 直接粘进 2.1 环境。
- 现场成功一次不等于跨机器兼容；记录驱动、CUDA runtime、PyTorch build、GPU compute capability 和关键包版本。
