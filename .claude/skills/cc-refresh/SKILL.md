---
name: cc-refresh
description: 当包版本、API、模型仓库、许可或公司环境变化，或教程命令在目标机失败时刷新 cc 周包：冻结→审计→三列版本表→只改受影响段落→审查→校验→记录。用法 /cc-refresh 02 "nanoVLM main 变了"。
---

# /cc-refresh — 版本与事实刷新

## 触发

- PyTorch/CUDA/Transformers/LeRobot/TRL/PEFT 等 API 或模型仓库变化；
- 公司环境版本变化；
- 用户报告 `lab/` 命令在目标机失败且排除输入错误；
- 数据集许可或官方仓库状态变化。

## 步骤

1. **冻结**：记录失败命令、环境探针摘要（非敏感）、日期；不先改文件。
2. **审计**：派 `cc-source-auditor`，对照源计划、公司固定版本、当前官方文档。
3. **三列表**：`公司 PyTorch 2.1 路径 / 当前公开路径 / 不支持或未知`。
4. **定点修改**：只改受影响的 `lab/` 文件、`02_LAB_GUIDE.md` 段落、任务卡步骤；不批量升级全部命令到最新版。所有新命令标注等级与已验证版本。
5. **审查**：派 `cc-reliability-reviewer` 检查跨周引用、旧 API 混用、题答一致性。
6. **校验**：运行 `.claude/scripts/validate_cc_week.ps1 -Week NN`。
7. **记录**：`memory/cc/02_DECISIONS.md` 追加版本决策；`01_PROGRESS.md` 追加 `[cc]` 记录，写清变更原因、官方链接、核验日期、尚未实机验证的边界。

## 版本原则

- current 文档只说明当前事实，不保证公司环境可运行。
- 现场成功一次不等于跨机器兼容；记录驱动、CUDA runtime、PyTorch build、compute capability 和关键包版本。
- 不通过 `strict=False`、`ignore_mismatched_sizes` 或整套升级静默绕过。
