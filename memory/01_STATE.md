# 当前状态

> 最近更新：2026-09-03  
> 状态：`COURSE-DOCUMENT-READY / EXECUTION-PENDING`

## 1. 当前结论

- LLM Training Learning OS 已完成首版落地：1 个主代理、6 个项目级专职 agent、4 条 workflow、2 个 schema 和 2 个确定性校验器。
- 14 周课程目录完整；每周均有 Foundations、Lab、闭卷口试和参考答案，共 56 份周文档。
- 28 份 Foundations/Lab 已按“从空目录到可检查结果”的标准扩写，不再是 roadmap。核心手册约 973 KB；口试 336 题与 336 个答案 ID 一一对应。
- 文档结构与 Markdown 示例静态验收通过；本机没有可用 Python/CUDA，所有训练、pytest、resume 和性能门仍是 `PENDING`，不能升级成实测 PASS。
- 尚无学员 A0/A1 基线、代码、日志或口试证据，因此没有 K/A/D/E 或 mastery 分数。

## 2. 当前发布门

| 项目 | 状态 | 证据/说明 |
| --- | --- | --- |
| 14 周 × 4 文件 | PASS | `scripts/validate_curriculum.ps1`：56/56，0 error，0 warning |
| Markdown 示例 | PASS | 59 个课程 Markdown、338 个 fenced blocks；JSON/PowerShell/Bash 解析无错误 |
| 问题—答案映射 | PASS | 336/336 |
| 本地 Markdown 链接 | PASS | 67/67 存在 |
| 外部来源链接 | PASS | 95/95 在 2026-09-03 可访问；以后使用时仍需复核 |
| Codex 项目配置 | PASS | 6 个 agent TOML 均含必需字段；`codex features list` 可解析项目配置 |
| Python/CUDA 实跑 | PENDING | 当前 Windows 机器的 `python`/`python3` 均为无效 Store shim，退出码 9009 |

## 3. 已确认边界

- 公司 8×V100、32 GB/卡、PyTorch 2.1 和拓扑均为用户自述待现场探针核验。
- V100 支持 FP16 Tensor Core mixed precision；通常不使用原生 BF16、TF32、FP8 或官方 FlashAttention-2 路径。
- 公司数据、代码、配置、日志、图像/视频、trace、checkpoint、拓扑与性能数字不进入本仓库；仅接收经批准的抽象非敏感文字。
- 个人 RTX 5070 Ti 16 GB 用于公开数据的小规模复现；H100 只有通过预算与必要性门后才使用。
- Week 10 的 `lerobot/smolvla_base` 权重许可在核验时没有明确声明，组织许可和运行时兼容门未通过前保持 `LICENSE-BLOCKED/ENV-BLOCKED`。

## 4. 当前学习位置

- 当前周：Week 01 readiness，尚未开始计分。
- 当前目标：闭卷解释 next-token shift、causal mask、train/valid 隔离、`loss≈ln(V)`、BPE 稳定 tie-break 和完整 resume 状态。
- 下一次执行：按 Week 01 Lab 的 Day 0 探针与 Day 1 toy tokenizer 测试开始；先 CPU/toy，再单 batch、短 smoke，最后才是正式数据与 GPU。

## 5. 下一次需要记录的真实证据

1. 执行环境探针和依赖版本，不包含公司敏感拓扑细节。
2. Week 01 readiness 闭卷答案及 AI 辅助等级。
3. toy BPE round-trip/save-load/hash 测试输出。
4. 首个单 batch forward/backward 的 shape、loss、grad finite 结果。
5. 若在公司执行，只在公司内保存原始产物；本仓库最多记录获准的抽象状态。
