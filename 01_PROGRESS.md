# 01_PROGRESS.md - 项目进展日志

按时间顺序记录项目进展。每一次智能体调用都要留下一条，最新的在最上面。

## 记录格式

```md
## YYYY-MM-DD - 标题

`TYPE: TASK|REFINE` · `DEPTH: AUTO|DEEP|SCRIPT`

### 输入
简要描述已处理的输入。

### 动作
- ...

### 输出
- ...

### 决策
- ...

### 下一步
- ...
```

不是每条都必须写满五段，没有的段落删掉即可。

---

## Log

## 2026-09-03 - 14 周 LLM 训练教程与陪跑教练首版落地

`TYPE: TASK` · `DEPTH: AUTO` · `MODE: COACH`

### 输入

- 将 `input_info/` 中总体计划与 14 份周计划扩展为完整训练教程。
- 建设长期 LLM 训练陪跑教练、项目级 Codex agent 与 workflow，并替换原 FDE 项目契约。
- 补充要求：Foundations 与 Lab 必须是可从空目录照做的详细操作指南，明确数据集、模型、revision、下载、实现、训练、观测、排错和验收，不能只是 roadmap。
- 所有副作用严格限制在 `D:\zz\00_RealProjects\0_LLM_Training\`；`input_info/` 保持只读。

### 动作

- 将 `AGENTS.md` 重写为 LLM Training Learning OS 契约，建立 1 主代理 + 6 专职 agent 和 4 条 workflow。
- 建立详细手册发布标准、共享执行协议、profile/state/decision/source memory、evidence/run-manifest schema 和两个确定性 validator。
- 生成并扩写 14 周 × 4 文档；对 28 份 Foundations/Lab 做多轮交叉审查，修复数据泄漏、loss reduction、AMP 时钟、checkpoint/hash-before-load、resume、DDP/FSDP、评测 lineage 与 A/B 公平性问题。
- 联网核验 OpenAI agent 结构、V100 FP16、CUDA/Volta 边界、数据/模型 revision 与官方资料；纠正“V100 不支持 FP16”的输入错误。

### 输出

- `outputs/0_basic_training/`：14 个周目录、56 份周文档；28 份核心手册约 973 KB。
- `outputs/00_2026-09-03_教练系统实施规划.md`
- `outputs/01_2026-09-03_技术事实与兼容性核验.md`
- `outputs/02_2026-09-03_详细教程质量审查报告.md`
- `.codex/config.toml`、`.codex/agents/*.toml`、`workflows/`、`schemas/`、`scripts/`、`memory/`。

### 验收

- 课程 validator：56/56 PASS，0 error，0 warning。
- Markdown 示例：59 文件、338 fenced blocks；JSON 22、PowerShell 25、Bash 154，0 error，0 warning。
- 336 个口试问题与 336 个答案一一对应；本地链接 67/67；外部 URL 95/95；核心手册禁用占位符命中 0。
- `codex features list` 可解析项目配置；6 个 agent TOML 必需字段齐全。
- 当前机器无可用 Python/CUDA，`python`/`python3` 退出码均为 9009；因此文档门为 PASS，训练/pytest/GPU 执行门仍为 PENDING。

### 决策

- 14 周按 Gate 推进，不因日历到期跳过 correctness。
- 公司原始资产全部留内；只有获准的抽象非敏感文字可用于陪跑。
- PyTorch 2.1 是公司兼容线，不为教程强行升级；H100 不是默认依赖。
- 在真实 A0/A1 证据前不生成 mastery 分数。

### 下一步

- 从 Week 01 readiness 与 Day 0/Day 1 开始：先闭卷诊断，再运行环境探针、toy BPE 和单 batch；把真实结果按 evidence schema 记录。
