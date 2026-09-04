# 01_PROGRESS.md - 项目进展日志

按时间顺序记录项目进展。每一次智能体调用都要留下一条，最新的在最上面。

## 记录格式

```md
## YYYY-MM-DD [codex|cc] - 标题

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

标题中的 `[codex]` / `[cc]` 标明处理方（2026-09-04 起启用；之前的记录均为 Codex）。两套系统只追加自己的记录，不改写对方的记录。

---

## Log

## 2026-09-05 [cc] - MiniMind 两周课程落地（M01 单机 5070 Ti / M02 八卡 V100）

`TYPE: TASK` · `DEPTH: DEEP` · `MODE: BUILD`

### 输入

- `00_INPUT.md` 需求第 2 条（AGENT: claude code）：根据 `input_info/minimind_5070ti_v100.md` 打造两周课程，第一周单机 5070 Ti，第二周八卡 V100。

### 动作

- 用 cc 多智能体流水线生成：来源审计 → 课程规划 → 理论 ∥ 实验 → 任务卡 → 口试 → 可靠性审查 → 校验。
- 新建轨道 `outputs/1_cc_coaching/track_minimind/`，两个周包各含周卡、Foundations、Lab Guide、6 张任务卡、口试、参考答案与可运行 `lab/`。
- 校验器扩展 `-Track` 与 `-IdPrefix` 参数以支持子轨道；修正三处误判（代码块内的链接样式、自处理 `--help` 的脚本、合法的 `except: pass`）。
- 联网核验 MiniMind 固定 commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b` 的目录结构、各训练脚本 argparse 默认值、数据集文件名与字节数、硬件与版本边界。

### 输出

- `outputs/1_cc_coaching/track_minimind/`：2 个周包，103 个文件，1.3 MB；课程文档 621 KB。
- M01：`42 passed`（CPU）；M02：`73 passed, 1 skipped`（跳过项需 CUDA）。
- 两周结构校验均 `PASS`，0 error 0 warning；口试 28/28 与 27/27 一一对应。

### 决策

- 本轨道是 14 周主线之前的热身，不替代主线 Week 01。
- M02 全部命令改用 fp16：MiniMind 默认 bfloat16 在 V100 上直接抛 RuntimeError，且 `train_grpo.py` 无 GradScaler，已在 lab 中补齐。
- 教学版专家并行为 `dist.all_to_all_single` 补了自定义 autograd Function，否则梯度在 dispatch 处静默断掉。
- 未在目标机执行的部分一律标 `估算` 并列入待验证清单，不升级为实测。

### 可靠性审查与修复

- 审查结论 `PASS-WITH-FIXES`：1 个 P0、8 个 P1、6 个 P2/P3，已全部修复并复验。
- P0（公司边界）：M02 Day 0 的证据字段 `topology_link_types` 会把互联链路类型带出，与 `CLAUDE.md` 第 2 节冲突。改为 `topology_link_types_count`（只报种类数）。这是 40 个证据字段中唯一不是抽象值的一项。
- P1（会直接失败的命令）：Day 0 第一条写入命令缺 `mkdir`；Day 5 缺 `PYTHONPATH` 导致 `ModuleNotFoundError`。
- P1（假因果）：Day 5 步骤 6 标题写"从最后一个分片 checkpoint 恢复"，但故障脚本不写 checkpoint，恢复脚本无条件重新保存。改为"独立跑一次 8→4 分片恢复回归"。
- P1（悬空引用）：周卡与 Foundations 声称复用 M01 的 `mm_probe/`，但全周零引用。改为说明两周 lab 相互独立。
- P1（参数名）：散文中 3 处下划线写法（`--num_experts`、`--sdpa_backend`）与实际的连字符不符。可执行任务卡里的 69 个参数经逐字核对零不匹配。
- P1（字段位置）：Day 4 要核对的 `expert_counts` 与 `aux_loss_unit_coef` 不在 CSV 列里（`extrasaction="ignore"` 静默丢弃），改为指向逐步 JSONL。
- P2：`bucket_cap_mb` 单位是 MiB 不是十进制 MB，推导过程改正（结论"约 11 个桶"不变，但原推导碰巧对）；补充单位与 dtype 约定，说明 FSDP 335 MB 与 DDP 447 MB 的 0.75 比值纯粹来自两侧精度不同，不是结构性结论；Day 2/Day 5 的双主要产物改为主次分明；Day 4 补回填证据步骤。
- 审查确认零命中的类别：虚构实测、硬件事实错误、危险命令、torch 2.1 API 纯度、禁用词、题答映射、状态表一致性。

### 纠错（相对来源文档与首版文档）

- 训练脚本在 `trainer/` 而非 `scripts/`；`train_pretrain.py` 默认 `max_seq_len=340` 而非 768；`train_grpo.py` 默认 `loss_type=cispo`；LoRA rank 固定 16 无命令行参数。
- 参数量：M02 首版每层漏算 `q_norm`+`k_norm` 共 192，全模型少 1,536。以实例化后 `sum(p.numel())` 为准改为 dense 63,912,192 / MoE 198,416,640，与 M01 一致。
- M01 理论文档中三处引用了不存在的命令行选项，已改为真实入口。

### 下一步

- 用户确认每周投入小时数与 5070 Ti 的 PyTorch 版本后，`/cc-day 01 0`（轨道内 Day 0 探针）开始。
- M02 首次在公司机器运行任何多卡脚本前，先 `MAX_STEPS=2` 试跑（本机多进程启动器不可用，这些命令行仅人工核对）。

## 2026-09-04 [cc] - 本机 Python 切换到 conda 环境

`TYPE: TASK` · `DEPTH: AUTO` · `MODE: REFINE`

### 输入

- `00_INPUT.md` 需求（AGENT: claude code）：以后使用 Python 就用 conda 环境。

### 动作

- 探针本机：Anaconda 23.10.0 位于 `D:\Software\Large\Anconda`；无 GPU；PATH 上的 `python` 为商店占位。用户指定环境 `ResearchAgentPy310`（Python 3.10，初始无 torch）。
- `CLAUDE.md` 新增 3.1 节“本机 Python 规则”；`AGENT` 字段接受 `claude`/`claude code` 作为 cc 别名。
- 新增 `.claude/scripts/cc_py.ps1` 统一入口（默认 `ResearchAgentPy310`，可用 `CC_CONDA_ENV` 切换）。
- 更新 `cc-lab-builder`、`cc-reliability-reviewer` 的本机执行要求；更新 `memory/cc/` profile、decisions、state。

### 输出

- `CLAUDE.md`、`.claude/scripts/cc_py.ps1`、`.claude/agents/cc-lab-builder.md`、`.claude/agents/cc-reliability-reviewer.md`、`memory/cc/*`。

### 决策

- 本机 conda 环境固定为用户指定的 `ResearchAgentPy310`。

### 下一步

- 同一输入块的第 2 条需求（MiniMind 两周课程）在下一条记录中处理。

## 2026-09-04 [cc] - Claude Code 陪跑教练系统建成

`TYPE: TASK` · `DEPTH: AUTO` · `MODE: REFINE`

### 输入

- 用户要求在同一仓库内新建一套与 Codex 完全独立的 Claude Code 多智能体陪跑教练系统；两套系统不共用 skill 与契约；`01_PROGRESS.md` 共享并按 `[codex]`/`[cc]` 标签区分；`outputs/` 分目录存放。
- 用户明确：暂不处理 `input_info/minimind_5070ti_v100.md`。

### 动作

- 新建 `CLAUDE.md`：cc 运行契约，含目录归属硬边界、颗粒度合同、训练门、A0–A4、8 个角色与 7 个 skill 的定义。
- 新建 `.claude/agents/` 8 个子智能体：source-auditor、curriculum-planner、theory-tutor、lab-builder、task-card-writer、examiner、reliability-reviewer、evidence-coach。
- 新建 `.claude/skills/` 7 个命令：`/cc-input`、`/cc-build-week`、`/cc-day`、`/cc-evidence`、`/cc-gate`、`/cc-refresh`、`/cc-status`。
- 新建 `.claude/templates/`（任务卡、周卡、实验合同、进度记录）、`.claude/schemas/evidence.schema.json`、`.claude/scripts/validate_cc_week.ps1`。
- 新建 `memory/cc/`（profile、state、decisions、evidence log）与 `outputs/1_cc_coaching/00_README.md`。
- 共享文件仅做标签级改动：`00_INPUT.md` 增加 `AGENT:` 路由字段；本文件增加处理方标签约定；`README.md` 增加“两套智能体系统”一节。

### 输出

- `CLAUDE.md`、`.claude/**`、`memory/cc/**`、`outputs/1_cc_coaching/00_README.md`。

### 决策

- cc 的交付单位是“可执行的一天”：任务卡 ≤1 页 ≤7 步、代码放 `lab/` 不放 Markdown、每卡以证据字段结束。
- 周包结构固定为周卡 + Foundations + Lab Guide + 6 张任务卡 + 口试 + 答案 + `lab/`。
- 尚无周包与学员证据；系统状态 `SYSTEM-READY / NO-WEEK-PACKAGE-YET`。

### 下一步

- 用户确认每周可投入小时数与 Week 01 主线环境后，运行 `/cc-build-week 01`，再 `/cc-day 01 0`。

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
