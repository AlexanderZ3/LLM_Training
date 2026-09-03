# 第 14 周执行手册：Capstone——独立复现、1→8 卡、故障恢复与总报告

> 对应原 12 周主线第 12 周。标准投入 7.5 小时。  
> 本周不学习新框架，不做两个项目。只用一个载荷证明你已掌控全局。

每日时间盒：10 分钟核对预注册清单，50–60 分钟完成当天唯一主任务，15–20 分钟更新总报告；第 2 小时只用于复测或答辩准备。

## 1. Capstone 二选一

推荐：

    Mini-WAM action-only vs action+world

备选：

    FinExec 0.6B SFT vs short CPT→SFT

选择规则：

- 哪条线在前周已有可信 data/evaluator；
- 哪条线的系统 bug 已清零；
- 哪条线能在本周完成单卡 reference、四/八卡、resume、eval；
- 不选择“更炫”的，而选择“可完整闭环”的。

周一之后不得换题。

## 2. 一键入口标准

最终只允许一个顶层入口：

    python run_experiment.py train ...
    python run_experiment.py resume ...
    python run_experiment.py eval ...
    python run_experiment.py profile ...
    python run_experiment.py validate-data ...

或统一 shell/Makefile，但不得需要你手工改源码。

所有结果关联：

    run_id
    git commit
    config hash
    data revision/hash
    seed
    world size/rank mapping
    precision
    checkpoint
    eval command

## 3. 目录

    week14-capstone/
      README_OFFLINE.md
      run_experiment.py
      configs/reference.yaml
      configs/treatment.yaml
      manifests/
      tests/
      runs/
      reports/
        executive_summary.md
        correctness.md
        scaling.md
        profiling.md
        failure_recovery.md
        model_results.md
        capability_matrix.md
        next_8_weeks.md

## 4. 预注册验收问题

必须在运行前写：

1. treatment 相对 reference 改变什么；
2. 哪些变量固定；
3. primary metric；
4. secondary metrics；
5. 数值等价容差；
6. FP16 稳定门槛；
7. 单/多卡 scaling 定义；
8. 故障恢复门槛；
9. PASS/FAIL-MODEL/FAIL-SYSTEM/INCONCLUSIVE；
10. 最大 steps、卡时和磁盘预算。

不要在看结果后改变 primary metric。

## 5. 每日安排

### 周一：冻结代码、数据、配置与 clean-room dry run

目标：从空 output directory 复现。

任务：

1. git status/commit；
2. 数据 validator；
3. source/data/config hash；
4. dependency freeze；
5. clean output；
6. CPU/单卡 1-step dry run；
7. 20-step smoke；
8. 所有 unit/integration tests；
9. README_OFFLINE 只按文字即可操作；
10. 不依赖 Claude Code/Codex/网络服务。

clean-room 测试：

- 新 shell；
- 只阅读 README；
- 不改源码；
- 从 config 运行；
- 终端得到 PASS/FAIL summary；
- checkpoint/eval 路径自动明确。

PASS：

- 所有测试；
- source/data/config 可追踪；
- 一条命令 dry run；
- 无云 logger/upload；
- 未使用 test 调参。

### 周二：单卡数值 reference 与完整 checkpoint

目标：建立后面多卡的数学 reference。

配置：

- 1×GPU0；
- FP32 tiny fixed-batch 20-step reference；
- FP16 AMP 正式 100–300 step；
- 20 warmup + 100 timed profile；
- reference/treatment；
- 同 seed/data/order；
- 保存 step 50/last/best。

检查：

- overfit gate；
- loss/grad finite；
- scaler；
- model/optimizer/scheduler/scaler/RNG/sampler/EMA；
- eval 独立进程加载；
- terminal summary；
- peak allocated/reserved；
- primary metric。

重复一次短 reference，估计 run-to-run 方差。

PASS：

- 单卡 reference 可重复；
- resume 所需状态齐全；
- primary metric 有 baseline；
- fixed next-batch loss reference 保存；
- profile 主瓶颈已知。

### 周三：四卡→八卡等价与 scaling

目标：证明扩展没有改变任务。

顺序：

1. 四卡 0–3 fixed-global-batch correctness；
2. 四卡 4–7 symmetry smoke；
3. 八卡 20-step smoke；
4. 八卡 fixed-global-batch correctness；
5. 1/4/8 弱扩展或强扩展，按预注册；
6. profiler/NCCL 短 trace；
7. 不运行无上限长作业。

必须对比：

- sample IDs；
- global batch；
- successful optimizer steps；
- loss；
- gradient norm；
- updated params；
- primary eval；
- step p50/p95；
- throughput/E_N；
- communication ratio；
- peak memory。

门槛：

- 固定 global batch 前 20 step loss 偏差目标 <2%；
- 同一 DDP run 各 rank 参数一致；
- 300M+ 弱扩展 E4/E8 使用既定诊断阈值；
- 小模型效率低不自动判 NCCL 故障；
- 8 卡不值得长跑时仍可通过短 profile 完成结论。

### 周四：安全中断、恢复、评测与坏 checkpoint

目标：证明失败后不会静默产生错误结果。

流程：

1. 从完整 checkpoint 开始；
2. 运行到预设 step；
3. 按公司安全方式正常停止；
4. 验证 last complete marker；
5. resume；
6. 与连续 run 比较 fixed next batch；
7. 完成最终 eval；
8. 构造 incomplete checkpoint 标记/临时目录；
9. loader 必须拒绝；
10. 重复/漏样检查。

如果 FSDP：

- 每 shard 完整；
- manifest；
- world-size；
- strategy/wrap policy；
- optimizer reshard 约束；
- rank0/full state 不常驻导致 OOM。

如果 DDP：

- rank0 保存；
- barrier；
- optimizer/scaler；
- sampler epoch/offset；
- all ranks resume step 一致。

PASS：

- next fixed-batch loss 相对偏差 <1%；
- LR/scaler/RNG/sampler/EMA 连续；
- incomplete checkpoint 被拒；
- final eval 与 checkpoint 唯一关联；
- failure recovery 文档完整。

### 周五：6–10 页总报告、闭卷答辩与下一阶段

报告结构：

1. 一页 executive summary；
2. 问题与假设；
3. 数据/任务合同；
4. 模型/loss；
5. FP16 数值；
6. 单卡 profiling/memory；
7. DDP/FSDP；
8. 1/4/8 scaling；
9. checkpoint/recovery；
10. treatment vs reference；
11. failure taxonomy；
12. limitations/next step。

每个结果都带：

- config；
- data hash；
- seed；
- checkpoint；
- eval；
- status 标签。

能力矩阵自评 0–3：

| 能力 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| tokenizer/data | 不懂 | 会调用 | 会改 | 能独立实现/验证 |
| Transformer/VLM/policy | 不懂 | 会跑 | 会改 | 能从 shape 推导 |
| loss/evaluator | 不懂 | 会用 | 会调 | 能建 oracle/消融 |
| FP16 | 不懂 | 会开 | 会排错 | 能定位数值模块 |
| checkpoint | 只存权重 | 可恢复模型 | 恢复 optim | 完整 RNG/sampler/scaler |
| profiling | 看利用率 | 会 profiler | 能归因 | 能验证优化 |
| DDP | 会 torchrun | 会训练 | 能等价 | 能拓扑/效率诊断 |
| FSDP | 会配置 | 能跑 | 能 checkpoint | 能设计 shard/wrap |
| eval/statistics | 看 loss | 有 metric | 有对照 | 有 seed/OOD/错误桶 |

闭卷答辩 30–45 分钟：

- 从 raw sample 讲到 optimizer update；
- 从物理 GPU 讲到 rank/collective；
- 从 OOM 讲到状态/activation；
- 从 checkpoint 讲到 resume；
- 从 auxiliary loss 讲到因果消融；
- 从生成文本讲到 deterministic evaluator。

## 6. 最终硬验收

| 项 | PASS |
|---|---|
| reproducibility | clean output 一键 train/resume/eval/profile |
| lineage | config/code/data/seed/checkpoint/eval 全关联 |
| data | validator/split/mask/normalization |
| numeric | FP16 无非有限；scaler/grad 可解释 |
| single→multi | 单卡 reference、四卡、八卡 correctness |
| scaling | 吞吐/效率/通信有证据 |
| profile | 主瓶颈归为 compute/communication/data/memory/sync |
| recovery | 完整状态恢复；坏 checkpoint 拒绝 |
| model result | reference/treatment 公平对照 |
| conclusion | FAIL-MODEL 与 FAIL-SYSTEM 明确 |
| boundary | 公司数据、日志、checkpoint、指标均不外传 |

## 7. 14 周之后的“从零训练”项目阶梯

以下按学习价值排序，不要求同时做。

### A. 继续补基础

| 项目 | 你应完成什么 | V100 用法 |
|---|---|---|
| Karpathy Neural Networks: Zero to Hero | micrograd、MLP、GPT lecture；闭卷重写 | CPU/单卡 |
| minBPE | byte BPE、regex tokenizer、tests | CPU |
| Raschka LLMs from Scratch | 逐章实现 GPT、pretrain、finetune | 单卡小模型 |
| CS336 Assignment 1 | tokenizer/model/optimizer/minimal LM 全部自己实现 | 单卡，小数据 |

地址：

- https://github.com/karpathy/nn-zero-to-hero
- https://github.com/karpathy/minbpe
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/stanford-cs336/assignment1-basics

### B. 训练系统

| 项目 | 学习重点 | V100 适配 |
|---|---|---|
| CS336 Assignment 2 | benchmark、memory、distributed | 做 profiling/distributed；跳过 FA2 Triton 实机目标 |
| nanoGPT | 约 300 行 model/train，理解完整 pretrain | 训练小 GPT；仓库已较旧，作为对照 |
| llm.c | raw C/CUDA 训练路径、kernel/数值 | 先 CPU/reference；确认 SM70 build 后再小测 |
| Picotron | DP/TP/PP/CP 的教学实现 | 关闭 fused/FA2；先 DP/TP4 阅读与小测 |

地址：

- https://github.com/stanford-cs336/assignment2-systems
- https://github.com/karpathy/nanoGPT
- https://github.com/karpathy/llm.c
- https://github.com/huggingface/picotron

### C. 从教学到生产架构

| 项目 | 学习重点 | 建议 |
|---|---|---|
| nanochat | tokenizer→pretrain→SFT→eval→chat 全栈 | 当前参考面向 8×H100；只移植小配置/读架构 |
| Nanotron | 真实预训练与 3D parallelism | 先读配置、parallel context、checkpoint；不装 FA2 |
| LitGPT | 透明模型实现、pretrain/finetune recipes | 固定兼容版本、小模型 |
| OLMo-core | 公开大模型训练构件 | 读 config/train/checkpoint/tests；现代 fused kernels 可选 |
| TorchTitan | PyTorch-native 多维并行与分布式 checkpoint | 最新 main/nightly 不作为 V100 运行目标，做代码阅读 |

地址：

- https://github.com/karpathy/nanochat
- https://github.com/huggingface/nanotron
- https://github.com/Lightning-AI/litgpt
- https://github.com/allenai/OLMo-core
- https://github.com/pytorch/torchtitan

### D. Stanford CS336 完整路线

官方课程覆盖：

1. A1 Basics：tokenizer、architecture、optimizer、minimal LM；
2. A2 Systems：profile、benchmark、memory-efficient distributed；
3. A3 Scaling：模型/数据/算力缩放；
4. A4 Data：Common Crawl filtering/dedup；
5. A5 Alignment/Reasoning RL：SFT、RL，可选 DPO。

课程地址：https://cs336.stanford.edu/spring2025/

建议把它作为 14 周后的 12–20 周长期项目，而不是塞进当前每天 1–2 小时。A2 中自己实现 FlashAttention-2 的硬件目标不适合 V100；可以完成数学/reference/tests，并把 Triton 性能部分留给 Ampere/Hopper 机器。

## 8. 推荐后续 8 周

| 周 | 项目 |
|---:|---|
| 1–2 | 完整做 CS336 A1，不看答案 |
| 3 | 用你的 TinyStories 数据训练 50M/100M/200M 三点 |
| 4 | CS336 A2 profiling/memory，跳过 V100 不支持 kernel |
| 5 | Picotron：从 collective 构建 DP，再读 TP4 |
| 6 | llm.c：CPU reference 与一个 CUDA kernel |
| 7 | nanochat/Nanotron/TorchTitan 三套架构对照 |
| 8 | 从空目录重建一个 100M LM 的数据→训练→eval→resume→DDP |

评价标准仍然不是“clone 了多少仓库”，而是：

- 你能否先独立实现；
- 是否有 oracle 和 tests；
- 是否能做固定 batch 等价；
- 是否能解释显存/吞吐；
- 是否能恢复；
- 是否能在没有 AI 编程工具时重建。

## 9. 最终闭卷口试

1. 从原始文本到 LM optimizer update 的每一步是什么？
2. 图像 token 怎样进入 decoder-only VLM？
3. flow matching target 与 sampler 怎样对应？
4. 数据 lag/split/mask 如何制造伪结果？
5. FP16 的 overflow/underflow 如何观测？
6. DDP 与 FSDP 的 collective 分别是什么？
7. hybrid shard 如何映射这台 V100 拓扑？
8. profiler 怎样区分 data、compute、communication？
9. world loss 的 shuffled-action 消融为什么必要？
10. 金融 program evaluator 怎样防止模型“答案对、过程错”？
11. resume 为什么必须包含 scaler/RNG/sampler？
12. 你现在能从空目录独立重建哪些组件，还有哪些依赖框架？
