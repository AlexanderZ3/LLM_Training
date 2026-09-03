# 第 13 周执行手册：FinExec 放大——1.5–1.7B FSDP、4B LoRA 与可选 GRPO

> 对应原 12 周主线第 11 周。标准投入 7.5 小时。  
> 主目标：观察模型放大后显存、吞吐、通信、稳定性和任务收益怎样变化。  
> GRPO 只是 stretch；SFT/evaluator/profile 未通过时禁止进入 RL。

每日时间盒：10 分钟写预算/止损，50–60 分钟完成唯一主配置，15–20 分钟做 evaluator 与资源汇总；第 2 小时才允许可选 GRPO。

## 1. 推荐优先级

1. 1.5–1.7B base 全参 FSDP：最适合巩固 sharding；
2. 4B base FP16 LoRA：依赖少于 QLoRA，容易在 32GB 单卡或多卡运行；
3. 4B QLoRA：只有公司已有经验证的 bitsandbytes/PEFT SM70 环境才做；
4. 0.6B GRPO 50–100 step：只验证 RLVR 管线；
5. 8B：只做纸面预算或最多 50-step FSDP smoke，不作正式目标。

不要一周同时完成所有项。核心交付是：

    一个放大配置完成可 resume 的 200–500 step SFT
    + 一份 0.6B→放大模型的系统/任务比较

## 2. 官方资源

| 资源 | 用途 | 地址 |
|---|---|---|
| PEFT | LoRA 配置和保存 | https://github.com/huggingface/peft |
| TRL | GRPO 参考实现 | https://github.com/huggingface/trl |
| Transformers | 模型/生成 | https://github.com/huggingface/transformers |
| PyTorch FSDP | 全参 sharding | https://docs.pytorch.org/docs/stable/fsdp.html |
| CS336 A5 | SFT/RL 推理训练思路 | https://cs336.stanford.edu/spring2025/ |

使用锁定版本，不从 main 直接安装。当前库若要求新 CUDA/FlashAttention/vLLM，不升级 V100 环境；回退最小纯 PyTorch/Transformers 路径。

## 3. 目录

    week13-scale/
      budgets/model_memory.md
      configs/
        fsdp_17b_sft.yaml
        lora_4b_sft.yaml
        grpo_06b_smoke.yaml
      src/
        inspect_trainable.py
        train_fsdp.py
        train_lora.py
        rewards.py
        train_grpo.py
        compare_scale.py
      tests/
        test_lora_targets.py
        test_reward_components.py
        test_reward_adversarial.py
      reports/
        scale_table.md
        decision.md

## 4. 每日安排

### 周一：模型/状态/激活预算与止损点

目标：启动作业前就知道什么可能 OOM。

为 0.6B、1.7B、4B、8B 写：

- FP16/FP32 parameter bytes；
- gradient；
- Adam moments；
- FSDP shard degree；
- LoRA trainable parameters；
- sequence/batch activation；
- logits tensor；
- all-gather buffer；
- generation KV cache；
- checkpoint size；
- 预估 step time。

LoRA 参数估算，对一个 Linear 权重 out×in：

    trainable LoRA params = rank × (in + out)

列出 target modules：

- q_proj、k_proj、v_proj、o_proj；
- up/down/gate MLP 是否加入；
- lm_head 是否训练；
- embeddings 是否训练。

不要用名字匹配后盲信。脚本打印：

- total/trainable params；
- 每个 trainable module；
- LoRA dtype；
- base params requires_grad；
- optimizer state 参数数。

预注册止损：

- reserved >30.5GB；
- 3 次连续非有限/scale 回退；
- 预计 200 step 超出本周资源预算；
- 依赖要求 Ampere+/CUDA 13；
- tokenizer/model 版本不兼容；
- 50-step 吞吐低到没有完成价值。

PASS：

- 四种规模预算；
- 选择一个核心放大配置；
- 有明确 fallback；
- 不启动 8B 长训。

### 周二：1.7B FSDP 或 4B LoRA 的 50-step smoke

#### 路径 A：1.7B 全参 FSDP

- 先 2 卡 NV2；
- 再四卡 0–3；
- 必要时 8 卡；
- transformer-block wrap；
- FP16 mixed precision；
- sequence 256–512；
- gradient checkpointing 依据显存；
- fixed global batch；
- 第 08 周 checkpoint。

#### 路径 B：4B FP16 LoRA

- 单卡 batch 1 起；
- gradient accumulation；
- base weights frozen；
- LoRA rank 8 或 16，只选一个主 rank；
- 不默认量化；
- 若单卡余量不足，四卡 DDP 仍复制 base，不节省单卡 base 显存；可结合 FSDP 但不增加不必要复杂性。

smoke 检查：

- forward/loss finite；
- 只有预期参数有 grad；
- optimizer state 规模合理；
- 50 step；
- checkpoint save/load；
- greedy eval 10 条；
- peak allocated/reserved；
- scale/skips；
- step p50。

LoRA merge 不作为训练 checkpoint 必需。adapter 与 base revision 必须同时记录。

PASS：

- 至少一个路径 50 step；
- trainable manifest 正确；
- peak safe；
- checkpoint 可重载；
- 不因 strict=false 隐藏 adapter missing keys。

### 周三：200–500 step 正式 SFT 与 scale 对照

目标：完成可比较放大实验。

固定与第 12 周 0.6B：

- train/dev IDs；
- prompt/schema；
- target mask；
- packing 选择；
- effective target tokens；
- generation config；
- evaluator；
- checkpoint selection；
- 两个核心 seed 中至少主 seed一致。

允许不同：

- 模型/tokenizer；
- micro-batch/accumulation；
- FSDP/LoRA；
- LR 合理缩放。

不要比较 raw train loss 数字后直接判模型规模，因为 tokenizer 可能不同。重点比较：

- valid JSON；
- valid program；
- execution accuracy；
- evidence F1；
- scale；
- target tokens/s；
- step time；
- peak memory；
- communication；
- trainable/full params；
- checkpoint size。

要求：

- 200–500 successful updates；
- 或相同 target tokens seen；
- 中途 resume 一次；
- best selection 使用 dev；
- test 只在最终运行。

PASS：

- 一个放大配置正式完成；
- resume；
- 实测与预算差异解释；
- 0.6B vs larger 的任务收益与系统成本同表。

### 周四：可选 0.6B GRPO/RLVR smoke

只有以下全部满足才做：

- 第 12 周 gold evaluator 100%；
- 0.6B SFT valid JSON ≥90% 或足够高；
- SFT checkpoint 稳定；
- 本周核心放大 SFT 已完成；
- 本地生成依赖可运行；
- 不需要外部 API/vLLM Ampere-only 路径。

reward 分项：

    r_format
    r_evidence
    r_program_valid
    r_execution
    r_answer
    r_scale

先不要合成一个只看 total 的黑盒。每条 sample 记录各分量。

推荐保守结构：

- model：0.6B SFT；
- group size：2–4；
- prompts：小固定 train subset；
- 50–100 optimizer steps；
- FP16；
- 生成长度上限；
- KL/reference control；
- deterministic executor；
- 不用 PnL；
- 不连接交易/外部工具。

adversarial tests：

- 空 JSON；
- 复制 prompt；
- 伪造 evidence ID；
- program 正确但 answer 不一致；
- answer 对但 program 无效；
- 极长响应；
- 除零/NaN；
- 只输出固定高频答案。

记录：

- 每项 reward mean/p50；
- total；
- KL；
- entropy；
- response length；
- valid JSON/program；
- execution/evidence；
- gradient norm；
- scale/skips；
- unique response rate。

reward hacking 信号：

- valid program 上升、evidence F1 明显下降；
- response 变长以刷局部 reward；
- 只学格式；
- exploit tolerance/rounding；
- answer 对但 program/证据失真；
- KL 爆炸或 entropy collapse。

本周不设“必须提升”。smoke 只判断管线是否可观察、reward 是否不被轻易利用。

### 周五：决策与能力复盘

结果表：

| model/method | full params | trainable | cards | peak GB | tok/s | valid prog | exec acc | evidence F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6B full | | | | | | | | |
| 1.7B FSDP or 4B LoRA | | | | | | | | |
| 0.6B GRPO optional | | | | | | | | |

回答：

1. 放大收益是否超过 evaluator 方差？
2. 成本增长主要来自 model states、activation、communication 还是 generation？
3. LoRA 限制是否出现在 trainable capacity？
4. FSDP 是否为内存而牺牲吞吐？
5. 更大模型是否只改善格式而非 execution？
6. 下一阶段应投数据/evaluator、模型规模还是 RL？

决策：

- CONTINUE-SCALE：任务提升有意义、成本可接受；
- CONTINUE-DATA：错误主要是 evidence/data；
- CONTINUE-SFT：格式/程序仍未稳定；
- RL-NOT-READY：reward/evaluator/生成基础不足；
- STOP-SCALE：收益不抵成本；
- INCONCLUSIVE。

## 5. 本周硬验收

| 项 | PASS |
|---|---|
| budget | 0.6B/1.7B/4B/8B 纸面预算与止损 |
| core run | 至少一个放大配置可 resume 200 step |
| trainable | FSDP/LoRA 参数与 optimizer state audit |
| stability | FP16 无非有限；skipped <1% |
| memory | 实测与预算差异解释 |
| comparison | 0.6B vs larger 同数据/evaluator/generation |
| GRPO | 可选；若做，reward 分项/KL/entropy/length 齐全 |
| safety | 无 PnL reward、交易连接、外部 API |
| conclusion | 任务收益与系统成本共同决定 |

## 6. 闭卷口试

1. 全参 FSDP 与 LoRA 各节省什么？
2. DDP 为什么不帮 4B base 分摊显存？
3. LoRA rank 如何影响参数量？
4. 不同 tokenizer 的 train loss 为什么不可直接比较？
5. effective target tokens 为什么比 steps 更公平？
6. GRPO group 内比较提供什么信号？
7. KL/entropy/response length 为什么都要看？
8. valid program 上升但 evidence F1 下降意味着什么？
9. 为什么 PnL 不是当前合适 reward？
10. 何时应该 STOP-SCALE 转而改数据/evaluator？
