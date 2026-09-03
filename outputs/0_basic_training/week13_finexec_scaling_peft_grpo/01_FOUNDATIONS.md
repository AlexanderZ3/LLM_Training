# Week 13 基础篇：FinExec 放大、FSDP、LoRA 与 GRPO 安全边界

> 状态：教学规格，未在用户机器运行；所有容量、显存、吞吐和指标均需实机验证。  
> 核验日期：2026-09-03。8×V100 32GB、拓扑和公司 PyTorch 2.1 是用户自述待核验。

## 1. 本周问题与唯一核心交付

本周不是模型展览。只需完成一条主路径：把 Week 12 已通过 evaluator/mask/生成门的 FinExec 数据，迁移到 `Qwen2.5-1.5B` 全参 FSDP，完成 50-step smoke、200-step 有界 SFT、同 world-size resume 和独立 dev eval；然后把它与 0.5B 的任务收益、内存、吞吐、通信和 checkpoint 成本放在同一张表。

两条 stretch 必须排在主路径之后：

- 约 4B LoRA：只有锁定模型/API/显存 probe 通过才做；
- 0.5B GRPO：只有 Week 12 gold evaluator、SFT 生成和 reward adversarial tests 全过才做。

最终决策只能是 `CONTINUE-SCALE / CONTINUE-DATA / CONTINUE-SFT / RL-NOT-READY / STOP-SCALE / INCONCLUSIVE`。GRPO smoke 不以“指标必须提升”为验收，也不使用 PnL、交易结果或外部服务。

## 2. 环境与跨周输入

本周复用 Week 12 的以下冻结对象：`data/processed/{train,dev,test}.jsonl`、`manifest.json`、`evaluator.py`、`train.py`、`evaluate.py`、`models/Qwen2.5-0.5B`。开始前重新跑 Week 12 unit、gold manifest 与 20-step smoke。若这些对象缺失，先按 Week 12 Lab 重建；不能假定“应该还在”。

| 环境 | 主要职责 | 精度/并行 | 禁止项 |
|---|---|---|---|
| 公司 8×V100 | 2/4/8 卡 1.5B FSDP；4B LoRA 仅 gated smoke | FP16 mixed precision；PyTorch 2.1 FSDP1 | BF16、FA2、CUDA/torch 擅自升级、结果外传 |
| 个人 5070 Ti | 0.5B LoRA/reward 单测；1.5B 极短 smoke 视显存 | 探针后决定 | 不复制公司 checkpoint/指标/数据副本 |
| 可选 H100 | 独立 BF16 或大模型对照 | 单独 config/manifest | 不与 V100 结果合并 |

V100 可高效 FP16，通常无原生 BF16。FlashAttention-2 官方 CUDA kernel 不支持 SM70；所有模型强制 eager attention。FSDP API 以 PyTorch 2.1 为准，不把新版 FSDP2/DeviceMesh 示例倒灌进旧环境。

## 3. 固定模型、代码和许可

| 对象 | 官方来源 | 冻结 revision/tag | 许可与角色 |
|---|---|---|---|
| 1.5B 主模型 | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | `8faed761d45a263340a0528343f099c05c9a4323` | Apache-2.0；全参 FSDP 主路径 |
| 0.5B 对照/GRPO | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | `060db6499f32faf8b98477b0a26969ef7d8b9987` | Apache-2.0；复用 Week 12 |
| 4B stretch | [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) | `906bfd4b4dc7f14ee4320094d8b41684abff8539` | Apache-2.0；只有兼容审计通过才下载/LoRA |
| Transformers 主栈 | [transformers](https://github.com/huggingface/transformers) | `v4.46.3` | Apache-2.0；1.5B 主路径 |
| PEFT | [peft](https://github.com/huggingface/peft) | `v0.13.2` (`34d479632d63a7e29c4b75202c9eef94335ceb14`) | Apache-2.0；LoRA stretch |
| GRPO 参考 | [trl](https://github.com/huggingface/trl) | `v0.15.2` (`77f9e82ff963b0a82581e78a21554fc90f96b843`) | Apache-2.0；独立 compatibility gate，不污染主环境 |
| PyTorch FSDP | [PyTorch 2.1 FSDP](https://pytorch.org/docs/2.1/fsdp.html) | 2.1 文档 | BSD-style；主路径 API |

Qwen3-4B 通常需要比主栈更新的 Transformers；因此它不是“照命令必跑”的核心依赖。若环境审计显示 `model_type` 未注册，正确结论是 `ENV-BLOCKED`，不是升级公司 torch/CUDA。主交付仍由 1.5B FSDP 完成。

## 4. 规模与显存预算

### 4.1 全参 AdamW 粗估

设参数量 `P`。常见 mixed-precision AdamW 近似包含 FP16 参数 `2P`、FP16 梯度 `2P`、FP32 master/一二阶矩 `12P`，合计约 `16P bytes`，实现可能在 12–16P。再加 activation、logits、all-gather、通信 bucket 和 allocator。

| 模型 | 16P 状态粗估 | 4-way full shard 理想下限 | 8-way full shard理想下限 |
|---|---:|---:|---:|
| 0.5B | 8GB | 2GB/卡 | 1GB/卡 |
| 1.5B | 24GB | 6GB/卡 | 3GB/卡 |
| 4B | 64GB | 16GB/卡 | 8GB/卡 |
| 8B | 128GB | 32GB/卡 | 16GB/卡 |

这些不是峰值；FSDP forward/backward 会 all-gather 当前 wrap unit，过粗 wrap 可能瞬时聚合整个模型。sequence/activation 和 `[B,S,V]` logits 仍可能主导。预算只用于排除配置，最终以 `max_memory_allocated/reserved` 实测。

### 4.2 LoRA 参数

对线性层 `W∈R^{out×in}`，LoRA 训练矩阵 `A∈R^{r×in}`、`B∈R^{out×r}`：

\[
P_{LoRA}=r(in+out).
\]

例如 `in=out=2560,r=16`，单层单 projection 为 `16×5120=81,920` 参数。LoRA 大幅降低梯度/optimizer 状态，但 FP16 base 权重和 activation 仍驻留；DDP 会在每卡复制 base，不能靠多卡解决单卡 base OOM。QLoRA 只有已有经验证的 bitsandbytes SM70 wheel 时才做，本手册不将其列为主路径。

## 5. FSDP1 的对象、wrap 与通信

PyTorch 2.1 `FullyShardedDataParallel` 的 full shard 对参数、梯度和 optimizer state 分片。典型一个 transformer block：

```text
forward: all-gather block params → compute → reshard
backward: all-gather params → compute grads → reduce-scatter grads → reshard
optimizer: rank 只更新本地 shard
```

wrap 太细：collective 太多，latency 主导；wrap 太粗：all-gather 峰值过大。Qwen2.5 主路径以 `Qwen2DecoderLayer` 自动 wrap；必须通过 `named_modules()` 验证匹配数量等于层数，而不是只相信字符串。

FSDP mixed precision 分三类 dtype：parameter/reduce/buffer。本周设 FP16 parameter/reduce、FP32 loss reduction；模型仍从 FP32 load，再由 FSDP 管理 mixed precision。不要先 `model.half()`。

梯度裁剪必须调用 FSDP-aware 的 `model.clip_grad_norm_` 或由 Trainer 正确处理；不能对本地 shard 用普通全局范数并宣称全局 clip。

## 6. checkpoint：resume 与 eval 是两类产物

1. **sharded training checkpoint**：model/optimizer shards + trainer/scheduler/scaler/RNG + 冻结 lineage，适合同 world-size resume；
2. **full inference export**：rank0 CPU-offloaded full state，适合独立 eval，但可能占主机 RAM/磁盘；不是 optimizer resume。

PyTorch 2.1 的上下文形式：

```python
from torch.distributed.fsdp import FSDP, StateDictType, FullStateDictConfig
cfg=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    full_state=model.state_dict()
```

这是原理片段，不是后续命令依赖；Lab 按 Transformers 4.46.3 官方建议用 `SHARDED_STATE_DICT` 保存中间 resume checkpoint，训练结束后才把 FSDP plugin 切到 `FULL_STATE_DICT` 做独立评测导出，并现场审计两类文件。world-size 8→4 恢复不是本周默认承诺；若使用 sharded optimizer state，必须同 world-size，或专门验证 reshard API。

发布 checkpoint 的不变量：每次 Trainer 完成 save 时所有 rank 先 barrier，由 rank0 原子写该 checkpoint 的 `RESUME_CONTRACT.json`，再 barrier → 审计 model/optimizer/scheduler/FP16 scaler/各 rank RNG/shard/trainer state/lineage 与非空文件 → 计算逐文件 SHA → 原子写 completion marker → 立即按 marker 复验。这样后续作业即使崩溃，较早的完整 savepoint 仍可审计；marker 必须最后出现，临时、缺 shard、缺 marker、hash 漂移或 lineage 不匹配的目录必须在反序列化前拒绝。

恢复实验还要冻结“未来会发生什么”，而不仅是保存过什么：continuous 与 interrupted 分支使用相同总 `max_steps`（因此 LR schedule 相同）、相同 seed、同一无放回样本排列、相同 world/micro/accum 和相同数据/model/config SHA。恢复后的第一个 successful update 必须比较全局 sample IDs、LR、loss scale 与 next loss，训练终点还要比较参数 state；只比较两个 run 都“到了 step 200”没有证明等价。

AMP 使用两只时钟：`attempted_update` 在每次 optimizer 尝试时前进；只有所有 rank 都确认没有 overflow/skip 后，`successful_update` 才前进。由于 Transformers Trainer 的 `global_step` 可能包含被 GradScaler 跳过的尝试，本周逐 update 审计；一旦任一 rank skip，就记录 `attempted=global_step, successful=global_step-1` 并令全作业失败，禁止继续发布 checkpoint。

## 7. 公平放大比较

固定：FinQA supported subset 与 hash、prompt/schema/mask、generation、evaluator、dev selection、主 seed、target tokens seen。允许变化：tokenizer/model、micro/accum、FSDP/LoRA、合理 LR。因此不可直接比较不同 tokenizer 的 raw loss；应比较：

```text
valid JSON/program, execution accuracy, evidence F1,
target tokens/s, step p50/p95, peak GB, communication ratio,
full/trainable params, checkpoint bytes, gpu-hours
```

任务收益必须超过评测方差或至少跨 seed 同方向，否则 `INCONCLUSIVE`。先建立 0.5B 重跑方差，再判断 1.5B 差异。

## 8. GRPO：优化对象、可观察性与 reward hacking

对同一 prompt 采样 `G` 个 response，reward 标准化后形成组内相对优势。简化形式：

\[
A_i=\frac{r_i-\bar r}{\operatorname{std}(r)+\epsilon},\qquad
L=-\frac1G\sum_i A_i\log \pi_\theta(y_i|x)+\beta D_{KL}(\pi_\theta||\pi_{ref}).
\]

reward 必须直接复用 Week 12 的稳定接口 `score(text, gold, allowed_evidence) -> (flags, error)`，不能另造一个“近似 evaluator”。落盘 `gold` 的唯一 schema 为 `{"evidence": list[str], "program": str, "answer": finite number, "scale": enum}`，其中 `answer` 是受限程序的原始执行结果，`scale` 单独评分。`flags` 固定且完整为：

```text
json, schema, evidence_ids_valid, evidence,
evidence_precision, evidence_recall, evidence_f1, scale,
program, execution, answer, consistent
```

其中 `evidence_ids_valid` 只回答 ID 是否来自允许集合，`evidence` 是对 gold evidence set 的精确匹配，二者不可混写。reward 必须逐项记录，不能只存 total。对于金融程序任务：

- program invalid 但答案碰巧对，不给 execution reward；
- evidence ID 不存在，evidence reward 为 0；
- reported 与 executed 不一致要惩罚；
- 限制 response 长度，防刷局部项；
- 不使用 PnL，不连接交易，不把预测当投资建议。

监控 KL、entropy、长度、unique response rate 与每项 reward。valid program 升而 evidence F1 降、输出变长、固定高频答案、利用 tolerance/rounding 都是 reward hacking。gold evaluator、SFT JSON 或 adversarial tests 任一未过，决策必须是 `RL-NOT-READY`。

TRL v0.15.2 的 GRPO generation batch 有整除约束：`num_processes × per_device_train_batch_size` 必须能被 `num_generations` 整除。本周单卡最小合法 smoke 固定 `per_device_train_batch_size=2, num_generations=2`；入口在 Trainer 构造前再次计算并 gate，不能用 `1/2` 配置碰运气。GRPO 仍须遵守 FP16 双时钟：逐 update 记录 attempted/successful、scale 与跨 rank skip，一次 skip 即失败；模型与 tokenizer 只从本地固定目录加载并强制 eager attention，reward/KL/长度/unique 指标必须持久化而不能只留终端滚屏。

LoRA adapter 不是自解释模型。可发布目录必须同时含 adapter 权重、`adapter_config.json`、tokenizer、base `repo@revision` 与 base tree SHA、训练 config/data/evaluator SHA，以及逐参数 name/shape/dtype 的 trainable manifest。验收必须在 offline 环境重新加载固定 base + adapter，比较 tokenizer probe，并在加载前后核对 adapter tensor 的完整 key/shape/value 集，再跑同一 eligible dev evaluator；缺键/多键不得通过关闭严格键校验来隐藏。

## 9. Shape/dtype worked example

1.5B config 的实际值必须从冻结 `config.json` 打印。若 `B_rank=1,S=512,H=1536,V=151936`：

| 张量 | shape | dtype |
|---|---|---|
| input/labels | `[1,512]` | int64 |
| hidden | `[1,512,1536]` | FSDP mixed FP16；norm/reduction 实现相关 |
| logits | `[1,512,151936]` | FP16 约 0.145GiB |
| target mask | `[1,512]` | bool |
| LoRA q A/B（示例 r=16） | `[16,1536]`, `[1536,16]` | trainable FP32/FP16 需审计 |

world=4、micro=1、accum=2 时 global samples/update=`8`。若各样本 target token 不等长，仅平均 rank loss 会产生权重偏差；主 scale 对照至少报告 effective target tokens，严谨等价需全局 `loss_sum/token_count`。

## 10. 最小可运行预算代码

```python
def gib(n): return n/1024**3
for name,p in {"0.5B":.5e9,"1.5B":1.5e9,"4B":4e9,"8B":8e9}.items():
    print(name,{"unsharded_16P_GiB":round(gib(16*p),2),
                "fsdp4_ideal_GiB":round(gib(16*p/4),2),
                "fsdp8_ideal_GiB":round(gib(16*p/8),2)})
def lora(in_f,out_f,r): return r*(in_f+out_f)
assert lora(2560,2560,16)==81920
```

预计输出只是纸面下限；任何人都不能把它写成 `nvidia-smi` 实测。

## 11. 正确性不变量

1. Week 12 evaluator/data/model hashes重验；
2. 模型 snapshot、许可和 config model_type 固定；
3. wrap 命中所有且仅 transformer blocks；
4. FSDP 启动前打印 full/trainable params、每 rank world/local rank 和 mixed precision；
5. global batch/effective target tokens 对照明确；
6. FP16 无非有限，各 rank 对 skip 一致，attempted/successful update 语义不混；
7. resume 只从完整 checkpoint，同 world-size 默认；continuous/resume 固定 schedule/sample stream 并比较 next loss/state；
8. adapter checkpoint 与 base revision/tree SHA、tokenizer、config及 trainable manifest绑定，offline reload/eval 成功；missing/unexpected keys 不得通过宽松加载隐藏；
9. test 只在冻结最终模型一次；
10. GRPO batch/generation 整除 gate 通过，且 reward 使用 Week12 全部固定 flags、adversarial、KL/entropy/length；无 PnL/交易/API。

## 12. 失败诊断树

| 症状 | 最小检查 | 根因候选 | 修复 | 回归 |
|---|---|---|---|---|
| 启动即 OOM | wrap 数、load 峰值、rank0 RSS | 先全量上 GPU、wrap 失配、full state gather | CPU load/sync states；修 wrap；缩 seq | 2 卡 1-step |
| step 中 OOM | peak 时点 | activation/logits/all-gather | checkpointing、缩 seq/micro、细化 wrap | 20-step memory curve |
| FSDP hang | rank/collective 日志 | 某 rank exception、数据数量不同、NCCL | 全 rank error；drop_last；2-step NCCL debug | 2→4 卡 |
| checkpoint 能列出但不能 resume | completion/files | optimizer shard/scaler/RNG/world mismatch | 拒绝坏目录；同 world 恢复 | continuous/resume |
| LoRA trainable 异常 | 打印每个 requires_grad | target modules 名不匹配或 base 未冻结 | 精确 module 列表 | optimizer param count |
| 放大只改善 JSON | waterfall | 容量用于格式，不是 evidence/program | 数据/evaluator优先 | 固定错误桶 |
| reward 上升、任务下降 | 分项/KL/长度 | hacking/collapse | 降权/修 oracle/停 RL | adversarial set |

## 13. 闭卷自检与 teach-back

1. 1.5B 的 `16P` 状态粗估是多少？为什么不等于 FSDP 峰值？
2. 为什么 DDP 不帮 4B base 分摊显存？
3. `r=16,in=out=2560` 的单层 LoRA 参数量是多少？
4. wrap 太细/太粗各有什么代价？
5. sharded resume checkpoint 与 full inference export 差什么？
6. 为什么不同 tokenizer 的 train loss 不可直接比较？
7. valid program 上升但 evidence F1 下滑是什么风险？
8. 用 2 分钟讲清 raw FinQA → 1.5B FSDP update → sharded/full checkpoint → greedy evaluator → scale decision。

## 14. 官方资料

- [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [PyTorch 2.1 FSDP](https://pytorch.org/docs/2.1/fsdp.html)
- [Transformers 4.46.3 FSDP](https://huggingface.co/docs/transformers/v4.46.3/en/fsdp)
- [PEFT v0.13.2 source tag](https://github.com/huggingface/peft/tree/v0.13.2)（该补丁版本没有稳定的版本化文档页；以此 tag 的源码/API 为准）
- [TRL v0.15.2 GRPO](https://huggingface.co/docs/trl/v0.15.2/en/grpo_trainer)

链接和 revisions 于 2026-09-03 核验。执行时先以锁定代码的本地 `--help`/signature 为准；API 不同即停止或形成新环境变更单，不能边跑边升级。
