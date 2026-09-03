# Week 12 基础篇：FinExec 0.5B 的数据、mask、evaluator、SFT 与短 CPT

> 状态：从零训练手册，尚未在用户机器执行；任何性能或效果值均为验收目标/估算。  
> 核验日期：2026-09-03。公司 PyTorch 2.1 与 8×V100 32GB 为用户自述待核验。

## 1. 本周问题与最终产物

本周不做交易 Agent、RAG、PnL reward 或投资建议。唯一任务是把公开财报问题变为：

```text
document + question → evidence IDs → restricted program → executed answer + scale
```

最终需要一个可从空目录重建的 FinQA 主路径：固定源数据/模型 revision，安全 executor，统一 JSON schema，不 packing 的正确 label mask，128 样本 overfit，`base→SFT` 与可选 `base→短 CPT→SFT`，独立生成评测和完整 checkpoint。TAT-QA 只作为后续 transfer；没有 adapter 测试时不得把它混入训练。

本周交付回答两个不同问题：

- 工程正确性：evaluator、schema、mask、生成、resume 是否可靠；
- 模型效果：SFT/CPT 是否改变 valid program、execution accuracy 与 evidence F1。

前者失败时只能标 `FAIL-SYSTEM`；后者只有在前者通过后才能判 `FAIL-MODEL/INCONCLUSIVE/PASS`。

## 2. 环境与使用边界

| 环境 | 职责 | 建议路径 | 边界 |
|---|---|---|---|
| 公司 Linux 8×V100 | 0.5B 全参 SFT，1/2/4/8 卡 correctness/profile | PyTorch 2.1、FP16 AMP/eager attention | 公开数据也须审批；模型、日志、指标、checkpoint 不外传 |
| 个人 5070 Ti | 16–128 样本、少步公开复现 | 显存不足则缩短 sequence/micro batch | 不复制公司结果；重新下载公开源并重跑 |
| 可选 H100 | 仅预注册 BF16/规模对照 | 单独 manifest | 不将 BF16/H100 与 V100 FP16 合并成同一结果 |

V100 支持 FP16；通常没有原生 BF16。固定 `attn_implementation="eager"`，不安装 FlashAttention-2。公司 torch 2.1 是约束：缺库时优先使用批准 wheelhouse，不能为了 Qwen/TRL 升级 torch/CUDA。

## 3. 固定数据、模型、框架与许可

| 对象 | 官方来源 | 冻结版本 | 许可/限制 |
|---|---|---|---|
| FinQA | [czyssrs/FinQA](https://github.com/czyssrs/FinQA) | `0f16e2867befa6840783e58be38c9efb9229d742` | 仓库 LICENSE 为 MIT；保留 LICENSE/引用；主训练数据 |
| TAT-QA | [NExTplusplus/TAT-QA](https://github.com/NExTplusplus/TAT-QA) | `870accc41953dcde885aabeb963d94aabdc0fbc3` | 数据 CC BY 4.0；仓库代码 LICENSE 另核；只作 transfer |
| 基座 | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | `060db6499f32faf8b98477b0a26969ef7d8b9987` | Apache-2.0；使用 base 而非 instruct/chat |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) | `v4.46.3` (`97e5c42…`) | Apache-2.0；锁 wheel，不从 main 安装 |
| Accelerate | [huggingface/accelerate](https://github.com/huggingface/accelerate) | `v1.1.1` | Apache-2.0；由 Trainer/DDP 使用 |
| huggingface_hub | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) | `v0.26.2` (`63db6d6…`) | Apache-2.0；仅下载固定 snapshot |

`datasets` 不是核心依赖：Lab 直接读取官方 JSON，减少 Arrow/版本变量。模型 snapshot 只允许 `safetensors`；避免加载不明 pickle。若固定 revision 无法获得，允许的 fallback 是审批过的同 revision 离线包，或 Week 01 的本地小模型做 pipeline smoke；后者不能产生 0.5B 效果结论。

## 4. 数据对象：从 FinQA 原始条目到监督样本

FinQA 的 `dataset/train.json` 每条包含官方唯一 `id`、财报前文 `pre_text`、表格 `table`、后文 `post_text` 和 `qa`。adapter 必须原样保留官方 item ID，并产生稳定 evidence ID：

```json
{
  "id": "ADI/2009/page_49.pdf-1",
  "context": "[text_0] Revenue rose year over year.\n[table_0] 2023 | 128.4 | 97.1",
  "question": "What was the percentage change from 97.1 to 128.4?",
  "target": {
    "evidence": ["table_0", "text_0"],
    "program": "subtract(128.4,97.1),divide(#0,97.1)",
    "answer": 0.3223,
    "scale": "percent"
  },
  "source_program": "subtract(128.4, 97.1), divide(#0, 97.1)"
}
```

`source_program` 保留原标注；`program` 是去除无意义空白后的 canonical sequential DSL，仍保持官方 step 与 `#N` reference 语义。不能把不受支持的程序静默改写成“看起来等价”的 gold。gold 解析/执行异常与 tokenizer 长度排除进入不同 manifest，记录原值、原因、处理动作。

官方 split 必须保留。CPT corpus 只能来自 train 文档，不包含 train answer、更不能包含 dev/test context/answer；CPT 与 SFT 的数据 hash 分开。

## 5. 输出 schema 与安全执行

模型只允许输出一个 JSON 对象：

```json
{"evidence":["table_0"],"program":"divide(10,4)","answer":2.5,"scale":"none"}
```

schema 不变量：键集合固定；evidence 是无重复的字符串数组；program 长度与 step 数有上限；answer 必须是有限数；scale 属于 `none|percent|thousand|million|billion`。这里 `answer` 固定表示 restricted program 的原始数值结果，`scale` 是从 official answer 推导并单独评分的展示单位；evaluator 不按模型自报 scale 偷偷乘除数值。

FinQA 的 `greater` 可把官方 `yes/no` execution result canonicalize 为数值 `1/0`；带 `%` 的 operand 进入算术时必须除以 100。官方 program 同时存在“保留 ratio、展示时乘 100”和“program 已乘 `const_100`”两类 percent 约定，因此 adapter 要用两类冻结 fixture 和按展示精度计算的舍入容差验证 display/raw 关系，不能无条件再乘一次，也不能只凭字符串后缀猜 scale。

executor 只接受 top-level step splitter、严格正则匹配的 op 和向后 `#N` reference，不使用 Python `eval/exec`，不允许嵌套任意表达式、属性访问、import、变量、文件、网络或任意函数。执行 waterfall：

```text
text → JSON parse → schema → evidence existence → program parse
     → op/arity → finite execution → scale → gold comparison
     → reported answer 与 executed result 一致性
```

证据与答案也必须拆成可诊断指标：

- evidence ID validity：预测 ID 是否全都存在于完整 prompt；
- evidence exact / precision / recall / F1：与 gold evidence 集合比较；gold 非空时空预测的 F1 为 0，绝不能因“空集是 allowed 子集”而通过；
- scale accuracy：预测 scale 是否精确等于 gold scale；
- execution accuracy：canonical program 执行后是否等于 gold；
- reported consistency：JSON 的 `answer` 是否等于其 program 的执行结果。

答案碰巧正确但 program 无效，不算过程正确。

## 6. 受限 DSL

最小语法：

```ebnf
program   := step ("," step)*
step      := ident "(" operand "," operand ")"
operand   := number | reference
reference := "#" digit+
ident     := add | subtract | multiply | divide | exp | greater
```

FinQA 原程序常以逗号分隔多个 step，后续 step 用 `#0/#1` 引用。本周 adapter 去除空白并由 executor 顺序解析每个 step。每个 op 固定 arity；除零、NaN/Inf、超长/过多 step、嵌套调用与未知 reference 都拒绝。tolerance 预注册，例如：

\[
|\hat y-y|\le \max(10^{-4},10^{-3}|y|).
\]

percent 不是“随手乘 100”：adapter 以官方 `exe_ans` 作为 raw executor gold，另从 official displayed answer 提取 scale；score 同时检查 raw execution、reported consistency 与 scale equality。该合同必须由 30+ 人工案例与清洗后 official gold 验证。gold evaluator 未达到已声明有效集合的 100%，不得训练。

### 6.1 Eligibility 不是隐式截断

冻结 tokenizer 后先生成 `eligible_ids.json` 与带 SHA 的 manifest。prompt 始终包含完整线性化文档，不按 gold evidence 选句或截句。train 只有在 `prompt+target+EOS≤max_length` 时进入 SFT；dev/test 在 supported gold set 内只按 prompt token 长度选择，不能利用 gold evidence/answer 长度挑“容易样本”。所有候选模型读取同一份 eligible dev/test JSONL；否则指标不可比较。运行时发现超长必须报错并重跑 adapter，Dataset 不得静默 drop。

## 7. Prompt、token、mask 与 shape

prompt 固定为：任务约束 + context + question + `OUTPUT_JSON:`。target 是 compact JSON 加 EOS。以 `B=2,S=512,V=151936` 为 worked example（实际 V/H 必须从冻结 `config.json` 打印）：

| 张量 | shape | dtype/含义 |
|---|---|---|
| `input_ids` | `[2,512]` | int64 |
| `attention_mask` | `[2,512]` | int64/bool |
| `labels` | `[2,512]` | int64；prompt/pad 为 `-100` |
| logits | `[2,512,V]` | autocast FP16；softmax/CE 内部按实现处理 |
| target mask | `[2,512]` | `labels != -100` |

decoder-only LM 在位置 `s` 的 logits 预测 token `s+1`；Transformers 模型内部完成 shift。collator 不应再手工 shift。token-level loss：

\[
L=-\frac{\sum_{b,s}m_{bs}\log p_\theta(y_{bs}\mid x_{b,<s})}
{\sum_{b,s}m_{bs}}.
\]

prompt、padding 必须 `m=0`；target JSON 与 EOS 为 1。adapter 在训练前把 `prompt+target+EOS` 超限样本写入 length exclusions；Dataset 若仍遇到超限必须报错，不能静默截断、丢弃或把残缺 JSON 当 gold。

本周正式路径明确关闭 packing。原因：普通 causal mask 不能阻止 packed 样本 B 看见样本 A；若没有 block-diagonal attention 与完整测试，就不以吞吐换正确性。

## 8. SFT 与短 CPT 的差异

| 项 | SFT | CPT |
|---|---|---|
| 输入 | prompt + target JSON | 仅 train 财报文本/表格线性化 |
| labels | 只在 target/EOS 计 loss | 非 padding token 计 next-token loss |
| 目的 | 学 schema/evidence/program | 演练继续预训练管线 |
| checkpoint 选择 | dev execution / 预注册组合 | train/dev LM loss，仅作为中间点 |
| 能否宣称金融知识 | 不能仅凭 loss | 0.5M–2M token 更不能 |

公平比较 `base→SFT` 和 `base→CPT→SFT` 时，SFT train IDs、target tokens、generation config、evaluator、seed 与 dev selection 相同。CPT 后一般重新建立 SFT optimizer；需记录这一选择。

## 9. 最小可运行 mask 示例

```python
import torch
prompt=torch.tensor([[10,11,12]])
target=torch.tensor([[20,21,2]])       # 2 假设为 EOS
ids=torch.cat([prompt,target],dim=1)
labels=ids.clone(); labels[:,:prompt.shape[1]]=-100
logits=torch.randn(1,6,32,requires_grad=True)
loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100)
loss.backward()
print({"ids":ids.tolist(),"labels":labels.tolist(),"loss_finite":bool(torch.isfinite(loss))})
```

预计示例（非实测）：labels 前三个位置为 `-100`，loss finite。注意第一个 target token 由最后一个 prompt 位置预测，因此模型内部 shift 不能被重复实现。

## 10. 训练系统联动

### 10.1 显存与吞吐

全参 AdamW 粗估每参数约 12–16 bytes（参数/梯度/FP32 optimizer 状态实现相关），0.5B 已可能需要 6–8GB 以上模型状态，再加 activation、logits、bucket 和 allocator。logits `[B,S,V]` 很大：`2×512×151936×2 bytes≈0.31GB`，这是缩 sequence/micro batch 常有效的原因。

DDP 不分摊模型状态；每卡仍有完整 0.5B。`global_batch=micro×accum×world`。比较卡数时固定 target tokens 或 global batch，记录 successful updates。FSDP 只有在确需 sharding 时使用 Week 08 已验证路径。

### 10.2 FP16

V100 使用 FP16 AMP + GradScaler，不设 BF16。gradient clip 在 unscale 后；非有限与 skip 决策必须跨 rank 一致。固定 Transformers v4.46.3 的 Trainer 可能在 AMP optimizer update 被跳过时仍增长 `global_step`，因此本周逐步审计并在任一 skip 时整作业失败；只有所有 `optimizer_step_was_skipped=false` 时才把 `global_step` 解释为 successful update。记录 loss scale、skip、grad norm、eligible target-token 分布、Trainer samples/s 和 peak allocated/reserved。

### 10.3 checkpoint

“仅权重”可评测，不能保证等价续训。完整 resume 至少要 model、optimizer、scheduler、FP16 GradScaler、Python/NumPy/Torch/CUDA RNG、global successful step、epoch/sampler offset、config/data/code/model revision。有效恢复合同还要绑定 resolved precision、world size、micro batch、accum、LR/warmup/weight decay 与 max steps；只允许同一 output-dir 的 checkpoint。每个 checkpoint 的 on-save 验证器必须枚举必要文件，为每个 payload 写字节数与 SHA-256，最后发布 `COMPLETED`；resume 要在任何权重反序列化前核对该 seal，并核对上次 run manifest 冻结的 checkpoint-manifest SHA。StopAfter 是恢复测试的中断控制，不改变合同：中断状态必须是 `INTERRUPTED_FOR_RESUME`，不得发布 `final_model`；完整结束才保存 model+tokenizer。Trainer checkpoint 的实际内容必须现场列举并做 continuous-vs-resume fixed batch 测试，不能因目录存在就宣称通过。

## 11. 正确性不变量

1. source revision、LICENSE、数据 SHA 与模型 SHA 入 manifest；
2. official split 不改变，dev/test 不进 SFT/CPT train；
3. prompt 中所有 gold evidence ID 存在；
4. gold-supported set 与 length-eligible set 分开列出；dev/test 长度选择不读取 gold evidence/answer，所有模型使用同一 eligible ID 清单；
5. 清洗后有效 gold set 的 parse/execute/raw-answer/scale 合同为 100%，异常显式列出；
6. executor 不调用 `eval/exec`；step/reference、op 与数值有限；
7. labels decode 仅得到 target JSON+EOS；prompt/pad 均为 `-100`；
8. packing 关闭；truncated target 拒绝且 Dataset 不静默 drop；
9. generation 从 `input_length` 后切片，固定 greedy/max_new_tokens；
10. evidence validity、exact/F1、scale、execution、answer consistency 分开报告；
11. best checkpoint 只按 dev，test 最终一次；
12. 模型输出必须有“研究用途、非投资建议”，且绝不连接交易/外部工具。

## 12. 失败诊断树

| 症状 | 最小检查 | 根因 | 修复 | 回归 |
|---|---|---|---|---|
| gold evaluator <100% | 打印首个 op/scale/error | parser、reference、percent、异常标注 | 修 spec 或显式 exception | 30+ unit + 全 gold |
| loss 降而 JSON 无效 | decode 128 train | mask/shift/EOS/prompt mismatch | 可视化 labels；统一 template | 8 样本 greedy overfit |
| answer 对、program 错 | 比 executed/reported | 模型猜数或 schema 漏检 | 分开两指标；程序优先 | adversarial outputs |
| 多卡 loss 偏小 | 对比 1 卡 fixed batch | 重复除 world 或 token 分母局部化 | 全局 token sum/count 或等长 batch | 1/2/4 卡 20 step |
| resume 后曲线跳变 | fixed next batch | scheduler/scaler/RNG/sampler 缺失 | 补完整状态 | continuous/resume <1% |
| FP16 NaN | 首个坏 sample/step | 长序列、LR、坏数、overflow | FP32 重放；缩 batch/LR | 20-step FP32→FP16 |
| evidence F1 高但 exec 低 | error waterfall | op/operand/scale 能力不足 | 定向数据，不上 RL | 固定错误桶回归 |
| 空 evidence 却“通过” | 看 validity 与 F1/exact | 只做 allowed-set 子集检查 | 增加 gold set P/R/F1/exact | 空集与部分命中单测 |
| 模型间 eval n/ID 不同 | 比 data/manifest/ID SHA | 动态 truncation 或各跑各的过滤 | 回到冻结 eligible JSONL | base/SFT hashes 完全一致 |

## 13. 闭卷自检与 teach-back

1. 为什么模型 reported answer 与 executor result 必须分开？
2. prompt 共 400 token、target 60、pad 52 时，loss 分母是多少？
3. 为什么普通 causal mask 下的 packing 会泄漏？
4. 什么条件下可以从 gold evaluator 中排除异常样本？
5. DDP world=4、micro=1、accum=8 时 global batch 是多少？
6. 为什么短 CPT loss 下降不能证明获得金融知识？
7. 用 2 分钟讲：官方 JSON → stable evidence → prompt/labels → FP16 update → greedy generation → safe executor → waterfall。

合格回答必须区分系统 oracle、teacher-forced loss、autoregressive generation、程序执行与最终答案；并说明这不是投资建议。

## 14. 官方资料

- [FinQA 官方仓库](https://github.com/czyssrs/FinQA)
- [TAT-QA 官方仓库](https://github.com/NExTplusplus/TAT-QA)
- [Qwen2.5-0.5B 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [Transformers v4.46.3 文档](https://huggingface.co/docs/transformers/v4.46.3/en/index)
- [Transformers causal language modeling](https://huggingface.co/docs/transformers/v4.46.3/en/tasks/language_modeling)
- [PyTorch AMP 2.1](https://pytorch.org/docs/2.1/notes/amp_examples.html)

链接和上述 revisions 于 2026-09-03 核验。将来若兼容矩阵改变，先解析并记录新 revision/许可，再更新手册；不得把 `main` 当可复现实验版本。
