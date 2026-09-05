# Week M01 — MiniMind 单机全链路（5070 Ti）（一页周卡）

> 状态：`BUILT / RUNTIME-UNVERIFIED`（文档与 CPU 门通过；GPU 门待 5070 Ti）  
> 生成日期：2026-09-04 · 构建完成：2026-09-05 · 生成方：cc · 轨道：`track_minimind`  
> 来源：`input_info/minimind_5070ti_v100.md`（用户意图）+ 2026-09-04 来源审计

## 本周唯一主问题

一条 JSONL 样本如何变成 token/label（谁进 loss、谁是 -100），走过 decoder-only 前向，再经 pretrain → SFT → LoRA → DPO → GRPO 改变权重，并被你自己写的仪表观察到？

## 核心实现 · 关键故障练习 · Gate

| 项 | 内容 |
| --- | --- |
| 核心实现 | 自写仪表包 `lab/src/mm_probe/`：数据/label-mask 检查、日志解析与曲线、显存/吞吐 hook、DPO/GRPO 统计；挂在固定 commit 的 MiniMind 原脚本之上 |
| 关键故障练习 | SFT label mask 错位注入（assistant 段全 -100，或 user 段也进 loss），只凭 loss 曲线与固定 prompt 生成对照反推故障 |
| Gate | A1 限时 45 分钟：给一条新的多轮 chat 样本手写 token/label 序列；解释 DPO 与 GRPO 的损失各吃什么输入；一次 mask 错位 debug 证据；两个不同日期的证据 |

## 先修

- 本轨道位于 14 周主线之前，无周证据前置。
- 本机 `ResearchAgentPy310` 已装 CPU torch 2.14 + pytest（`已确认`，2026-09-04），用于课程构建期的单测。
- 用户需确认：每周可投入小时数（预算约 11 小时）；5070 Ti 上的驱动与 PyTorch 版本（Day 0 探针）。

## 数据与模型对象（已核验，2026-09-04）

| 对象 | 官方 ID | revision / 固定方法 | license | 用途 | 标签 |
| --- | --- | --- | --- | --- | --- |
| MiniMind 代码 | `https://github.com/jingyaogong/minimind` | commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`（2026-08-31） | Apache-2.0 | 全周训练脚本、模型、tokenizer（vocab 6400，自带，README 不建议重训） | 已确认 |
| 预训练数据 | HF `jingyaogong/minimind_dataset` → `pretrain_t2t_mini.jsonl` | 1,241,043,656 B；下载当日记录 sha256 | 数据集卡片 | Day 1–2，字段 `{"text"}` | 已确认 |
| SFT 数据 | 同上 → `sft_t2t_mini.jsonl` | 1,739,201,170 B | 同上 | Day 1、3，字段 `{"conversations":[{role,content}]}`；loss 只在 `<\|im_start\|>assistant\n … <\|im_end\|>\n` 之间；无 packing | 已确认 |
| DPO 数据 | 同上 → `dpo.jsonl` | 53,653,322 B | 同上 | Day 4，字段 `{"chosen":[...],"rejected":[...]}` | 已确认 |
| RLAIF 数据 | 同上 → `rlaif.jsonl` | 23,754,740 B | 同上 | Day 5 GRPO prompts | 已确认 |
| Reward 模型 | HF `internlm/internlm2-1_8b-reward` | safetensors 3,399,182,888 B；`trust_remote_code` | 模型卡片 | Day 5；放不下时用规则 reward 替代 | 已确认存在；16 GB 显存占用未实测 |

## MiniMind 关键事实（决定任务卡参数）

- 训练脚本在 `trainer/`（不是 `scripts/`）：`train_pretrain.py`、`train_full_sft.py`、`train_lora.py`、`train_dpo.py`、`train_grpo.py` 等；多卡用 `torchrun --nproc_per_node N`，无 DeepSpeed 配置。
- 默认模型 `hidden_size=768, num_hidden_layers=8, heads=8, kv_heads=4, vocab=6400, tie_word_embeddings=True`；attention 走 `F.scaled_dot_product_attention`（不依赖 flash-attn 包）。
- `train_pretrain.py` 默认 `max_seq_len=340`（用户文档写 768 是误记）、`batch_size=32`、`accumulation_steps=8`、`dtype=bfloat16`、`GradScaler(enabled=dtype=='float16')`。
- `train_full_sft.py` 默认 `max_seq_len=768, batch_size=16, lr=1e-5`。
- `train_lora.py` rank 固定 16，无 `--lora_rank`；只作用于方阵 Linear。
- `train_dpo.py` `beta=0.15, lr=4e-8, batch=4`；ref 模型二次加载并冻结。
- `train_grpo.py` 默认 `loss_type=cispo`（不是 grpo），`num_generations=6, max_gen_len=1024, batch=2`，bf16 且无 GradScaler；advantage = 组内 (r−mean)/(std+1e-4)。
- checkpoint：`{weight}_{hidden}.pth`（half）+ `_resume.pth`（model/optimizer/epoch/step/world_size）。
- 5070 Ti（sm_120）：需 PyTorch ≥ 2.7 的 cu128 wheel（`已确认`）；MiniMind requirements 注释 `torch==2.6.0`，以 5070 Ti 可用 wheel 为准。

## 逐日地图

| Day | 目标 | 主要产物 | 门 | 估时 | 环境 |
| --- | --- | --- | --- | --- | --- |
| 0 | 探针 + 锁定 commit + mini 数据 sha256 + CPU tiny 配置 | `lab/scripts/probe_env.py` 输出 `probe.json`；`tests/test_env.py` 通过 | 静态 | 60 | 5070 Ti（本机可先做 CPU 部分） |
| 1 | 数据 → tokenizer → token/label 因果链 | `mm_probe/inspect_dataset.py` + `tests/test_label_mask.py` | CPU 单测 | 90 | CPU |
| 2 | 单 batch 前反向 → 128 样本 overfit → 有界 pretrain | `mm_probe/hooks.py` + `scripts/run_pretrain_bounded.*` + `mm_probe/parse_log.py` 曲线 | 单 batch → smoke → overfit → 有界 | 120 | 5070 Ti |
| 3 | Full SFT + mask 故障注入（可选 LoRA） | `scripts/run_sft_bounded.*` + `tests/test_mask_fault.py` + 正常/错位对照 | 有界训练 + 故障 | 120 | 5070 Ti |
| 4 | DPO：chosen/rejected 对数比与 ref 冻结 | `mm_probe/dpo_check.py` + `scripts/run_dpo_bounded.*` | 有界训练 | 90 | 5070 Ti |
| 5 | GRPO smoke + 独立 eval + resume | `scripts/run_grpo_smoke.*` + `mm_probe/grpo_stats.py` + `scripts/eval_generate.py` | smoke → eval → 恢复 | 120 | 5070 Ti |

总估时：约 10 小时 + Gate 1 小时。可用时间不足时先砍 Day 3 的 LoRA 扩展与 Day 5 的 `num_generations=4` 复跑，不砍 Gate。

## 三类环境边界

| 环境 | 本周用途 | 限制 |
| --- | --- | --- |
| 个人 5070 Ti 16 GB | 主线 | 需 torch ≥ 2.7 cu128；GRPO 默认配置极可能 OOM，按 batch 1 / gen 2 / seq 256 起步；结果不能冒充多卡结果 |
| 本机 CPU（ResearchAgentPy310） | 课程构建期单测、tiny-config dry run | 无 GPU；不做任何有界训练 |
| 公司 8×V100 | 本周不需要 | — |
| 可选 H100 | 本周不需要 | — |

## 本机验证状态

构建完成日：2026-09-05。本机（Windows，conda `ResearchAgentPy310`，Python 3.10.20，torch 2.14.0+cpu，transformers 4.57.6，**无 GPU**）实测：

| 检查 | 结果 |
| --- | --- |
| `py_compile` 全部 23 个 `.py` | 通过，退出码 0 |
| `pytest lab/tests -q`（设 `MINIMIND_ROOT` + `--basetemp`） | `42 passed`，0 失败 0 跳过 |
| 不设 `MINIMIND_ROOT` 时 | `29 passed, 13 skipped`（跳过项需真实 tokenizer 或模型类）；无一项因缺 GPU 而跳过 |
| 周包结构校验 `validate_cc_week.ps1 -Week 01 -Track track_minimind -IdPrefix M` | `PASS`，0 error 0 warning；口试 28 题 / 28 答一一对应 |
| 11 个 CLI 的真实输出 | 已实跑并写入 `02_LAB_GUIDE.md`，标注 `本机实测` |

**仍未验证（必须在 5070 Ti 上执行）**：任何 CUDA 路径、fp16 与 GradScaler 行为、显存与吞吐数字、真实数据集下载与 hash、GRPO smoke 是否 OOM、reward 模型能否放下。`02_LAB_GUIDE.md` 中标 `估算` 的数字全部属于此类。

**本机已知环境问题**：`C:\Users\13289\AppData\Local\Temp\pytest-of-13289` 目录权限异常，直接跑 pytest 会有 11 个用到临时目录的用例报 `PermissionError [WinError 5]`。加 `--basetemp=<可写目录>` 即可绕过；删除该残留目录可永久解决。

## 已知限制与风险

- 5070 Ti 的 PyTorch/驱动组合在探针前是 `未知`；Day 0 出现 `sm_120 not supported` 即为 `BLOCKED`，先装 cu128 wheel 再继续。
- 16 GB 上 GRPO + 1.8B reward 模型的显存无实测；Day 5 预注册 `INCONCLUSIVE` 出口。
- 数据文件较大（预训练 mini 约 1.2 GB，SFT mini 约 1.7 GB）；Day 0 只校验 hash，不加载全文件。全量数据集加奖励模型共 27.00 GB，整夜下载的完整说明见 [`06_DATASET_DOWNLOAD.md`](06_DATASET_DOWNLOAD.md)。
- 数据集许可同时标注 `apache-2.0` 与 `cc-by-nc-2.0`（**含非商用条款**），奖励模型许可标注为 `other`；本地学习无碍，公开作品集或商业用途前自行确认边界。
- 若用户只运行原仓库脚本而没写仪表，当天没有主要产物，证据不计。

## 来源

- `input_info/minimind_5070ti_v100.md`（用户意图，其中文件名、默认值已按官方纠正）
- MiniMind 仓库、HF 数据集 API、PyTorch 2.7 发布说明、PyTorch main `sdp_utils.cpp`（核验 2026-09-04）
