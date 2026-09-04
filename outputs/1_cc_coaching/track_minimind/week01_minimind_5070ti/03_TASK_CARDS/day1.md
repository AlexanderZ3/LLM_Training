# Week M01 · Day 1 — 一条 JSONL 到底怎么变成 token 与 label

| 字段 | 值 |
| --- | --- |
| 主要产物 | 你能对着 `inspect_dataset` 的逐 token 表逐行复述"谁进 loss、谁是 -100、为什么"，并有 `pytest lab/tests/test_label_mask.py -q` 的 `11 passed` |
| 估时 | `90 分钟`（步骤估时之和） |
| 环境 | `CPU 即可`（本机 `ResearchAgentPy310` 全程可跑，不需要 5070 Ti） |
| AI 辅助等级要求 | `A1`（只查 `01_FOUNDATIONS.md` 第 1.2 节与 `lab/src/mm_probe/inspect_dataset.py` 源码；步骤 2 的手写序列必须先闭卷写完再核对） |
| 前置 | Day 0 的证据：`commit_matches = true`、`tokenizer_loads = true`、`pytest_tail = 42 passed` |
| 本卡对应门 | `静态检查`（`02_LAB_GUIDE.md` 的门 2：数据 → token/label 因果链） |
| 可裁剪项 | 步骤 5（真实数据聚合统计）在数据未下载完时可推迟；步骤 1、2、6 不可裁 |

## 为什么做

后面五天所有 loss 数字的含义都取决于分母里到底有哪些 token：pretrain 全进、SFT 只有 assistant 段进。这条规则如果只是"看过"而不是"能手写并被工具证实"，Day 3 的 mask 错位故障就无法只凭曲线反推，Gate 的手写 token/label 题也答不了。今天把这条因果链钉死在一张逐 token 表上。

## 步骤

### 1. 读懂 fixture 模式的逐 token 表（15 分钟）· `可直接执行`

```powershell
$Py = "D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe"
& $Py lab\scripts\mmp.py inspect_dataset --stage sft --fixture --n 1 --max-seq-len 64
```

- 预期（`本机实测`，fixture 第 0 条 `你好` / `你好！有什么可以帮你？`）：idx 0–11 全部 label = -100，idx 12 起 assistant 内容 label = input_id，idx 19–20 的 `<|im_end|>` 与 `\n` 也进 loss；`stats` 为 `n_nonpad: 21, n_in_loss: 9, first_loss_pos: 12, eos_in_loss: true`，`per_segment` 里 `user` 与 `pad` 的 `in_loss` 都是 0。
- 三条必须成立的不变量：`first_loss_pos` 正好落在 `<|im_start|>assistant\n` 这 5 个 token 之后（这里 12 = 7 + 5）；`user` 段与 `pad` 段 label 全为 -100；`eos_in_loss: true`。
- 不对时先查：`first_loss_pos` 为 `null` 时核对 `special_sequences(tokenizer)` 返回的 `bos_seq` 是不是 `[1, 1388, 570, 811, 234]`；不是就是 tokenizer 不对（`02_LAB_GUIDE.md` 第 5 节 F4）。

### 2. 闭卷手写一条新样本的 label 再核对（20 分钟）· `模板`

先在纸上写：自己造一条两轮 chat 样本（有 system、两问两答），闭卷写出 `<|im_start|>` / role / `\n` / 内容 / `<|im_end|>` 的分段顺序，标出哪些位置是 -100、`first_loss_pos` 应落在第几位。写完再落盘核对——把下面的样本内容换成你自己写的那条：

```powershell
'{"conversations":[{"role":"user","content":"什么是久期？"},{"role":"assistant","content":"久期衡量债券价格对利率的敏感度。"},{"role":"user","content":"谢谢"},{"role":"assistant","content":"不客气。"}]}' | Set-Content -Encoding utf8 lab\runs\my_sft_sample.jsonl
& $Py lab\scripts\mmp.py inspect_dataset --stage sft --data-path lab\runs\my_sft_sample.jsonl --n 1 --max-seq-len 96
```

- 预期：逐 token 表里出现**两段** label = input_id 的区间（两个 assistant 回合各一段），两段都以 `<|im_end|>` + `\n` 结尾且这两个 token 进 loss；两个 user 回合与所有模板 token 全是 -100；`stats.first_loss_pos` 等于你手写的那个位置。
- 不对时先查：如果只出现一段进 loss 的区间，说明第二轮的 `<|im_start|>assistant\n` 没被匹配上；把 `--max-seq-len` 调大到 128 再跑一次，看 `stats.truncated` 是不是 `true`（截断会吃掉第二轮）。

### 3. 把两个随机分支变成确定开关各看一次（10 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\mmp.py inspect_dataset --stage sft --fixture --n 1 --max-seq-len 64 --add-system --keep-empty-think
```

- 预期：加 `--add-system` 后开头多出一个 system 块，其 label 仍全为 -100；加 `--keep-empty-think` 后 assistant 段内多出 `<think>` / `</think>` token，且它们**进** loss（label = input_id）。MiniMind 在 `__getitem__` 里以 20% 概率加 system、80% 概率删空 think 块，本模块把它改成确定开关，否则同一条样本两次编码结果不同、没法对照。
- 不对时先查：若加开关前后 `n_in_loss` 完全没变，检查命令里两个开关的拼写（`--add-system`、`--keep-empty-think`），它们是 store_true 开关，拼错时 argparse 会直接报错而不是静默忽略。

### 4. 对照 pretrain 的另一套规则（10 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\mmp.py inspect_dataset --stage pretrain --fixture --n 1 --max-seq-len 64
```

- 预期：与 SFT 完全不同——除 pad 外所有位置 label = input_id，包括首尾的 `<|endoftext|>`；`first_loss_pos` 为 0，`per_segment` 里没有 -100 的非 pad 段。用一句话写下"这两者在数学目标上唯一的差别"。
- 不对时先查：若 pretrain 也出现大段 -100，说明 `--stage` 传错或数据字段名不对（pretrain 数据每行只有一个 `text` 字段）；打开 `lab/src/mm_probe/fixtures.py` 看 pretrain fixture 的字段名。

### 5. 在真实数据上跑聚合统计（20 分钟）· `可直接执行`

需要 Day 0 步骤 1 下载的 `sft_t2t_mini.jsonl` 与 `pretrain_t2t_mini.jsonl`。

```powershell
& $Py lab\scripts\mmp.py inspect_dataset --stage sft --n 5 --no-table --out lab\runs\inspect_sft.json
& $Py lab\scripts\mmp.py inspect_dataset --stage pretrain --n 5 --no-table --out lab\runs\inspect_pretrain.json
```

- 预期（`估算`）：`aggregate.mean_loss_token_ratio` 在 0.3–0.6 之间（SFT 只有 assistant 段进 loss）；默认 `max_seq_len 768` 下 `truncation_ratio` 远小于 1；`eos_in_loss_ratio` 应为 1.0。
- 数据没下载时先做：把两条命令的 `--n 5` 段改成 `--fixture --n 3`，输出分别写到 `lab/runs/inspect_sft_fixture.json` 与 `lab/runs/inspect_pretrain_fixture.json`，走内置样本先把字段读一遍；注意 fixture 的截断率与平均长度不可外推到真实数据。
- 不对时先查：`mean_loss_token_ratio` 接近 0 或接近 1 就立刻停——前者是 bos 序列匹配失败，后者是 mask 没生效，两种都在 `02_LAB_GUIDE.md` 第 5 节 F5。

### 6. 跑 label mask 单测并读它断言了什么（15 分钟）· `可直接执行`

```powershell
& $Py -m pytest lab\tests\test_label_mask.py -q --basetemp=D:\tmp\pt
```

- 预期：`11 passed`。打开 `lab/tests/test_label_mask.py` 找到与 MiniMind 原 `SFTDataset.generate_labels` / `PretrainDataset` **逐位**对比的那几条，用一句话说出它们各自钉死了哪个不变量。
- 不对时先查：出现 `skipped` 而不是 `passed`，说明 `MINIMIND_ROOT` 没设（`conftest.py` 会让依赖 tokenizer 的用例 skip 而不是失败）；先 `$env:MINIMIND_ROOT = "D:\work\minimind"` 再重跑。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M01-R-01`：分别写出 pretrain 与 SFT 的 `labels` 构造规则，以及两者数学目标上唯一的差别。
2. `M01-E-02`：为什么 `generate_labels` 按序列而不是按单个 token id 匹配起点与终点，简化后各会产生什么后果。

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 1,
  "day": 1,
  "card": "day1",
  "skill": "token_label_causal_chain",
  "env": "cpu",
  "ai_level": "A1",
  "config_id": "minimind@7a6fddd + inspect_dataset fixture/mini",
  "primary_artifact": "lab/runs/inspect_sft.json + pytest lab/tests/test_label_mask.py",
  "observations": {
    "fixture_first_loss_pos": 12,
    "fixture_n_in_loss": 9,
    "fixture_eos_in_loss": "true",
    "my_sample_first_loss_pos": 0,
    "my_sample_handwritten_matches": "是 | 否",
    "pretrain_vs_sft_diff": "一句话",
    "mean_loss_token_ratio_sft": 0.0,
    "truncation_ratio_sft": 0.0,
    "eos_in_loss_ratio": 0.0,
    "test_label_mask_tail": "11 passed"
  },
  "status": "PASS | FAIL-MODEL | FAIL-SYSTEM | INCONCLUSIVE",
  "failure_class": "none | data_contract | measurement | env",
  "self_check": {"q1": "能 | 不能", "q2": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "一句话，不含敏感信息"
}
```
