# Week M02 · Day 3 — 让教学版专家并行的六步流水在 8 卡上跑起来，扫 EP=1/2/4/8 并读出通信与计算的此消彼长

| 字段 | 值 |
| --- | --- |
| 主要产物 | 一张四行扫描表（EP=1/2/4/8 各一行：`a2a_time_fraction`、`expert_compute_fraction`、`wait_spread_ratio`、`recv_token_imbalance`），来自 `<公司内路径>/day3/ep{1,2,4,8}_rank<r>.jsonl` |
| 估时 | `120 分钟`（步骤估时之和） |
| 环境 | `公司 V100`（步骤 1、2 在任何 CPU 机器上也能跑） |
| AI 辅助等级要求 | `A1`（`ep_moe.py` 的六步流水要能对着源码逐步复述；不变量 I2 的对账必须自己做） |
| 前置 | Day 2 的证据：`verify_pass = true`（分片 checkpoint 链路成立）；Day 1 的 `equiv_pass = true` |
| 本卡对应门 | `CPU 单测` → `单 batch` → `短 smoke` |
| 可裁剪项 | 步骤 4 里的 EP=2 与 EP=4（时间不够时只跑 EP=1 与 EP=8 两端）；步骤 1、2、5 不可裁 |

## 为什么做（三句以内）

专家并行是本周唯一由你自己实现的放置方式，它把"token 去哪张卡"从数据维度换成了专家维度。EP 的所有故障（hang、木桶效应）都来自一个跨 rank 的约定：我发给你多少 token，你必须正好准备好收多少——这就是不变量 I2。今天先在单进程上证明 EP=1 与原始 `MOEFeedForward` 数值一致，再上 8 卡量 a2a 与专家计算的占比如何随 EP 变化。

## 步骤

### 1. 先过 EP 的 CPU 单测门（10 分钟）· `可直接执行`

```bash
export MINIMIND_ROOT=<MiniMind 仓库绝对路径>
python -m pytest lab/tests/test_ep_cpu.py lab/tests/test_split_sizes.py -q
```

- 预期：`test_ep_cpu.py` 7 个用例、`test_split_sizes.py` 11 个用例全部通过（含 2 进程 gloo 的 permute/unpermute 往返、某专家收到 0 token 不崩）。这一门在写课程的机器上已经通过（`本机实测`），在公司机器上再跑一遍是为了确认这台机器的 gloo 也支持 `all_to_all_single`。
- 不对时先查：`test_ep_cpu.py` 整体 skip 并写明 gloo 不支持 `all_to_all_single` → 这是环境差异不是代码缺陷，记进 `notes` 后直接进步骤 2；报 `PermissionError` 指向临时目录 → 加 `--basetemp <一个可写目录>`。

### 2. EP=1 与稠密实现的数值等价（15 分钟）· `模板`

```bash
export PYTHONPATH=$PWD/lab/src:$PYTHONPATH
python -m mm_dist.ep_moe equiv --config lab/configs/v100_moe.json \
  --out-dir <公司内路径>/day3/gate7 --num-experts 8 --batch 2 --seq-len 64 \
  --tol 1e-5 --force-cpu --dtype float32
```

`--num-experts 8` 在命令行覆盖 config（EP=8 需要 8 个专家，约束是 `ep_size <= num_experts` 且整除），config 文件本身不改，同一份 config 服务 EP=1/2/4/8。

- 预期：打印 `[equiv] ep_size=1 vs MOEFeedForward -> PASS  max|diff|=0.000e+00`（写课程的机器上用 `moe_tiny_cpu.json` 跑出的 `max|diff|` 恰好为 0，`本机实测`）。
- 不对时先查：`max|diff|` 明显非零 → 先确认 `--force-cpu --dtype float32`（fp16 下 1e-5 的容差本来就不成立），再看 `unpermute` 那一步的加权是不是用了 top-1 的 `scores` 而不是归一化后的权重。

### 3. 第一次跑 `run_ep_scan.sh`：先用 `MAX_STEPS=2` 试线路（10 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day3/wiring NUM_EXPERTS=8 MAX_STEPS=2 bash lab/scripts/run_ep_scan.sh
```

- 预期：equiv 那一段先 `PASS`，然后四次 `torchrun`（EP=1/2/4/8）各写 2 行；header 里 `ep_size`、`ep_groups`、`experts_per_rank`、`local_expert_ids` 都有值，EP=4 时 `ep_groups` 是连续 rank 的两组。
- 不对时先查：某个 EP 规模在建组前就报错 → `ep_moe.validate_ep` 在检查 `ep_size <= num_experts`、`num_experts % ep_size == 0`、`world_size % ep_size == 0`，看哪一条不满足；如果是在通信时才挂住，说明绕过了这道检查（故障树 F5 第 4 条）。

### 4. EP=1/2/4/8 扫描，带阶段计时（35 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day3/gate7 NUM_EXPERTS=8 MAX_STEPS=10 bash lab/scripts/run_ep_scan.sh
```

脚本的 `DTYPE` 默认 `float16`，落到每条 `torchrun` 上就是 `--dtype float16`，并带 `--time-breakdown`（CUDA 上用 `torch.cuda.Event` 计时，因为 kernel 是异步下发的，`perf_counter` 量不到 all_to_all 真正的等待）。

- 预期：`<公司内路径>/day3/gate7/` 下出现 `ep1_rank<r>.jsonl` … `ep8_rank<r>.jsonl`，每个 rank 一份；每行的 `time_ms` 有九个阶段（`router`、`topk`、`count_exchange`、`permute`、`wait_before_dispatch`、`dispatch_a2a`、`expert_compute`、`combine_a2a`、`unpermute`）；EP=1 时两个 a2a 恒为 0（代码走"不发生通信"的分支）；总耗时 5–8 分钟（`估算`）。
- 不对时先查：跑到某个 EP 规模就停住、没有异常也没有新日志 → 这是破坏 I2 的典型形态（故障树 F5），进步骤 5 做对账；同时把 `NCCL_TIMEOUT_S` 压到 60 让它快点失败，而不是等默认的 600 秒。

### 5. 不变量 I2 的逐对对账（20 分钟）· `可直接执行`

对每个 EP 规模、每一对 `(r, j)`，核对 rank r 的 `input_split_sizes[j]` 是否等于 rank j 的 `output_split_sizes[r]`。两边的数在日志里本来就有，直接对账。

- 预期：所有 `(r, j)` 对都相等；每个 rank 的 `sum(input_split_sizes)` 等于本 rank 的 (token, expert) 对数（top-1 时就是 `n_tokens = batch × seq_len`）；`dropped_pairs` 恒为 0（`--capacity-factor 0` 是 dropless）；`per_local_expert_tokens` 里出现 0 是**正常**的，代码里有 `0×sum(p)` 分支保证梯度参与者一致。
- 不对时先查：某一对不相等 → 接收缓冲区的长度只能由对端告诉我，所以数据 all_to_all 之前必须先有一次元数据 all_to_all，检查 `count_exchange` 那一步是不是漏了；本地不自洽（`sum(input_split_sizes) != send.shape[0]`）会被 `ep_moe._a2a` 里的断言抓住。

### 6. 读扫描表：a2a 占比、木桶效应、负载不均（25 分钟）· `可直接执行`

对每个 EP 规模算四个抽象量：`a2a_time_fraction = (dispatch_a2a + combine_a2a) / sum(time_ms)`、`expert_compute_fraction = expert_compute / sum(time_ms)`、`wait_spread_ratio = max(wait_before_dispatch) / mean(wait_before_dispatch)`（跨 rank）、`recv_token_imbalance = max(recv_tokens) / min(recv_tokens)`（跨 rank）。

- 预期（趋势为 `估算`）：随 EP 增大，`a2a_time_fraction` 上升、`expert_compute_fraction` 下降；EP=1 时 `a2a_time_fraction` 恰为 0。负载均衡时 `recv_token_imbalance` ≈ 1、`wait_spread_ratio` 接近 1；出现热点时 `recv_token_imbalance` 趋向 E=8，且最慢那张卡的 `wait_before_dispatch` 接近 0 而其余卡很大。一个 MoE 层一步的通信是 1 次计数交换 + 4 次 `all_to_all_single`（dispatch/combine 各自的前向与反向）。
- 不对时先查：`recv_token_imbalance` 远大于 1 但 `expert_fractions` 看起来均衡 → 你算的可能是同一个 rank 不同步之间的极差而不是同一步跨 rank 的极差，先按 `step` 对齐再取 max/min。

### 7. 填证据字段（5 分钟）· `可直接执行`

对着 `02_LAB_GUIDE.md` 第 6.5 节填：`ep1_equiv_max_abs_diff`、`ep1_equiv_pass`、`split_invariant_holds`、四个 EP 各一组 `a2a_time_fraction` / `expert_compute_fraction` / `wait_spread_ratio` / `recv_token_imbalance`、`zero_token_expert_observed`。

- 预期：全部是占比、比值与布尔，没有绝对毫秒数。
- 不对时先查：想把 `time_ms` 的九个阶段原样记下来时停一下——出公司的只有占比和跨 rank 比值。

## 公司路径差异

- 步骤 3–6 在公司 8×V100 上执行；步骤 1、2 在任何 CPU 机器上都能跑，并且已在写课程的机器上通过。
- `scripts/run_ep_scan.sh` 里的 `torchrun` 命令在写课程的机器上**没有端到端跑过**（那台机器无 GPU 且 torch wheel 缺 libuv），所以步骤 3 的 `MAX_STEPS=2` 试跑是必须的一步。
- **fp16 不是 bf16**：步骤 4 的扫描由脚本的 `DTYPE=float16` 落成 `--dtype float16`；步骤 2 的 equiv 用 `--force-cpu --dtype float32`，因为 1e-5 的数值等价容差只在 fp32 下成立。
- 日志、`time_ms` 的绝对毫秒、`nvidia-smi topo -m` 矩阵全部留在公司内；离开公司机器的只有下面证据字段里的占比、比值与布尔。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M02-E-03`（一次 MoE 层前向的六步；不变量 I2 与 I3 的完整表述；一个 step 里几次 `all_to_all_single`）
2. `M02-A-01`（EP=4/DP=2 下 8 个 rank 的组归属与两类梯度的 all-reduce 组；EP=8 要先改什么）

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 2,
  "day": 3,
  "card": "day3",
  "skill": "expert_parallel_all_to_all",
  "env": "v100",
  "ai_level": "A1",
  "config_id": "v100_moe.json + --num-experts 8 + top-1 + capacity-factor 0",
  "primary_artifact": "EP=1/2/4/8 四行扫描表（a2a 占比 / expert_compute 占比 / wait 极差比 / recv 不均比）",
  "observations": {
    "ep1_equiv_max_abs_diff": "0.000e+00 | <数量级>",
    "ep1_equiv_pass": "true | false",
    "split_invariant_holds": "true | false",
    "a2a_time_fraction_ep1": 0,
    "a2a_time_fraction_ep2": "<占比>",
    "a2a_time_fraction_ep4": "<占比>",
    "a2a_time_fraction_ep8": "<占比>",
    "expert_compute_fraction_ep1": "<占比>",
    "expert_compute_fraction_ep8": "<占比>",
    "wait_spread_ratio_ep8": "<比值>",
    "recv_token_imbalance_ep8": "<比值，均衡 ≈1，热点趋向 8>",
    "zero_token_expert_observed": "true | false",
    "dropped_pairs_total": 0
  },
  "status": "PASS | FAIL-MODEL | FAIL-SYSTEM | INCONCLUSIVE",
  "failure_class": "none | numeric | system | measurement | insufficient",
  "self_check": {"M02-E-03": "能 | 不能", "M02-A-01": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "<一句话：EP 增大时占比怎么动、I2 对账是否全通过，不含绝对毫秒>"
}
```
