# Week M03 lab — MiniMind 纯 V100 全链路训练系统

> 生成日期：2026-09-08 · 生成方：cc
> 状态：**本机 CPU 门全部实测通过；所有 GPU 结论待公司 8×V100 验证。**
> 本机（Windows，conda `rfm`，Python 3.11.16，torch 2.14.0+cpu，**无 GPU**）跑完了
> CPU 单测、静态检查、2 进程 gloo 的等价/故障/分片实验；**GPU 上一行都没跑过。**

本周要回答的问题是三本账：**字节账**（每卡 param/grad/optim/activation 各多少字节）、
**误差账**（fp16 相对 fp32 从哪一步开始偏）、**通信账**（一个 step 几次集合通信、多大）。
代码全部围绕这三本账 + 三档静默故障的判别展开。

---

## 1. 先决条件与锁定版本

| 项 | 值 |
| --- | --- |
| MiniMind 仓库 commit | `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`（Apache-2.0） |
| 目标机 torch | `2.1.0`（**不升级**，不新建 conda 环境） |
| 目标机 GPU | 8×V100 32 GB，sm70：有 FP16 Tensor Core，**无 BF16、无 FlashAttention-2、无 INT8 Tensor Core** |
| 精度 | 一律 `--dtype float16` + GradScaler。`bfloat16` 会在参数解析阶段被拒绝 |
| `torch.compile` | 本周关闭 |

**MiniMind 仓库不是本 lab 的运行前提。** `lab/src/mm_v100/model.py` 自带一个与
MiniMind 形状完全一致的 dense 模型（RMSNorm + RoPE + GQA + SwiGLU + tied embedding），
所以 Day 0–5 的全部门在没有那个仓库、没有 tokenizer、没有 23 GB 数据的情况下也能跑通。
`MINIMIND_ROOT` 只在两处用到：Day 0 的兼容审计（静态扫描它的源码），
以及你想跑 MiniMind 原脚本时。

---

## 2. 两条环境路径

### 2.1 公司 8×V100（唯一执行环境）

不装任何新包。`lab/requirements.txt` 里没有一行是需要新装的——本周的量化与 RL
扩展模块也一样，全部纯 PyTorch 实现。

每次开工先设四个环境变量（或写进 `~/.bashrc`）：

```bash
export MM_DATA_ROOT=/your/path/to/minimind/datasets
export MM_WEIGHTS_ROOT=/your/path/to/minimind/weights
export MM_RUNS_ROOT=/your/scratch/minimind/runs      # 大容量可写分区
export MINIMIND_ROOT=/your/path/to/MiniMind          # 可选，只 Day 0 审计用
```

确认解析结果：

```bash
python -c "import sys; sys.path.insert(0,'lab/src'); from mm_v100 import paths; print(paths.describe())"
```

**产物留在公司内。** `$MM_RUNS_ROOT` 下的 checkpoint、日志、profile 不外带；
唯一设计成可以带出来的是 `evidence.json`（由 `lab/scripts/make_evidence.py`
从白名单字段聚合生成，不含路径、不含拓扑、不含原始数字序列）。

### 2.2 本机（写代码 + CPU 单测）

解释器固定为 conda 环境 `rfm`。所有命令都用它，不要用裸 `python`：

```bash
"C:/Users/zzz_7893/miniconda3/envs/rfm/python.exe" -m pytest lab/tests -q
```

本机没有 GPU，`torchrun` 也不可用。多进程实验一律用
`torch.multiprocessing.spawn`——各脚本的 `--launcher spawn` / `--nproc N`
就是干这件事的，所以「2 进程等价」「故障 C」「分片 checkpoint 重切」
这三件事在本机是**真跑过**的，不是纸面推导。

Windows 控制台默认 cp1252，跑带中文输出的脚本请加 `PYTHONIOENCODING=utf-8`
（脚本入口也会自己调 `use_utf8()` 兜底）。

---

## 3. 五分钟 smoke（从零开始的验证顺序）

```bash
cd outputs/1_cc_coaching/track_minimind/week03_minimind_v100_Only
export PY="C:/Users/zzz_7893/miniconda3/envs/rfm/python.exe"   # V100 上换成你的 python
export MM_RUNS_ROOT=/tmp/mm_runs                                # 本机用一个临时目录

# 1) 静态：全部源码可编译
"$PY" -m compileall -q lab/src lab/scripts

# 2) CPU 单测（本机约 8-11 分钟；含 2 进程 gloo 的真实分布式用例）
"$PY" -m pytest lab/tests -q

# 3) 探针：这台机器到底能做什么
"$PY" lab/scripts/probe_env.py

# 4) wiring smoke：world_size 逐档放大，各 2 步
"$PY" lab/scripts/wiring_smoke.py --sizes 1,2 --force-cpu --config tiny_cpu

# 5) 一步跑完某一天（首次试跑用 MAX_STEPS=2）
MAX_STEPS=2 "$PY" lab/scripts/run_day.py --day 2 --config tiny_cpu --force-cpu --nproc 2
```

在 V100 上第 4、5 步换成：

```bash
python lab/scripts/wiring_smoke.py --sizes 1,2,8 --config smoke_v100 --launcher torchrun
MAX_STEPS=2 bash lab/scripts/run_day0_probe.sh
```

---

## 4. 目录

```text
lab/
├── README.md               本文件
├── requirements.txt        零新依赖；两条路径的 API 差异写在注释里
├── configs/                四档配置
│   ├── tiny_cpu.json           CPU 单测与 dry-run（秒级）
│   ├── smoke_v100.json         V100 上 2 步跑完的 wiring smoke
│   ├── v100_768.json           Gate 主配置：hidden=768 layers=8 heads=8 kv=4
│   └── v100_768_accum.json     同一全局批、accum=4 的显存对照组
├── src/mm_v100/            主线包
│   ├── paths.py                路径合同（唯一的路径来源；不许绕过）
│   ├── console.py              UTF-8 兜底
│   ├── cli.py                  scripts/ 共用的 setup / out-dir / 写文件
│   ├── common.py               dtype 守卫、ScalerAdapter、分布式、数据切分、日志
│   ├── model.py                MiniMind 形状的自带模型 + 参数量手算公式
│   ├── probe.py                Day 0 能力探针
│   ├── compat_audit.py         Day 0 静态兼容审计（四个计数）
│   ├── data_contract.py        Day 1 数据契约与两个不变量
│   ├── numeric_ref.py          Day 1 误差账（fp32 vs fp16）
│   ├── collectives.py          Day 2/3 通信账（计数 + 字节 + 手算表）
│   ├── bounded_train.py        Day 1-5 统一有界训练（single / ddp / fsdp）
│   ├── faults.py               Day 2 三档静默故障与判别表
│   ├── ledger.py               Day 3 字节账（手算 vs 实测）
│   ├── ckpt_reshard.py         Day 3 分片 checkpoint 与跨 world_size 恢复
│   ├── eval_generate.py        Day 4 独立 eval（不看 loss）
│   └── rl_numeric.py           Day 5 DPO/GRPO 的四条解析断言
├── scripts/                入口（每个都有 argparse、--help、--dry-run）
│   ├── run_day.py              一天的全部步骤，按门的顺序
│   ├── run_day0_probe.sh …     六个 shell，每个只做一件事：用对解释器调一次 run_day.py
│   ├── probe_env.py            Day 0
│   ├── audit_compat.py         Day 0
│   ├── wiring_smoke.py         Day 0
│   ├── check_data_contract.py  Day 1
│   ├── run_numeric_ref.py      Day 1
│   ├── equiv_check.py          Day 2
│   ├── run_faults.py           Day 2
│   ├── byte_ledger.py          Day 3
│   ├── reshard_check.py        Day 3
│   ├── resume_check.py         Day 3
│   ├── train_bounded.py        Day 1-5 通用训练入口
│   ├── eval_compare.py         Day 4
│   ├── rl_numeric_check.py     Day 5
│   ├── make_evidence.py        每天最后一步
│   ├── check_data_layout.py    数据字节数校验（先前交付）
│   └── check_weights_layout.py 权重目录校验（先前交付）
└── tests/                  pytest；CPU + gloo，不需要 GPU 也不需要 torchrun
```

量化（`src/mm_quant/`）与低精度 RL（`src/mm_rl/`）两个扩展模块及其入口脚本
`run_quant_suite.*` / `run_rl_suite.*` 由另一路交付，有各自的 `test_quant_*` /
`test_rl_*` 测试；它们与主线共用 `paths.py`、`console.py` 与本目录的 conftest。

---

## 5. 逐日入口速查

| Day | 命令 | 主要产物 | 判据 |
| --- | --- | --- | --- |
| 0 | `bash scripts/run_day0_probe.sh` | `probe.json` / `compat_audit.md` / `wiring_smoke.json` | 每档 world_size 都跑满 2 步 |
| 1 | `bash scripts/run_day1_numeric.sh` | `data_contract.json` / `numeric_ref.json` | I1 违例=0；`max_dloss`、`first_diverge_step` |
| 2 | `bash scripts/run_day2_equiv_faults.sh` | `equiv_table.json` / `fault_table.json` | `max_abs_delta < 1e-4`；四档各被唯一不变量抓住 |
| 3 | `bash scripts/run_day3_ledger_reshard.sh` | `byte_ledger.md` / `reshard_check.json` / `resume_check.json` | 前三项手算==实测；reshard 与 resume 的 loss 连续 |
| 4 | `bash scripts/run_day4_sft_eval.sh` | `eval_before/after.jsonl` | eos_rate、role_leak_rate、echo_ratio（定性） |
| 5 | `bash scripts/run_day5_rl.sh` | `rl_numeric.json` | D1 ln2、D2 ref 梯度为 None、G2 ratio==1、G1 统计 |

每天最后：

```bash
python lab/scripts/make_evidence.py --day N --skill <标签> --run-tag day<N>_<...> \
    --env v100 --ai-level A2 --status PASS --time-spent <分钟>
```

生成的 `evidence.json` 是唯一设计成可以带出公司的文件。**带走之前自己再看一眼——
脚本只做机械白名单过滤，最后一道关是你。**

---

## 6. 三个概念上的坑（读代码前先看这一节）

1. **labels 的对齐约定与 MiniMind 不同。** 本 lab 的 `labels` 与 `input_ids`
   等长且对齐，移位在模型 `forward` 内部做（HF 风格）；MiniMind 的
   `lm_dataset.py` 是在数据侧就切成 `X=ids[:-1]` / `Y=ids[1:]`。两者数学等价，
   混用就会差一格——`data_contract.to_minimind_pair()` 提供显式转换，
   单测里验证了两种形式下进 loss 的 token 一一对应。

2. **CPU 上的 GradScaler 是本 lab 自己实现的。** torch 2.1 没有 CPU GradScaler，
   本机的新版 torch 有。如果两台机器各用各的，Day 2 故障 B 的本机验证就没有
   迁移价值。所以 `ScalerAdapter` 在 CPU 上一律走 `PurePythonScaler`
   （同样的 init_scale / backoff / growth_interval），只在 CUDA 上用官方实现。

3. **通信计数在 DDP 下可能少数。** DDP 的梯度 allreduce 有一部分在 C++ reducer
   里直接走 ProcessGroup，不经过 Python 的 `dist.all_reduce`。所以通信账以
   FSDP 路径为主，DDP 侧的判据是「跨 rank 梯度逐元素一致」（不变量 I3），
   不是次数。

---

## 7. 已知限制（本机验不了的部分）

| 项 | 状态 | 第一次在 V100 上要先做什么 |
| --- | --- | --- |
| NCCL 通信、`torchrun` 启动 | **未验证** | `wiring_smoke.py --sizes 1,2,8 --launcher torchrun`，失败即停在那一档 |
| FSDP1（`--placement fsdp`） | **未验证**，CPU 上直接 `SystemExit` 并说明原因 | 先 1 卡 `--placement fsdp --max-steps 2`，确认 `n_fsdp_units == layers+1` |
| FSDP 分片 checkpoint（`--format fsdp`） | **未验证** | 先把 `--format flat` 那条路径跑通（本机已验），再换 |
| fp16 的真实数值行为 | **未验证**（本机 fp16 是 CPU 软件模拟） | `run_numeric_ref.py --device cuda:0`，先看 `scale_trajectory` 是否稳定 |
| 显存与吞吐数字 | **未验证** | 8 卡非独占时 `step_time_ratio` 一律标 `measurement` 类不确定 |
| SDPA 后端实际命中 | **未验证** | `probe_env.py` 的 `sdpa.functional`，预期 `flash=False` |
| `torch._int_mm` 在 sm70 上 | **未验证** | `probe_env.py` 的 `int8.callable_on_device`，预期 `False`（这是预期结果，不是故障） |
| activation 字节的手算系数 | **估算** | `ACT_COEF_HIDDEN/INTER` 是数出来的，换 SDPA 后端会变；实测列以 `saved_tensors_hooks` 为准 |

另外两条与代码无关但会影响结论：

- **数据许可**：数据集卡片标了 `cc-by-nc-2.0`（非商用）。把这批数据放到公司机器上
  是否触及该边界，请按公司政策自行判断——cc 只标出这一条，不做法律判断。
- **`zero_std_group_ratio` 可能高到让 GRPO 无结论**：规则 reward 的值域只有 `[-3, 3]`。
  Day 5 预注册了 `INCONCLUSIVE` 出口（>0.8 时判 INCONCLUSIVE，不判 FAIL）。

---

## 8. 本机实测记录（2026-09-08）

以下是在 conda `rfm` 环境上真实执行过的：

- `python -m compileall -q lab/src lab/scripts` 通过；
- `python -m pytest lab/tests -q`：**368 passed, 6 skipped, exit code 0**，耗时 629 s
  （这个数字含量化与 RL 扩展模块的测试）。只跑主线的 13 个测试文件是
  **170 passed, 1 skipped**，耗时 461 s；那唯一一条 skip 的 reason 是
  「FSDP SHARDED_STATE_DICT 需要 CUDA；cc 的开发机没有 GPU，这条只能在公司 8xV100 上跑」；
- 2 进程 gloo：固定全局批等价（`world_size=1,accum=2` vs `world_size=2,accum=1`，
  逐步 loss 差 < 1e-4）、三档故障各被唯一不变量抓住、分片 checkpoint 2 卡写 1 卡读往返一致；
- 单进程：checkpoint/resume 逐步 loss 差 **0.0**；字节账 param/grad/optim 手算与实测**完全相等**。

**没有做过的**：任何 GPU 上的运行。文档里凡是标 `未验证` / `估算` 的都属于此类。
