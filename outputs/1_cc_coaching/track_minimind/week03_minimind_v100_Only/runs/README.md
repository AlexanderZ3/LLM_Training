# `runs/` — 运行产物落地位置（`MM_RUNS_ROOT`）

> 生成日期：2026-09-08 · 生成方：cc

## 这个目录在两台机器上的含义不一样

**在本仓库里**：只有这份说明和目录名，永远是空的。

**在 V100 上**：`MM_RUNS_ROOT` 指向一个大容量可写分区，装 checkpoint、日志、profile 和 eval 输出。**这些内容属于公司，留在 V100 上，不往外拷。**

不要把 `MM_RUNS_ROOT` 设成本仓库的这个目录再同步出来——那等于把公司运行产物带出公司。在 V100 上显式设一个 scratch 路径。

## 约定的子目录结构

`lab/src/mm_v100/paths.py` 里的 `run_dir(tag)` 按这个结构建目录，脚本不自己拼路径：

```text
$MM_RUNS_ROOT/
├── day0_probe/          ├── logs/  metrics.jsonl  evidence.json
├── day1_pretrain/       ├── ckpt/  logs/  metrics.jsonl  evidence.json
├── day2_sft/            └── 同上
├── day3_dist/
├── day4_profile/        └── 多一个 profiles/
├── day5_rl/
├── ext_qat/
└── ext_rl/
```

每个 run 目录下固定四样东西：

| 名字 | 内容 | 能不能带出公司 |
| --- | --- | --- |
| `ckpt/` | 权重与优化器状态 | 不能 |
| `logs/` | 原始 stdout、torchrun 日志 | 不能 |
| `profiles/` | profiler trace、`.json`、`.pt.trace.json` | 不能 |
| `metrics.jsonl` | 每步一行的标量：step、loss、lr、grad_norm、scale、tok/s、mem | 不能直接带；只带聚合后的比值 |
| `evidence.json` | 任务卡证据字段，**只含抽象值** | **能** |

`evidence.json` 是唯一设计成可以带出来的文件。它由 `lab/scripts/make_evidence.py` 从 `metrics.jsonl` 生成，生成时只做聚合（比值、是否单调、第几步、通过与否），不复制任何原始数字序列、不含路径、不含拓扑。生成后自己再看一眼再带走——**这一眼是你的责任，不是脚本的**。

## 磁盘

Day 1–3 的有界训练每次几百 MB 到几 GB。跑之前确认剩余空间：

```bash
df -h $MM_RUNS_ROOT
```

`lab/src/mm_v100/paths.py` 的 `require_free_space(gb)` 在每个训练脚本入口检查，不够就直接退出，不会训到一半塞满盘。
