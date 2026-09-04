# cc 当前状态

> 最近更新：2026-09-05 · 维护方：cc  
> 状态：`TRACK-MINIMIND-BUILT / EXECUTION-PENDING`

## 1. 当前结论

- cc 陪跑教练系统已建成：`CLAUDE.md` 契约、8 个子智能体（`.claude/agents/`）、7 个 skill（`.claude/skills/`）、4 个模板、1 个证据 schema、1 个周包校验脚本。
- 与 Codex 系统完全隔离：cc 只写 `.claude/`、`CLAUDE.md`、`outputs/1_cc_coaching/`、`memory/cc/`；共享文件只做 `[cc]` 标签级追加。
- MiniMind 两周热身轨道已构建完成，静态与 CPU 门全部通过；尚无学员证据。
- 14 周主线周包尚未生成。

## 2. 已建成的交付

| 轨道/周 | 位置 | 状态 | 结构校验 | 本机测试 |
| --- | --- | --- | --- | --- |
| MiniMind M01（单机 5070 Ti） | `outputs/1_cc_coaching/track_minimind/week01_minimind_5070ti/` | `BUILT / RUNTIME-UNVERIFIED` | PASS，口试 28/28 | `42 passed` |
| MiniMind M02（公司 8×V100） | `outputs/1_cc_coaching/track_minimind/week02_minimind_v100/` | `BUILT / RUNTIME-UNVERIFIED` | PASS，口试 27/27 | `73 passed, 1 skipped` |
| 14 周主线 Week 01–14 | `outputs/1_cc_coaching/weekNN_*/` | 未生成 | — | — |

M02 跳过的一项是分片训练必须有 CUDA，本机无 GPU。

## 3. 当前学习位置

- 当前轨道：MiniMind 热身轨道，Week M01 未开始。
- 当前天：无。Gate 状态：无。
- 尚无 A0/A1 基线、训练日志、口试答案或 debugging 证据，因此不给 K/A/D/E 或 mastery 分数。

## 4. 待闭合项

- 无（尚未产生证据）。

## 5. 开始前需要用户确认

1. 每周可投入小时数（M01 约 11 小时，M02 约 11.5 小时，均含 Gate）。
2. 个人 5070 Ti 上的驱动与 PyTorch 版本（Blackwell 需 ≥2.7 的 cu128 wheel；Day 0 探针会给出）。
3. 是否先走 MiniMind 两周热身，再进 14 周主线（当前默认是先热身）。

## 6. 唯一下一步

`/cc-day 01 0` —— 执行 MiniMind 轨道 Week M01 的 Day 0：环境探针、锁定 MiniMind commit、数据 sha256、CPU 单测。

本机可先跑的部分：

```powershell
$env:MINIMIND_ROOT = "<MiniMind 克隆路径>"
cd outputs/1_cc_coaching/track_minimind/week01_minimind_5070ti
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" -m pytest lab/tests -q --basetemp=D:\tmp\pt
```

## 7. 已知环境事实（`已确认`，2026-09-05）

- 本机 conda 环境 `ResearchAgentPy310`（Python 3.10.20，torch 2.14.0+cpu，pytest 9.1.1，transformers 4.57.6），**无 GPU**；入口 `.claude/scripts/cc_py.ps1`。
- 本机 `C:\Users\13289\AppData\Local\Temp\pytest-of-13289` 权限异常，跑 pytest 需 `--basetemp` 指向别处。
- 本机 torch CPU wheel 缺 libuv，`torchrun` 不可用；M02 的 14 个多卡运行脚本仅做过人工参数核对，首次在公司机器上跑要先 `MAX_STEPS=2`。
- MiniMind 锁定 commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`（Apache-2.0）。

## 8. 未处理的输入

- 无。`input_info/minimind_5070ti_v100.md` 已于 2026-09-05 处理完毕。
