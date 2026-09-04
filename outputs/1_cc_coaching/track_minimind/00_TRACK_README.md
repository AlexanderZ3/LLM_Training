# MiniMind 轨道（两周）— 纯 LLM 全链路热身

> 来源：`input_info/minimind_5070ti_v100.md`（用户意图，2026-09-04）  
> 生成方：cc · 状态见各周 `00_WEEK_CARD.md`  
> 定位：放在 14 周主线之前的热身轨道。第一周在个人 5070 Ti 上用 MiniMind 把 tokenizer → pretrain → SFT → LoRA → DPO → GRPO 的因果链亲手串起来；第二周在公司 8×V100 上把同一个模型改造成多卡训练系统实验台：DDP/FSDP、MoE 负载不均衡与 EP、GRPO 多角色拆分、恢复。

## 目录

```text
track_minimind/
├── 00_TRACK_README.md
├── week01_minimind_5070ti/     # 单机 5070 Ti 16GB：全链路 + 自写仪表
└── week02_minimind_v100/       # 公司 8×V100：DDP/FSDP/MoE-EP/GRPO 角色拆分/恢复
```

每周结构与 `CLAUDE.md` 第 11 节一致：周卡、Foundations、Lab Guide、6 张任务卡、口试、答案、`lab/`。

## 状态（2026-09-05）

| 周 | 状态 | 结构校验 | 本机测试 | Gate |
| --- | --- | --- | --- | --- |
| M01 | `BUILT / RUNTIME-UNVERIFIED` | PASS，0 error 0 warning；口试 28 题/28 答 | `42 passed` | 未开始 |
| M02 | `BUILT / RUNTIME-UNVERIFIED` | PASS，0 error 0 warning；口试 27 题/27 答 | `73 passed, 1 skipped` | 未开始 |

M02 跳过的一项是分布式分片训练必须有 CUDA，本机无 GPU 无法验证。

## 校验

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .claude/scripts/validate_cc_week.ps1 -Week 01 -Track track_minimind -IdPrefix M
powershell -NoProfile -ExecutionPolicy Bypass -File .claude/scripts/validate_cc_week.ps1 -Week 02 -Track track_minimind -IdPrefix M
```

跑测试（本机 conda 环境，需先设 `MINIMIND_ROOT` 指向锁定 commit 的克隆）：

```powershell
$env:MINIMIND_ROOT = "<你克隆 MiniMind 的路径>"
cd outputs/1_cc_coaching/track_minimind/week01_minimind_5070ti
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" -m pytest lab/tests -q --basetemp=D:\tmp\pt
```

`--basetemp` 是必须的：本机 `C:\Users\13289\AppData\Local\Temp\pytest-of-13289` 权限异常，不指定会有 11 个用例报 `PermissionError [WinError 5]`。

## 开始之前需要你确认

1. 每周可投入的小时数（M01 约 11 小时，M02 约 11.5 小时，含 Gate）。
2. M01 的 5070 Ti 上 PyTorch 与驱动版本（Day 0 探针会给出，Blackwell 需要 2.7 以上的 cu128 wheel）。
3. M02 在公司机器上第一次跑任何多卡脚本时，先用两步试跑，因为本机的多进程启动器不可用，这些命令行只做过人工核对。

## 与 14 周主线的关系

- 本轨道不替代主线 Week 01（TinyStories MiniGPT 自建模型）；它用一个现成的、组件齐全的小仓库先建立“整条链路感”，主线随后逐层重建 ownership。
- 本轨道的 Gate 通过后，主线 Week 01 的 readiness 可以直接从本轨道证据里取。

## 边界提醒

- Week 01 全部在个人机器，用公开数据与公开代码。
- Week 02 在公司机器：不新建 conda 环境、不升级 PyTorch 2.1、用 FP16 不用 BF16；原始日志、trace、checkpoint、图片、拓扑与性能数字留在公司内，只带出任务卡证据字段中的抽象值。
