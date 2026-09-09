# MiniMind 轨道 — 纯 LLM 全链路

> 来源：`input_info/minimind_5070ti_v100.md`（2026-09-04）+ 用户 2026-09-08 输入
> 生成方：cc · 状态见各周 `00_WEEK_CARD.md`

## 当前应该看哪个（2026-09-08 起）

**看 `week03_minimind_v100_Only/`。**

用户 2026-09-08 起不再能用个人 5070 Ti，唯一 GPU 环境是公司内网 8×V100。原来的"第一周单机 5070 Ti、第二周八卡 V100"两步走计划**作废**。Week M03 是替代品：把两周的内容重新取舍，压成一个**只在 V100 上执行**的自包含周包，另加两个扩展模块（8-bit 量化/QAT、低精度 + RL 后训练）。

| 周 | 状态 | 还要不要做 |
| --- | --- | --- |
| `week03_minimind_v100_Only/` | **当前** | **是。从这里开始** |
| `week01_minimind_5070ti/` | 已作废 | 否。硬件不存在了。保留作历史记录 |
| `week02_minimind_v100/` | 已作废 | 否。内容已被 M03 吸收并重新取舍 |

M03 **不依赖** M01/M02 的 Gate 或产物；它的 lab 代码是从那两周已通过 CPU 单测的实现里复制并裁剪的，逐文件标注了来源。

## 目录

```text
track_minimind/
├── 00_TRACK_README.md
├── week03_minimind_v100_Only/  # ← 当前。纯 V100 全链路 + 量化/RL 两个扩展模块
├── week01_minimind_5070ti/     # 已作废（5070 Ti 不再可用）
└── week02_minimind_v100/       # 已作废（内容并入 M03）
```

每周结构与 `CLAUDE.md` 第 11 节一致：周卡、Foundations、Lab Guide、6 张任务卡、口试、答案、`lab/`。M03 另有三份扩展文档（`06_QUANT_LOWBIT.md`、`07_RL_LOWPRECISION.md`、`08_INTRANET_SETUP.md`）和一套路径合同（`datasets/`、`weights/`、`runs/` 三个 README + `DATA_MANIFEST.json`）。

## 状态

| 周 | 状态 | 结构校验 | 本机测试 | Gate |
| --- | --- | --- | --- | --- |
| **M03** | `BUILT / RUNTIME-UNVERIFIED` | 见 `week03_minimind_v100_Only/00_WEEK_CARD.md` 第 9 节 | 同左 | 未开始 |
| M01 | 已作废 | （2026-09-05 曾 PASS） | （曾 `42 passed`） | 不再适用 |
| M02 | 已作废 | （2026-09-05 曾 PASS） | （曾 `73 passed, 1 skipped`） | 不再适用 |

## 校验

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .claude/scripts/validate_cc_week.ps1 -Week 03 -Track track_minimind -IdPrefix M
```

## 跑测试

本机 conda 环境是 `rfm`（用户 2026-09-08 指定，见 `CLAUDE.md` 3.1 节）：

```powershell
cd outputs/1_cc_coaching/track_minimind/week03_minimind_v100_Only
& "C:/Users/zzz_7893/miniconda3/envs/rfm/python.exe" -m pytest lab/tests -q
```

这台机器上 pytest 的默认临时目录正常，**不需要 `--basetemp`**——那是旧机器的问题，已随旧机器一起作废。

## 开始之前需要你确认

1. **每周可投入的小时数。** M03 满配约 12.3 小时（含 Gate），降级配约 9.6 小时。**默认按降级配排。**
2. **数据已拷到 V100**，且 `python lab/scripts/check_data_layout.py` 通过。
3. **许可边界。** 数据集卡片标了 `cc-by-nc-2.0`（非商用）。前两周在个人机器上无碍；M03 要把这批数据放到**公司机器**上，这条边界是否被公司用途触及，需要你按公司政策自己判断。cc 不做法律判断。

## 与 14 周主线的关系

- 本轨道不替代主线 Week 01（TinyStories MiniGPT 自建模型）；它用一个现成的、组件齐全的小仓库先建立"整条链路感"，主线随后逐层重建 ownership。
- M03 的 Gate 通过后，主线 Week 01 的 readiness 可以直接从 M03 的证据里取。
- M03 的两个扩展模块（量化 8 小时、RL 9 小时）体量上相当于额外一周，排在 M03 Gate 之后。

## 边界提醒

- **唯一执行环境是公司内网 8×V100**：不新建 conda、不升级 PyTorch 2.1、用 FP16 不用 BF16。
- 内网特性：`pip install` 与 `git clone` 可用但复杂依赖易失败，**权重必须外网下好再拷进去**。详见 `week03_minimind_v100_Only/08_INTRANET_SETUP.md`。
- 原始日志、trace、checkpoint、图片、拓扑与性能数字**留在公司内**，只带出任务卡证据字段中的抽象值。
- cc 所在的这台 Windows 机器有外网、无 GPU，交付是**开环**的：代码在这里写，在 V100 上第一次运行。
