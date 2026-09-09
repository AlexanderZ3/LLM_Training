# cc 当前状态

> 最近更新：2026-09-08 · 维护方：cc
> 状态：`WEEK-M03-BUILT / EXECUTION-PENDING`

## 1. 当前结论

- **训练环境发生根本变化（2026-09-08，用户确认）**：个人 5070 Ti 不再可用，唯一 GPU 执行环境是**公司内网 8×V100**。cc 所在的这台 Windows 机器有外网、无 GPU，交付从此是**开环**的——代码在这里写和静态验证，第一次真实运行在 V100 上。
- MiniMind 轨道的"两步走"计划（M01 单机 + M02 八卡）**作废**。替代品是 **Week M03**：只在 V100 上执行的自包含周包，另加两个扩展模块（8-bit 量化/QAT、低精度 + RL 后训练）。
- 本机 conda 环境改为 **`rfm`**（用户 2026-09-08 指令，`CLAUDE.md` 3.1 已更新）。旧机器的 `D:` 盘、`ResearchAgentPy310` 全部失效。
- 尚无学员证据。不给 K/A/D/E 或 mastery 分数。

## 2. 已建成的交付

| 轨道/周 | 位置 | 状态 | 本机测试 |
| --- | --- | --- | --- |
| **MiniMind M03（纯 V100）** | `outputs/1_cc_coaching/track_minimind/week03_minimind_v100_Only/` | `BUILT / RUNTIME-UNVERIFIED` | `368 passed, 6 skipped` |
| MiniMind M01 / M02 | 同轨道下 week01 / week02 | **已作废**，保留作历史记录 | — |
| 14 周主线 Week 01–14 | `outputs/1_cc_coaching/weekNN_*/` | 未生成 | — |

M03 包含：周卡、Foundations（约 1200 行）、Lab Guide、6 张任务卡、口试 + 答案、三份扩展文档（量化 / RL / 内网配置）、路径合同（`datasets`/`weights`/`runs`）、`lab/`（`mm_v100` + `mm_quant` + `mm_rl`，29 个测试文件）。

6 个跳过的用例全部是 `skipif(not torch.cuda.is_available())`：fake-quant 的 CUDA kernel ×2、FSDP `SHARDED_STATE_DICT` ×1、`GradScaler` 真实溢出行为 ×2、int8 ref 显存 ×1。

## 3. 当前学习位置

- 当前轨道：MiniMind M03，**未开始**。当前天：无。Gate 状态：无。
- 尚无 A0/A1 基线、训练日志、口试答案或 debugging 证据。

## 4. 构建期发现并修复的缺陷（值得记住的类型）

这些都是**本机单测全绿、但交付给用户的东西是坏的**，说明单测覆盖模块、不覆盖编排层与文档一致性：

1. `model.py` 的 `intermediate_size` 用了 Llama 的 8/3 规则（2048），注释还写着"MiniMind 的取整规则"——**注释断言了错误的出处**。MiniMind 实际是 `ceil(hidden*pi/64)*64` = 2432。且漏了 QK-norm。两者合计让参数量差 7,079,424（56.8M vs 63.9M），而 Gate 的闭卷手算题正是考这个数。
2. `run_day.py` 用合成模块名加载兄弟脚本，`mp.spawn` 的子进程 import 不到 → Day 2 / Day 3 的入口脚本 `PicklingError` 崩溃。**不是 Windows 特有，V100 上同样命中。**
3. `bounded_train` 强制放大 `num_samples`，导致 `CLAUDE.md` 第 7 节规定的 **128 样本 overfit 门根本跑不出来**。
4. Day 0 的 wiring smoke 默认走 `spawn`，而 Day 0 的门恰恰是"torchrun 能不能起来"——用 spawn 等于没验。
5. Day 1 验的是 `sft` 链但 run tag 是 `day1_pretrain`；`run_day1_numeric.sh` 用 micro=64，单卡同放 fp32+fp16 两份模型估算约 16 GB，大概率 OOM。

## 5. 待用户确认（不阻塞，但影响排期）

1. **每周可投入小时数。** M03 满配 12.3 h（含 Gate），降级配 9.6 h。**默认按降级配排。**
2. **许可边界。** 数据集卡片标 `cc-by-nc-2.0`（非商用）。前两周在个人机器上无碍，M03 要把这批数据放到**公司机器**上，这条边界是否被公司用途触及需要用户按公司政策自行判断。cc 不做法律判断。
3. 两个扩展模块合计 17 小时，体量超过主线本身，排在 M03 Gate 之后（相当于 Week M04 的内容）。

## 6. 唯一下一步

`/cc-day 03 0` —— 执行 M03 Day 0：路径合同校验 → 环境探针 → 兼容审计 → **wiring smoke（`world_size` 1 → 2 → 8，各 2 步）**。

Day 0 步骤 5 的 wiring smoke **不可裁**：cc 这边没有 GPU，`torchrun` 从没在这里跑起来过，这是全周唯一的早期闸门。跳过它，风险会一路累积到 Day 3 才爆发。

上机前先在 V100 上设四个环境变量（见 `week03_minimind_v100_Only/datasets/README.md` 第 1 节）：

```bash
export MM_DATA_ROOT=/your/path/to/minimind/datasets
export MM_WEIGHTS_ROOT=/your/path/to/minimind/weights
export MM_RUNS_ROOT=/your/scratch/minimind/runs
export MINIMIND_ROOT=/your/path/to/MiniMind
python lab/scripts/check_data_layout.py
```

## 7. 已知环境事实（`已确认`，2026-09-08）

- 本机：项目根 `C:\Users\zzz_7893\Desktop\0_Projects\LLM_Training`；conda 环境 `rfm`（`C:\Users\zzz_7893\miniconda3\envs\rfm\python.exe`，Python 3.11.16、torch 2.14.0+cpu、numpy 2.4.6、pytest 9.1.1）；**无 GPU**；`conda` 不在 PATH 上；pytest 默认 tmp 正常（不需要 `--basetemp`）；控制台 cp1252，中文输出要走 UTF-8 兜底。
- 公司 V100：8×sm_70，torch 2.1.0，**内网**。`pip install` / `git clone` 可用但复杂依赖易失败；权重必须外网下好拷进去；服务器只进不出。
- 关键版本上限：`transformers < 4.56`、`bitsandbytes <= 0.45.5`、`trl <= 0.22.2`、**`numpy < 2`**。`torchao` / `vllm` / `verl` / `OpenRLHF` **在 torch 2.1 + sm_70 上全部不可用**。
- V100 量化边界：**没有** INT8 Tensor Core、`LLM.int8()`（需 CC 7.5）；**有** fake-quant 的 CUDA kernel、8-bit 优化器、NF4/QLoRA（均 CC 6.0+）。

## 8. 未处理的输入

- 无。2026-09-08 的输入已处理并归档。
