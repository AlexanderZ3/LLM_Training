# outputs/1_cc_coaching — cc 陪跑教练交付区

> 本目录只由 Claude Code（cc）写入；Codex 的教程在 `outputs/0_basic_training/`，两者互不覆盖。

## 目录约定

```text
outputs/1_cc_coaching/
├── 00_README.md                     # 本文件
├── NN_YYYY-MM-DD_cc_<topic>.md      # 非周包类交付（核验报告、规划、审查）
├── track_minimind/                  # MiniMind 两周热身轨道（见其 00_TRACK_README.md）
└── weekNN_<slug>/                   # 14 周主线周包，结构见 CLAUDE.md 第 11 节
    ├── 00_WEEK_CARD.md
    ├── 01_FOUNDATIONS.md
    ├── 02_LAB_GUIDE.md
    ├── 03_TASK_CARDS/day0.md … day5.md
    ├── 04_ORAL_EXAM.md
    ├── 05_REFERENCE_ANSWERS.md
    ├── 06_GATE_RECORD_YYYY-MM-DD.md  # 口试后生成
    └── lab/                          # 可运行代码
```

## 使用方式

1. `/cc-build-week NN` 生成周包；
2. `/cc-day NN D` 按天陪跑；
3. 完成一张卡后用 `/cc-evidence` 归档证据；
4. 周末 `/cc-gate NN`；
5. 迷路时 `/cc-status`。

## 状态

### MiniMind 热身轨道（`track_minimind/`）

| 周 | slug | 环境 | 周包状态 | 本机测试 | Gate |
| ---: | --- | --- | --- | --- | --- |
| M01 | minimind_5070ti | 个人 5070 Ti | `BUILT / RUNTIME-UNVERIFIED` | 42 passed | — |
| M02 | minimind_v100 | 公司 8×V100 | `BUILT / RUNTIME-UNVERIFIED` | 73 passed, 1 skipped | — |

### 14 周主线

| 周 | slug | 周包状态 | Gate |
| ---: | --- | --- | --- |
| 01 | pure_llm_minigpt | 未生成 | — |
| 02 | pure_vlm_nanovlm | 未生成 | — |
| 03 | v100_fp16_topology | 未生成 | — |
| 04 | tinyflowpolicy | 未生成 | — |
| 05 | data_contract_repro | 未生成 | — |
| 06 | single_gpu_profiling | 未生成 | — |
| 07 | ddp_scaling | 未生成 | — |
| 08 | fsdp_checkpoint | 未生成 | — |
| 09 | diffusion_vs_flow | 未生成 | — |
| 10 | smolvla_finetune | 未生成 | — |
| 11 | mini_wam_ablation | 未生成 | — |
| 12 | finexec_sft_cpt | 未生成 | — |
| 13 | finexec_scale_peft_grpo | 未生成 | — |
| 14 | capstone | 未生成 | — |
