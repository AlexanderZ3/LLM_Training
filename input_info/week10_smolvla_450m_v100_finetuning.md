# 第 10 周执行手册：SmolVLA 450M 在 V100 上的兼容、微调与评测

> 对应原 12 周主线第 8 周。标准投入 7.5 小时。  
> 目标：完成真实 VLA checkpoint 的 processor → dataset → forward → FP16 fine-tune → resume → eval 链路。  
> 原则：单卡正确性优先；四卡短测；8 卡只在负载合理时 profile。

每日时间盒：10 分钟写兼容/训练问题，50–60 分钟核心操作，15–20 分钟记录版本、shape 和结果；第 2 小时只做一个冻结或卡数组合。

## 1. 官方资源

| 资源 | 用途 | 地址 |
|---|---|---|
| SmolVLA 文档 | 官方训练/数据建议与命令 | https://huggingface.co/docs/lerobot/main/en/smolvla |
| SmolVLA base | 约 450M 公开 checkpoint | https://huggingface.co/lerobot/smolvla_base |
| LeRobot 仓库 | policy、processor、train/eval | https://github.com/huggingface/lerobot |
| LeRobotDataset | 数据接口 | https://huggingface.co/docs/lerobot/main/en/lerobot-dataset |
| LeRobot multi-GPU | 官方分布式训练入口 | https://huggingface.co/docs/lerobot/main/en/multi_gpu_training |

官方当前文档说明 main 需源码安装，稳定版可使用标注的 release。你的优先级不是追 main，而是锁定一个可复现 commit：

1. 先尝试公司已有/批准的 LeRobot 版本；
2. 若新建环境，选择 stable tag 或明确 commit；
3. 确认该版本与 smolvla_base processor/config 相容；
4. 记录 git SHA；
5. 12 周内不随意 pull。

## 2. V100 兼容清单

必须明确：

- precision：FP16 AMP；
- bf16=false；
- TF32/FP8 不作为路径；
- 官方 FlashAttention-2 不支持 V100，禁止安装后硬跑；
- attention 使用 PyTorch/eager/项目兼容后端；
- vision/language/action 模块的 dtype；
- gradient scaler；
- CUDA 11.8 或经验证的 12.x；
- 不升级到 CUDA 13；
- W&B/Hub push 关闭；
- 所有输出本地。

兼容性表：

| 检查 | 结果 | 处理 |
|---|---|---|
| hard-coded bf16 | | 改 config 或 fallback |
| attn implementation | | eager/compatible SDPA |
| flash-attn dependency | | 不安装/禁用 |
| torch/transformers | | 锁版本 |
| checkpoint key | | 严格检查 |
| processor/camera names | | 与 Task-P 对齐 |
| tokenizer/model downloads | | local cache/revision |

## 3. 目录

    week10-smolvla/
      compatibility.md
      configs/
        smoke.yaml
        action_expert_only.yaml
        unfreeze_top.yaml
      src/
        inspect_checkpoint.py
        inspect_batch.py
        memory_scan.py
        eval_offline.py
        compare_base_ft.py
      scripts/
        train_1gpu.sh
        train_4gpu.sh
        eval.sh
      reports/model_card.md
      reports/decision.md

## 4. 获取与离线准备

代码：

    git clone --depth 1 https://github.com/huggingface/lerobot.git third_party/lerobot-smolvla
    cd third_party/lerobot-smolvla
    git rev-parse HEAD

若决定 stable tag，clone 时指定 tag。不要同时维护 main/stable 混合依赖。

安装：

    python -m venv --system-site-packages .venv
    source .venv/bin/activate
    python -m pip install -e "third_party/lerobot-smolvla[smolvla]"

执行前先看 pip dry-run/依赖计划；若它要替换已有 torch/CUDA wheel，停止并使用受控 constraints。不要盲目升级。

模型：

    hf download lerobot/smolvla_base --local-dir artifacts/smolvla_base

数据：

- 优先第 05 周固定 Task-P；
- 若 processor 不兼容，使用官方 SmolVLA 示例公开数据；
- 先下载 metadata/小子集；
- 记录 revision/hash；
- 不能访问 Hub 时，请管理员按公司流程导入，不绕过代理。

## 5. 每日安排

### 周一：compatibility audit 与只读 forward

目标：训练前回答“这份代码在 SM70 上走哪条路径”。

任务：

1. 记录 git SHA、model revision、torch/transformers/lerobot；
2. 搜索 bf16、flash_attention、attn_implementation、torch_dtype；
3. 检查 processor、camera mapping、state/action features；
4. load checkpoint 时严格查看 missing/unexpected keys；
5. model.eval/no_grad 单条样本 forward；
6. batch 1 generate/action prediction；
7. 打印每模块参数量、dtype、device；
8. 禁用所有外部 logger/upload。

建议代码搜索：

    rg -n "bf16|bfloat16|flash_attn|flash_attention|attn_implementation|wandb|push_to_hub" third_party/lerobot-smolvla

PASS：

- base checkpoint 完整加载；
- 单样本 forward finite；
- 无 Ampere-only kernel；
- processor 输出 shape 与 checkpoint config 一致；
- action dimension/mapping 明确；
- compatibility.md 完成。

若 checkpoint 无法下载，可以随机初始化同 config 做 system smoke，但本周最终状态只能 INCONCLUSIVE。

### 周二：batch/memory scan、冻结策略与 128 样本 overfit

目标：确定安全 batch，并证明 loss path 正确。

内存扫描：

    batch 1, 2, 4, 8...

每点：

- 3 warmup + 10 measured；
- FP16 AMP；
- allocated/reserved；
- forward/backward/optimizer；
- action horizon/image count/resolution 固定；
- 达 reserved 28–30GB 停止。

主冻结策略：

    vision encoder frozen
    language backbone mostly frozen
    action expert/head trainable

可选第二组：

    action expert + top language block

先用 128–256 windows：

- 关闭增强/dropout 或固定；
- FP32 小规模 reference；
- 再 FP16；
- action loss 明显下降；
- processor train/eval normalization 一致；
- shuffled observation/instruction 对照。

PASS：

- 安全 micro-batch 确定；
- trainable parameter manifest；
- 128–256 windows 可 overfit；
- condition shuffle 影响结果；
- reserved <30.5GB。

### 周三：单卡 200–500 step FP16 微调与 resume

目标：完成最小真实 fine-tune。

主 run：

- 单卡；
- 200–500 step correctness，时间允许到 1k–5k；
- action expert/head；
- fixed global batch；
- GradScaler；
- grad norm；
- 本地 best/last；
- 每 100–250 step fixed validation；
- 无云日志。

推荐先通过 dry run，再启动：

    CUDA_VISIBLE_DEVICES=0 lerobot-train \
      --policy.path=artifacts/smolvla_base \
      --dataset.repo_id=LOCAL_OR_APPROVED_DATASET \
      --batch_size=SAFE_BATCH \
      --steps=500 \
      --output_dir=runs/single_gpu \
      --job_name=smolvla_v100_smoke \
      --policy.device=cuda \
      --wandb.enable=false

实际 flags 以锁定版本的 lerobot-train --help 为准；不要机械照抄当前网页命令。

resume：

- 100–250 step 保存；
- 正常退出；
- 从 last 恢复；
- fixed next batch loss 相对偏差 <1%；
- optimizer/scheduler/scaler/RNG/step 连续；
- processor/config 从 checkpoint 或 manifest 一致重建。

PASS：

- 500 step 或预注册短程完成；
- loss/grad/params finite；
- warmup 后 skipped <1%；
- resume 通过；
- checkpoint 可由本地 eval 独立加载。

### 周四：四卡 DDP 短测与模块 profiling

目标：验证真实 VLA 载荷下前几周的系统结论。

顺序：

1. 单卡固定 global batch reference；
2. 四卡 0–3 20-step smoke；
3. 四卡 100 timed；
4. sample IDs/rank audit；
5. loss equivalence；
6. profiler 5–10 active；
7. 若 A 组异常，在 4–7 复测；
8. 只有 E4/通信比例合理才做 8 卡 50–100 step。

profile 模块：

- image decode/preprocess；
- vision encoder；
- language/VLM backbone；
- action expert；
- loss；
- backward；
- all-reduce；
- DataLoader。

450M 模型可能在四卡后通信主导。若 8 卡低效，不要求长训；8 卡只作为 topology sample。

记录：

| cards | global batch | step p50 | samples/s | E_N | comm % | peak GB |
|---:|---:|---:|---:|---:|---:|---:|

PASS：

- 四卡 100 step；
- 参数/rank/loss 正确；
- 主瓶颈可归因；
- 8 卡是否值得由证据决定。

### 周五：base vs fine-tuned 评测、错误桶与 Model Card

目标：给出受控模型结论。

评测固定：

- 相同 Task-P dev/test；
- 相同 processor/camera order；
- 相同 action normalization；
- 相同 sampling/action horizon；
- base 与 fine-tuned；
- raw/EMA 若训练支持；
- ID + 一个 OOD axis。

最低指标：

- offline action error；
- endpoint error；
- action saturation；
- jerk；
- batch=1 inference latency p50/p95；
- simulator/toy rollout success，若可用；
- crash/invalid action rate；
- 失败类型。

若 rollout：

- 每个模型至少 30–50 episodes；
- 相同 episode seeds；
- success interval；
- simulator reset/system failures 单列。

错误桶：

- perception；
- instruction；
- no motion；
- wrong trajectory；
- timing；
- gripper；
- saturation/limit；
- recovery；
- system。

Model Card 写：

1. base/revision；
2. data/revision/split；
3. action/state/image schema；
4. compatible backend；
5. precision/freeze；
6. trainable params；
7. train/eval configs；
8. performance/peak memory；
9. base vs FT；
10. limitations；
11. 公司边界。

## 6. 本周硬验收

| 项 | PASS |
|---|---|
| compat | BF16/FA2/Ampere-only 路径排除 |
| load | checkpoint 与 processor 严格相容 |
| data | camera/state/action/normalization train-eval 一致 |
| overfit | 128–256 windows 可过拟合 |
| FP16 | 200+ step 无非有限；skipped <1% |
| memory | peak reserved <30.5GB |
| resume | model/optim/scheduler/scaler/RNG/step 完整 |
| DDP | 四卡 100 step 正确；8 卡按证据可选 |
| eval | base vs FT 受控；ID/OOD 或明确缺口 |
| report | Model Card + FAIL/PASS/INCONCLUSIVE |

## 7. 常见故障

### import 时试图编译 flash-attn

- 检查 optional dependency；
- 安装不含 flash extra 的路径；
- 配置 eager/compatible attention；
- 若仓库 commit 强依赖 Ampere-only kernel，换已验证 stable commit；
- 不编译未知 SM70 fork。

### base checkpoint 与 processor 不匹配

- model revision；
- LeRobot commit；
- feature names；
- camera order；
- action dimension；
- config missing/unexpected keys；
- 不用 strict=false 静默跳过核心 head。

### 单卡好、四卡 loss 异常

- global batch；
- sampler；
- frozen parameters/unused params；
- loss reduction；
- processor randomness；
- scaler skips；
- gradient accumulation/no_sync。

## 8. 闭卷口试

1. SmolVLA 的视觉、语言和 action expert 各负责什么？
2. 为什么 8 卡不能解决单进程不兼容的 kernel？
3. freeze、LoRA、full fine-tune 的训练状态显存差异？
4. processor 为什么属于模型正确性的一部分？
5. camera order 错了为何 loss 仍可能下降？
6. 450M 为什么四卡后可能通信主导？
7. base vs fine-tuned 必须固定哪些推理参数？
8. offline action error 与 rollout success 为什么可能矛盾？
9. strict checkpoint load 何时尤其重要？
10. 哪些结果只能标记 INCONCLUSIVE？
