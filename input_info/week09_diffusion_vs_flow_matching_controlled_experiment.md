# 第 09 周执行手册：MultiTask DiT——Diffusion 与 Flow Matching 公平对照

> 对应原 12 周主线第 7 周。标准投入 7.5 小时。  
> 载荷：固定 Task-P、相同 DiT、相同数据与训练预算。  
> 本周目标：把训练系统能力用于一个真正算法问题，同时杜绝“比较了十个变量”的伪消融。

每日时间盒：10 分钟写预注册假设，50–60 分钟完成当天主实验，15–20 分钟核对不变量；第 2 小时只补一个 seed 或 sampler 点。

## 1. 官方资源

| 资源 | 用途 | 地址 |
|---|---|---|
| LeRobot MultiTask DiT 文档 | 官方同时支持 diffusion/flow matching | https://huggingface.co/docs/lerobot/main/en/multi_task_dit |
| LeRobot 仓库 | 训练代码与 policy 配置 | https://github.com/huggingface/lerobot |
| Flow Matching | 目标函数基础 | https://arxiv.org/abs/2210.02747 |
| Denoising Diffusion | diffusion 基础 | https://arxiv.org/abs/2006.11239 |
| Diffusion Policy | 动作 diffusion 背景 | https://github.com/real-stanford/diffusion_policy |

先使用你第 04 周 TinyFlow 代码重建两个 toy objective，再接 LeRobot。官方文档的推荐超参数不是本周结论，必须在相同预算下比较。

## 2. 预注册实验

唯一主变量：

    objective + 对应 sampler

固定不变：

- Task-P revision/split/lag/normalization；
- model layers/hidden/heads/condition encoder；
- trainable 参数量；
- initialization seed policy；
- global batch；
- optimizer、LR schedule、weight decay；
- 有效训练样本数或 tokens/windows seen；
- precision FP16；
- gradient accumulation；
- checkpoint/eval 频率；
- 两个 seeds；
- image/action preprocessing。

主实验：

| Run | objective | train target | inference |
|---|---|---|---|
| D | diffusion | epsilon 或固定 prediction_type | DDIM 10 steps 主结果 |
| F | flow matching | velocity | Euler 10 steps 主结果 |

采样步数 5/10/20 是评测 sweep，不得让训练预算不同。

## 3. 目录

    week09-objectives/
      theory/objective_notes.md
      configs/common.yaml
      configs/diffusion.yaml
      configs/flow.yaml
      src/objective_reference.py
      src/eval_offline.py
      src/eval_rollout.py
      src/compare_runs.py
      tests/test_diffusion_target.py
      tests/test_flow_target.py
      tests/test_sampler_determinism.py
      reports/pre_registration.md
      reports/decision.md

## 4. 每日安排

### 周一：在相同符号系统中推导两个 objective

目标：清楚模型在每个 objective 下预测什么。

写一张对照表：

| 项 | Diffusion | Flow |
|---|---|---|
| clean data | x0 或 x_data，统一你的记号 | x1 |
| noise | epsilon | x0 |
| noisy/interpolated sample | xt | xt |
| time | discrete/continuous | continuous |
| target | epsilon/sample/v，固定一种 | x1-x0 |
| loss | masked MSE | masked MSE |
| sampler | DDIM | ODE Euler |

最容易出错的是不同论文对 x0/x1 的命名相反。你的代码变量必须使用 clean_actions、noise、noisy_actions、target，避免只写 x0/x1。

任务：

1. 在人工 tensor 上手算两个 target；
2. 写 reference functions；
3. 用固定 model 输出验证 masked loss；
4. 在 2D toy distribution 上各训练 1k–3k step；
5. 相同 network capacity/updates；
6. 生成固定 seed samples；
7. 验证 sampler determinism。

PASS：

- 两套 target 单测；
- mask denominator 正确；
- 两套 toy 都能学；
- 能解释 train loss 数值不能直接跨 objective 比大小；
- pre_registration.md 完成。

### 周二：LeRobot 兼容审计与 Task-P adapter

目标：在正式训练前消除版本/数据隐患。

固定 LeRobot revision：

    git clone --depth 1 https://github.com/huggingface/lerobot.git third_party/lerobot
    cd third_party/lerobot
    git rev-parse HEAD

若已在第 05 周克隆，继续使用同 commit，不 pull main。安装以该 commit 的官方要求为准，并复用现有 torch。显式关闭：

- BF16；
- W&B；
- Hub push；
- FlashAttention-2；
- 未验证 compile/fused kernel。

检查：

- policy 是否支持 objective=diffusion/flow_matching；
- config key 是否与当前 commit 一致；
- vision/text encoder 是否需要网络下载；
- Task-P field/camera/action shape；
- horizon/n_action_steps 与 control rate；
- normalization；
- resume；
- output 仅本地。

先 1 卡各跑：

    1 step dry run
    20 step smoke
    100 step correctness

每个 batch 打印一次：

- images/state/instruction/action shape；
- valid action mask；
- objective target mean/std；
- trainable params；
- loss finite；
- peak memory。

若完整 CLIP encoder 下载/兼容受阻，使用冻结的本地可用视觉特征或小 surrogate；两组必须完全相同，并把结论限定为 objective 系统练习。

### 周三：两个 seed 的等预算主训练

目标：跑完预注册的 D/F 主配置。

最低四个 run：

    D seed0
    F seed0
    D seed1
    F seed1

如果时间只能支撑两个长 run：

- 四组先完成 300–500 step correctness；
- 主 seed 跑较长；
- 第二 seed 跑预注册短预算；
- 结论标记证据强度，不用单 seed 宣称优胜。

等预算优先用：

    effective training windows seen

而不是只看 steps，因为 batch/skip 可能不同。记录：

- attempted optimizer steps；
- successful optimizer steps；
- samples/windows seen；
- skipped AMP steps；
- wall time；
- tokens/windows per second；
- peak memory；
- validation fixed-seed loss/metric。

多卡选择：

- 主训练先单卡或四卡 0–3；
- 450M 以内且 global batch 够大才短跑 8 卡；
- 先通过第 07 周 DDP invariant；
- 不为了“用满 8 卡”改变 global batch。

PASS：

- 四个 run 都有有效 checkpoint/eval；
- FP16 稳定；
- 实际 samples seen 差异 <1% 或得到校正；
- objective 外无非预期 config diff。

### 周四：采样步数、延迟与短 DDP profile

目标：把生成质量与实时成本放到同一张表。

每个 checkpoint 在固定 validation conditions 和固定 initial noise 上评：

| objective | steps | solver | action MSE | endpoint | jerk | latency p50/p95 |
|---|---:|---|---:|---:|---:|---:|
| diffusion | 5 | DDIM | | | | |
| diffusion | 10 | DDIM | | | | |
| diffusion | 20 | DDIM | | | | |
| flow | 5 | Euler | | | | |
| flow | 10 | Euler | | | | |
| flow | 20 | Euler | | | | |

规则：

- 同硬件、batch=1；
- warmup；
- CUDA event；
- 固定 100+ samples；
- 不把第一次权重加载算推理延迟；
- action normalization 反变换一致；
- latency 包含 sampler loop，不含环境 rendering。

再用 2/4/8 卡各 50–100 training steps，比较两 objective 的：

- forward/backward；
- communication ratio；
- peak memory；
- step time；
- 是否因 sampler 不同影响训练；通常训练不运行完整 sampler。

不要用 inference steps 减少推断出训练吞吐一定更高。

### 周五：offline、rollout、错误桶与结论

目标：只根据证据作结论。

最低评测：

- validation masked action error；
- endpoint error；
- action smoothness/jerk；
- toy/simulator rollout ID；
- 一个 OOD axis；
- inference latency；
- 2 seeds；
- 失败类型。

rollout 若可用：

- 每配置至少 30–50 episodes；时间允许到 100；
- 相同 episode seed；
- success 置信区间；
- 环境 reset failure 单独统计；
- 系统 crash 不计为模型失败，但必须报告。

错误桶：

- no-motion；
- wrong direction；
- overshoot；
- oscillation；
- contact/timing；
- gripper/action saturation；
- recovery failure；
- environment/system。

决策语言：

- PASS-DIFFUSION：同预算、两 seed 多数指标支持 diffusion；
- PASS-FLOW：同预算、两 seed 多数指标支持 flow；
- TRADEOFF：质量/延迟各有优势；
- INCONCLUSIVE：方差、样本或 rollout 不足；
- FAIL-SYSTEM：不能形成公平对照。

禁止：

- 用最低 train loss 判优；
- 挑每个 objective 最好 checkpoint 而忽略 selection rule；
- 不同 sampling steps 却声称模型本身更快；
- 单 seed 结果写成普遍规律。

## 5. 本周验收

| 项 | PASS |
|---|---|
| 理论 | 两 target 与 sampler reference tests |
| 公平性 | objective 外主要 config 完全一致 |
| 预算 | samples seen/successful updates 差异 <1% |
| 稳定 | 两 objective、两 seed FP16 均可训练 |
| 评测 | steps 5/10/20 的质量-延迟表 |
| rollout | ID + 一个 OOD 轴，或明确环境不可用 |
| 统计 | 不用单次最好值；报告 seed 方差 |
| 系统 | 单卡正确后才多卡；8 卡非强制 |
| 结论 | PASS/TRADEOFF/INCONCLUSIVE 与证据匹配 |

## 6. 闭卷口试

1. Diffusion epsilon target 与 flow velocity target 分别是什么？
2. 为什么两者 train loss 数值不能直接比较？
3. 等 steps 为什么可能不是等训练预算？
4. sampling steps 对延迟和误差的影响是什么？
5. DDIM 与 Euler 各在积分什么？
6. 相同 initial noise 为什么重要？
7. offline action error 与 rollout success 为什么可能相反？
8. objective 对照中哪些变量必须固定？
9. 为什么需要至少两个 seed？
10. 什么结果只能叫 TRADEOFF 而不能叫某算法胜出？
