# 第 04 周执行手册：TinyFlowPolicy——从噪声到动作的完整训练闭环

> 对应原 12 周主线第 2 周。标准投入 7.5 小时。  
> 使用 1 张 V100；FP32 小样本 reference + FP16 AMP 主训练。  
> 本周是第三只“麻雀”：继 MiniGPT 的 token 预测、nanoVLM 的图文回答后，进入连续动作生成。

每日时间盒：10 分钟闭卷推导，50–60 分钟实现核心路径，15–20 分钟测试与记录；第 2 小时只做当天的唯一可选项。

## 1. 本周主问题

你要独立实现一个 10M–50M 参数 conditional flow-matching policy：

    observation/state + noisy action chunk + time
    → predicted velocity field
    → ODE integration
    → action chunk

它必须拥有：

- 公开或合成 episode 数据；
- window、padding、mask；
- condition encoder；
- flow-matching objective；
- FP32 overfit；
- FP16 AMP；
- EMA；
- checkpoint/resume；
- offline action metric；
- toy rollout 或轨迹可视化；
- 与简单 MSE behavior cloning baseline 的比较。

本周不依赖 LeRobot 大框架。你需要先拥有一个小而完整的实现，后面才有资格比较 MultiTask DiT/SmolVLA。

## 2. 推荐数据入口

优先级：

1. 自己生成的 2D point-mass/PushT-like 合成 episode：零下载、确定性、可无限生成；
2. LeRobot 公开 PushT 数据：用于验证真实 episode schema；
3. 若网络/依赖受限，只用合成集完成本周，不引入仿真器。

官方资源：

| 资源 | 用途 | 地址 |
|---|---|---|
| Flow Matching 论文 | 理论定义 | https://arxiv.org/abs/2210.02747 |
| LeRobot 官方仓库 | 数据协议、policy 参考 | https://github.com/huggingface/lerobot |
| LeRobotDataset 文档 | episode、timestamp、delta_timestamps | https://huggingface.co/docs/lerobot/main/en/lerobot-dataset |
| Diffusion Policy 官方项目 | action chunk/conditional policy 背景 | https://github.com/real-stanford/diffusion_policy |

如果论文 PDF 无法打开，只需提前在可访问浏览器阅读公式；公司机训练不依赖论文下载。

## 3. 目录与接口

    week04-tinyflow/
      configs/tinyflow.yaml
      configs/mse_bc.yaml
      src/synthetic_env.py
      src/dataset.py
      src/condition_encoder.py
      src/time_embedding.py
      src/flow_policy.py
      src/solver.py
      src/train.py
      src/eval.py
      src/checkpoint.py
      tests/test_windows.py
      tests/test_flow_target.py
      tests/test_solver.py
      tests/test_resume.py
      runs/
      decision.md

统一 CLI：

    python -m src.train \
      --config configs/tinyflow.yaml \
      --precision fp32 \
      --max_steps 20 \
      --seed 0 \
      --output_dir runs/debug

必须支持：

    --max_steps
    --seed
    --precision fp32 or fp16
    --resume_from
    --profile_steps
    --dry_run
    --output_dir

## 4. 数学约定

本周固定最简单线性 probability path。令：

- x0：高斯噪声 action chunk；
- x1：数据 action chunk；
- t：Uniform(0,1)；
- xt = (1-t)x0 + t x1；
- target velocity ut = x1 - x0；
- 模型 vθ(xt, t, condition)；
- loss = masked mean squared error(vθ, ut)。

采样从 z0 ~ Normal(0,I) 开始，解：

    dx/dt = vθ(x, t, condition)

先用 Euler：

    x next = x + delta_t × vθ(x, t, condition)

action chunk shape 固定：

    actions: B × H × Da
    mask: B × H
    state: B × Ds
    optional image features: B × Dv
    t: B × 1

loss denominator 必须是有效元素数，不是固定 B×H×Da，否则不同 padding 比例会改变 loss scale。

## 5. 每日安排

### 周一：推导 flow target，先在 2D 分布上证明

目标：不碰机器人数据，先让数学实现可信。

闭卷推导：

1. xt 对 t 求导为什么是 x1-x0；
2. t=0 与 t=1 分别是什么分布；
3. 模型输出是 velocity、noise 还是 clean sample；
4. 训练 sampling t 与推理 ODE steps 的关系。

实现 2D toy：

- x1 为 8 个 Gaussian modes 或 moon distribution；
- x0 为标准 Gaussian；
- MLP 输入 xt、sin/cos time embedding；
- 训练 2k–5k step；
- Euler 10/50/100 steps 采样；
- 计算 sample mean/covariance 或 sliced distribution proxy；
- 用终端打印统计，不要求导出图。

单元测试：

- 人工 x0/x1/t 的 xt、target 与手算一致；
- t broadcast 不跨错维；
- batch permutation 后 loss 不变；
- target.detach()；
- solver 在恒定 velocity field 上误差接近浮点容差。

PASS：

- 2D toy loss 明显下降；
- 采样统计向数据分布移动；
- Euler steps 增加时恒定/线性 field 测试误差下降；
- 能解释 flow loss 低不等于 rollout 成功。

可选第二小时：实现 midpoint 或 RK4，只用于 solver unit test，不换主配置。

### 周二：episode、window、action chunk 和 baseline

目标：把时序数据变成没有越界的 supervised windows。

先生成 100–500 个确定性 episode：

- state：2D position/velocity 或简化 PushT state；
- goal：2D goal；
- action：delta position/force；
- timestamp/control period；
- success；
- episode_id、seed。

window 规则：

- observation at time k；
- action target k...k+H-1；
- episode 尾部 padding；
- mask 标记真实 action；
- 不跨 episode；
- train/val 按 episode 切，不按 window 随机切。

单测至少覆盖：

- episode 长度小于 H；
- 恰好等于 H；
- k 为最后一步；
- padding action 不影响 loss；
- batch 中不同有效长度；
- split 无 episode_id 重叠。

先实现 deterministic MSE BC baseline：

    condition → H×Da action chunk

在固定 128–256 windows 上过拟合。它回答数据/condition/action head 是否正确，是 flow 模型的最小 oracle。

验收：

- validator 100%；
- train/val episode overlap 为 0；
- fixed 128 windows 的 BC loss 可接近很低；
- 随机抽 5 个 window 能逐项解释；
- action normalization 统计只来自 train。

可选第二小时：加入 one-step 与 H-step 两个 BC head，比较 horizon 对误差累计的影响。

### 周三：TinyFlowPolicy、mask 与小样本过拟合

目标：实现条件 flow policy，并通过 overfit gate。

建议模型：

- state/goal MLP encoder；
- action-token linear projection；
- sinusoidal time embedding；
- 4–8 个 Transformer/MLP residual blocks；
- final velocity head；
- 10M–50M 参数。

如果使用 Transformer：

- action horizon H 作为 token 维；
- condition 可以 prepend token 或 FiLM/AdaLN；
- 明确 causal 或 non-causal：action chunk 一次性生成通常可使用双向 action-token attention；
- mask 既用于 attention，也用于 loss，但含义不同。

训练顺序：

1. FP32；
2. 固定 128 windows；
3. 固定 x0/t 作为更强 debugging；
4. 证明能过拟合；
5. 恢复随机 x0/t；
6. 再训练并采样。

必须做三种 condition 对照：

- correct condition；
- zero condition；
- shuffled condition。

如果三者无差异，模型可能忽略 condition，不得进入正式训练。

验收：

- fixed-noise/time overfit；
- random-noise/time loss 继续下降；
- correct condition 优于 shuffled/zero；
- 所有有效 action 位置有梯度；
- padding 改变不影响相同有效区域 loss。

可选第二小时：加入 classifier-free condition dropout，但只在主模型通过后。

### 周四：FP16、EMA、checkpoint/resume 与采样器

目标：把算法原型变成可靠训练任务。

FP16 规则：

    with autocast float16:
      v_pred = model(...)
    loss = masked_mse(v_pred.float(), target.float(), mask)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm
    scaler.step
    scaler.update

EMA：

- 只在成功 optimizer step 后更新；
- shadow weights 默认 FP32；
- 保存/恢复 decay、update count 和 weights；
- eval 明确选择 raw 或 EMA；
- 不让 eval 永久覆盖训练权重。

checkpoint：

- model/optimizer/scheduler/scaler；
- EMA；
- Python/NumPy/torch CPU/CUDA RNG；
- sampler/dataloader epoch；
- global step、samples_seen；
- normalization stats；
- config/data hash。

resume test：

1. 连续 40 step；
2. 20 + resume + 20；
3. step 21 fixed batch、fixed x0/t；
4. loss relative difference <1%；
5. raw 与 EMA 两套 state 都连续。

solver 对照：

- Euler 5/10/20/50 steps；
- 固定初始 noise；
- 同一 condition；
- latency 与 offline action MSE；
- 不用不同 random seed 污染比较。

验收：

- FP16 200 step 无 NaN/Inf；
- warmup 后 skipped steps <1%；
- resume gate 通过；
- EMA 与 raw eval 都能独立运行；
- 采样器在 fixed seed 下可重复。

### 周五：正式对照、toy rollout 与闭卷重建

目标：形成第一个连续控制训练报告。

正式两组：

| Run | 模型 | 预算 |
|---|---|---|
| A | deterministic MSE BC | 与 B 相同训练 windows/optimizer steps |
| B | TinyFlowPolicy | 固定 ODE steps 作为主结果 |

可选关键消融：B-shuffle，训练或评测时打乱 condition，不能新增更多网格。

指标：

- validation masked action MSE；
- endpoint error；
- trajectory jerk；
- toy rollout success；
- inference p50/p95；
- ODE steps；
- tokens/windows per second；
- peak allocated/reserved；
- final loss scale、skipped steps；
- 2 个 seed，时间不足时一个正式 seed + 明确 INCONCLUSIVE。

toy rollout 至少 100 episodes：

- 固定 ID seed set；
- 一个 OOD 轴，例如 goal 范围扩大；
- 报告 success 与 95% Wilson interval；
- 按 overshoot、no-motion、oscillation、wrong-direction 分类。

最后闭卷写出：

- x0、x1、t、xt、ut；
- masked flow loss；
- Euler sampler；
- condition path；
- EMA update；
- checkpoint 内容。

## 6. 本周硬验收

| 项目 | PASS |
|---|---|
| math | toy target/solver unit tests 全通过 |
| data | window 不跨 episode；padding mask 正确；split 按 episode |
| baseline | MSE BC 可 overfit 128–256 windows |
| flow overfit | fixed noise/time 与 random noise/time 两级 gate 通过 |
| conditioning | correct condition 优于 shuffled/zero |
| FP16 | 200 step 无非有限；skipped <1% |
| resume | next fixed batch loss 相对偏差 <1%；EMA/RNG 连续 |
| eval | BC vs Flow 同预算；ID/OOD toy rollout 与错误桶 |
| 掌控力 | 不看代码能重建 objective、sampler 与 AMP loop |

## 7. 常见故障

### flow loss 降但采样很差

检查：

- 模型预测的到底是 velocity 还是 noise；
- Euler time 顺序是否 0→1；
- delta_t 是否正确；
- normalization 是否在采样后反归一化；
- 训练 t 分布与 sampler 范围；
- EMA 权重是否正确加载。

### padding 使短 episode loss 异常

- mask 扩展到 action dimension；
- denominator 使用有效元素数；
- padding value 不应影响 valid loss；
- attention padding mask 与 loss mask 分开测试。

### condition 被忽略

- correct/shuffle/zero 三对照；
- condition encoder grad；
- condition token 是否被 attention mask 掉；
- 数据中 action 是否几乎可由时间先验预测；
- 不先加更大模型。

### resume 不一致

- 是否保存 CUDA RNG；
- sampler epoch/index；
- fixed batch 是否真的相同；
- x0/t 是否固定；
- scaler/EMA/scheduler 是否恢复；
- zero_grad 与 global step 的时点。

## 8. 本周闭卷口试

1. Flow matching 的训练 target 为什么不需要求 ODE 解？
2. 模型为什么必须以 t 为条件？
3. action chunk 与 one-step policy 的优劣是什么？
4. padding mask 为什么会改变 loss scale？
5. condition shuffle 能排除什么伪结论？
6. EMA 为什么只应在成功 optimizer step 后更新？
7. solver steps 如何影响延迟和误差？
8. behavior cloning MSE baseline 有什么诊断价值？
9. offline action MSE 与 closed-loop success 为什么可能不一致？
10. MiniGPT、nanoVLM、TinyFlow 三个 loss 的共同结构是什么？
