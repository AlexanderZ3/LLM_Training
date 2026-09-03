# Week 09 基础篇：Diffusion 与 Flow Matching 的公平受控实验

> 核验日期：2026-09-03。目标不是预设胜者，而是让“objective + 对应 sampler”成为唯一主变量。所有命令/数字需实跑后才能标验证。公司 8×V100/Linux、PyTorch 2.1 与拓扑是用户自述；V100 使用 FP16，通常无原生 BF16。公司代码、Task-P、日志、trace、图片、rollout 和 checkpoint 不得导出。

## 1. 本周问题、两条实验轨与最终产物

核心问题：在相同 Task-P、模型容量、初始化规则、global batch、成功 optimizer update、有效 action windows、精度和评估条件下，diffusion epsilon prediction + DDIM 与 flow velocity prediction + Euler 的质量—延迟权衡是什么？

有两条严格区分的轨道：

- Track A（从零可执行）：本文自带 synthetic conditional-action surrogate，验证 target、mask、训练/恢复、采样和公平性。它只能得出“在该合成分布/实现中”的结论，不能冒充 MultiTask DiT 或机器人结论。
- Track B（官方集成，条件执行）：LeRobot v0.6.0 固定 commit + 获批 Task-P。该版本要求的 Python/PyTorch 高于公司 PyTorch 2.1，因此公司主环境默认 `ENV-BLOCKED`；不升级生产栈。只有组织已经提供独立、获批、兼容的现有环境才进入，且不能把公司材料带到个人环境。

最终产物：预注册表；两套 target/sampler 单测；D/F × seed 0/1 四个等预算 run；每个 checkpoint 在 5/10/20 步相同 initial noise 的 action MSE、endpoint、jerk、latency；配置 diff；恢复与 bad-mask 负测；有条件的 Task-P rollout 或明确 `INCONCLUSIVE/ENV-BLOCKED`。

## 2. 官方来源、版本、许可与硬件边界

| 对象 | 固定身份 | 许可/状态 | 用途 |
|---|---|---|---|
| LeRobot | tag `v0.6.0`，commit `30da8e687a6dfc617fcd94afc367ac7071c376ce` | Apache-2.0 | Track B 官方 MultiTask DiT 代码；公司 2.1 默认不安装 |
| LeRobot v0.6.0 runtime | `pyproject.toml` 声明 Python `>=3.12`、torch `>=2.7`（运行前再核） | 与公司 PyTorch 2.1 不兼容 | dry-run 审计必须在安装前停止冲突 |
| Task-P | 只接受组织批准的内部 revision/split/schema/license | 未提供，状态待核 | Track B；不得复制到个人机器 |
| Track A synthetic data | 本文确定性高斯条件分布 | 现场生成 | 教学/系统闭环，无外部数据 |
| DDPM / Flow Matching 论文 | DOI/arXiv 固定 | 论文各自版权 | 理论来源，不复制大段原文 |

官方一手链接：[LeRobot MultiTask DiT](https://huggingface.co/docs/lerobot/main/en/multi_task_dit)、[LeRobot 仓库](https://github.com/huggingface/lerobot)、[LeRobot v0.6.0](https://github.com/huggingface/lerobot/tree/v0.6.0)、[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)、[DDPM](https://arxiv.org/abs/2006.11239)、[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)。链接与 commit 于 2026-09-03 核验；执行前用 `git rev-parse` 与本地 `pyproject.toml` 复核。

三条资源路径：公司 V100 跑 Track A 与获批时的 Track B，但只用 FP16/PyTorch 可用原生 attention；个人 5070Ti 可公开重跑 Track A，绝不接收 Task-P/公司产物；可选 H100 只作单独环境对照。FA2 与新 SDPA backend 的可用性依硬件/版本而定，V100 不使用官方 FA2；调用 `scaled_dot_product_attention` 也不等于实际选择 flash kernel。

## 3. 统一符号，避免 `x0/x1` 反转

全篇代码只用语义名：

- `clean_actions = a ∈ R^[B,T,A]`：真实动作 chunk；
- `noise = ε ~ N(0,I)`：与动作同 shape；
- `time = t ∈ [0,1]`：`[B,1,1]`，再嵌入；
- `noisy_actions = x_t`：模型输入；
- `target`：diffusion 为 `ε`，flow 为 `a-ε`；
- `valid_mask ∈ {0,1}^[B,T,1]`：padding action 不计 loss。

有些 flow 文献把 data/noise 叫 `x_1/x_0`，有些 diffusion 语境用 `x_0` 表 clean；语义变量可消除这种命名冲突。

## 4. Diffusion：epsilon target 与确定性 DDIM 形式

为受控教学实现选择连续 cosine path：

```text
α(t) = cos(πt/2),  σ(t) = sin(πt/2)
x_t = α(t) a + σ(t) ε
target_D = ε
L_D = masked_MSE(ε_hat(x_t,t,c), ε)
```

在 `t≈1` 从噪声开始，模型给 `ε_hat`。由当前点估 clean：

```text
a_hat = (x_t - σ(t) ε_hat) / max(α(t), δ)
```

确定性 DDIM 风格下一步：

```text
x_s = α(s) a_hat + σ(s) ε_hat,  s<t
```

这是与本文 cosine 训练 path 配套的 deterministic update，不声称复现所有 DDIM schedule/parameterization。正式 LeRobot Track B 必须以其固定 commit 的 scheduler/prediction_type 为准，不能把 Track A 公式偷换进去。

## 5. Flow Matching：linear path 与 Euler ODE

取从 noise 到 data 的线性概率路径：

```text
x_t = (1-t) ε + t a
u_t = d x_t / dt = a - ε
target_F = a - ε
L_F = masked_MSE(v_hat(x_t,t,c), a-ε)
```

采样从 `x_0=ε` 开始，解 ODE `dx/dt=v_hat(x,t,c)`。显式 Euler，`K` 步：

```text
Δt = 1/K
x_(k+1) = x_k + Δt * v_hat(x_k, k/K, c)
```

减少 `K` 通常降低 latency，但离散化误差可能上升；模型误差与 solver 误差缠绕，因此要固定模型 checkpoint 再 sweep 5/10/20。

## 6. masked MSE 的单卡与分布式分母不变量

若 `error` shape `[B,T,A]`，mask `[B,T,1]`，广播后正确均值为：

```text
L = Σ_(b,t,a) mask[b,t,0] * error[b,t,a]^2
    / max(A * Σ_(b,t) mask[b,t,0], 1)
```

不能只除 `mask.sum()` 而忘记动作维 `A`；不能把 padding 位置也放进分母；mask 全零应 fail closed，不返回假 0。

DDP 下还要跨 rank 形成同一个总体均值。令 rank `r` 的局部分子、分母为 `N_r,D_r`，则报告值和正确目标都是

```text
N = all_reduce_sum(N_r)
D = all_reduce_sum(D_r)
L_global = N / D
```

绝不能先算每卡 `N_r/D_r` 再对 rank 求平均；各卡有效长度不同时，那会让短序列 rank 权重过大。由于 PyTorch DDP 在 backward 时再把参数梯度除以 `world_size=W`，每卡用于反传的标量应为 `L_r^backward = W*N_r/D`，而日志应 all-reduce 分子后记录 `N/D`。`D` 必须 detached；所有 rank 在任何 `optimizer.step()` 前共同 all-reduce `finite(loss)` 与 `finite(grad)`，一处失败则全部 fail closed，避免部分 rank 已更新。

具体例：`B=2,T=4,A=3`，有效长度 `[4,2]`，有效 scalar 数为 `(4+2)*3=18`，不是 24，也不是 6。模型预测仍是 `[2,4,3]`；mask 只改变 loss/metric，不改变 tensor shape。

## 7. 条件动作模型的 shape 路径

Track A 使用相同小 Transformer，唯一差异是 target/path/sampler：

| 张量 | 示例 shape | dtype | 说明 |
|---|---|---|---|
| condition | `[16,12]` | FP32→autocast | 合成“状态/任务”条件 |
| clean/noise/noisy action | `[16,8,6]` | FP32/FP16 | horizon 8、action dim 6 |
| time | `[16,1,1]` | FP32 | sinusoidal/MLP 后广播 |
| action tokens | `[16,8,128]` | FP16 | 输入投影 + condition + time + position |
| attention Q/K/V | `[16,8 heads,8 tokens,16]` | FP16 | `D=128,H=8` |
| predicted target | `[16,8,6]` | FP16→loss FP32 | 必须等于 action shape |
| valid mask | `[16,8,1]` | bool/FP32 | 广播动作维 |

不变量：同 seed D/F 获得相同 clean batch、noise、time 和初始网络参数；`pred.shape == target.shape == clean.shape`；全局 mask 至少一个有效元素；训练时 sampler 不参与 loss；评估 initial noise 对同一 sample/objective/step sweep 固定。正式计时窗口内不逐 step 调 `.item()`、`float(cuda_tensor)` 或 `torch.cuda.synchronize()`；D/F 走同一 telemetry 代码和同一日志频率，Profiler 另跑、不得混入质量/吞吐主结果。

## 8. 最小可运行 target 例子

以下代码无需 GPU，展示两 target 与 mask 分母；输出数值可人工复核：

```bash
python - <<'PY'
import math, torch
clean=torch.tensor([[[1.,2.],[3.,4.]]])
noise=torch.tensor([[[.5,-.5],[1.,-1.]]])
mask=torch.tensor([[[1.],[0.]]])
t=torch.tensor([[[.25]]])
alpha=torch.cos(t*math.pi/2); sigma=torch.sin(t*math.pi/2)
diff_x=alpha*clean+sigma*noise; diff_target=noise
flow_x=(1-t)*noise+t*clean; flow_target=clean-noise
def masked_mse(pred,target,mask):
    denom=mask.sum()*target.shape[-1]
    if denom.item()==0: raise ValueError('all-zero mask')
    return (((pred-target).float().square())*mask).sum()/denom
print('shapes',diff_x.shape,flow_x.shape,diff_target.shape,flow_target.shape)
print('zero-predict losses',masked_mse(torch.zeros_like(noise),diff_target,mask).item(),
      masked_mse(torch.zeros_like(noise),flow_target,mask).item())
PY
```

注意两个 zero-predict loss 数值不同不代表某 objective 更难或更好，因为 target 尺度/分布不同。跨 objective 判优必须使用共同输出空间的 action/rollout metric。

## 9. 公平性：唯一主变量与预算

预注册唯一主变量是：`objective + 与之配套的 sampler`。必须固定：

```text
data revision/split/lag/normalization/camera order
model layers/width/heads/condition encoder/trainable params
initialization seed policy + per-sample clean/noise/time
global batch + optimizer/LR/weight decay + FP16 policy
successful optimizer updates + effective valid windows seen
checkpoint selection rule + eval samples/initial noise/hardware
```

只固定 attempted steps 不够：AMP overflow 可能跳过 optimizer update，padding 比例会改变有效 windows。比较字段至少有 `attempted_steps`、`successful_updates`、`valid_windows_seen`、`skipped_updates`。

本周 A/B 的“唯一变量”是一个不可拆的处理组合：`diffusion objective + 配套 deterministic DDIM-like sampler` 对 `flow objective + 配套 Euler sampler`。这不是声称二者单项因果可分解。其余训练预算固定为每组相同 global batch、1000 次成功更新、相同全局有效 action-step/scalar 数、一次模型调用/更新；评估固定相同 held-out sample IDs、`eval_seed=91023` 生成的 initial noise、5/10/20 次模型调用/生成 chunk。成本同时报告训练成功更新数、训练模型调用、每 chunk sampler 调用、总 eval 模型调用、chunk latency 和“若 8 个动作全部消费”的摊销 latency/action。

恢复验证不是“能继续跑”即可。必须做同 seed 的 continuous `0→20` 与 split `0→10→20`：保存并恢复 model、optimizer、scheduler、GradScaler、Python/CPU/CUDA RNG、attempted/successful clock；本实验明确 `EMA=none`，若将来启用也必须入 checkpoint。两条路径还要逐 rank 固化 step 10/20 的下一批 sample IDs、noise、time，最后比较 step-20 完整状态和 fixed step-20 next-batch loss；不相同则恢复合同失败，不能用于 A/B 结论。

最低四个 run 是 D-seed0/F-seed0/D-seed1/F-seed1。单 seed 或挑各自最好 checkpoint 不能得出普遍胜负；第二 seed 未完成则结论上限为 `INCONCLUSIVE`。

## 10. 为什么训练 loss 不能横向比较

Diffusion 拟合标准噪声 `ε`，flow 拟合 `a-ε`；target 方差、时间权重与条件数不同。即使网络误差在动作空间相同，MSE 数字也可不同。可横向比较的是用各自 sampler 还原到同一动作单位后的：masked action MSE、endpoint error、jerk/smoothness、固定场景 success、latency/显存，并报告 seed 分布。

endpoint 例：`||a_hat[:,last_valid]-a[:,last_valid]||₂`。离散 jerk 可定义为三阶差分均方：

```text
jerk = mean ||a[t+3]-3a[t+2]+3a[t+1]-a[t]||²
```

它只是一种平滑度 proxy，不能替代机器人安全/成功率。

## 11. 算法与 infra 联动

| 变量 | 算法效果 | 系统效果 | 控制方法 |
|---|---|---|---|
| sampler steps 5→20 | 离散误差通常下降但非保证 | 推理近似线性变慢 | 同 checkpoint/noise/hardware；warmup 后一次总同步，报 p50/p95/调用数 |
| horizon/动作维 | 生成空间变大 | activation/输出/采样成本上升 | D/F 完全相同 |
| mask/padding | 有效监督改变 | 同样 batch 的有效样本不同 | 记录 valid scalar/windows |
| FP16 scaler skip | 某次无参数更新 | attempted step 仍耗时 | 比 successful updates |
| DDP world size | objective 不变 | 通信占比变化 | 固定 global batch；先通过 Week07 不变量 |
| Track B encoder/kernel | 表征能力改变 | 下载、显存、硬件门槛 | D/F 使用同一 encoder/backend；V100 无 FA2 |

训练通常只采样一个随机 `t` 做一次模型调用，不运行完整 5/10/20-step sampler；因此推理步数少不能推出训练吞吐更高。

## 12. 失败模式与诊断树

```text
D/F 结果差异异常大
├─ config manifest 除 objective 外不同 -> FAIL-SYSTEM
├─ clean/noise/time/sample IDs 不同 -> 随机流不公平
├─ valid denominator/windows seen 不同 -> 修 mask/预算
├─ checkpoint selection 不同 -> 按预注册 step 重评
└─ 两 seed 同方向且公共指标一致 -> 才讨论算法信号

采样 NaN/爆炸
├─ diffusion t=1 除以 α≈0 -> 使用预注册 epsilon/clamp 并一致记录
├─ flow Euler dt/方向错误 -> 检查从 noise t=0 到 data t=1
├─ normalization 未反变换/尺度错 -> 检查 train/eval stats revision
└─ FP16 算子非有限 -> FP32 tiny reference 定位，不改一组独有精度

latency 不公平
├─ 首次加载/JIT/warmup 被计入
├─ batch、硬件或同步不同
├─ sampler steps/solver 不同却未显式列出
├─ 每步 `.item()`/GPU→CPU 同步只出现在一组
└─ rendering/preprocess/telemetry 边界不同

Track B 启动失败
├─ Python<3.12 或 torch<2.7（v0.6.0）-> ENV-BLOCKED，不升级公司 2.1
├─ Task-P revision/license 未批准 -> DATA-BLOCKED
├─ BF16/FA2 hard-code on V100 -> 关闭或采用官方兼容 backend
└─ config key 与 commit 不符 -> 只查固定 commit 文档/源码
```

## 13. 结论语言与自检

允许：`PASS-DIFFUSION`、`PASS-FLOW`、`TRADEOFF`、`INCONCLUSIVE`、`FAIL-SYSTEM`。前两者要求两 seed、多数预注册共同指标和等预算均支持；质量与延迟各胜则 `TRADEOFF`；rollout/样本不足就 `INCONCLUSIVE`。

闭卷自检：

1. 写出 cosine diffusion 的 `x_t/target/a_hat` 和 linear flow 的 `x_t/target/Euler`。
2. `[B,T,A]=[2,4,3]`、有效长度 4/2 时 masked MSE 分母是多少？
3. 为什么 D/F train loss 不能直接比？共同输出空间指标有哪些？
4. attempted step 相同为何可能不等预算？
5. sampler 5/10/20 sweep 要固定哪些随机量和计时边界？
6. 训练为何通常不因 sampler steps 变少而更快？
7. 画出 Track A 的 `[16,8,6]→[16,8,128]→[16,8,6]` shape。
8. LeRobot v0.6.0 为什么在公司 PyTorch 2.1 路径默认阻塞？正确动作是什么？
9. offline MSE 更低而 rollout success 更低时怎样表述？
10. teach-back：用两分钟说明如何把一个“算法比较”变成可证伪、等预算的受控实验。
