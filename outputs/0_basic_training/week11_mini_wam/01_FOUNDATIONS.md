# Week 11 基础篇：从零理解 Mini-WAM 与可证伪的世界监督

> 状态：教学规格，尚未在用户机器执行。本文中的阈值、耗时和显存都是预注册目标或估算，不是实测。  
> 核验日期：2026-09-03。公司 8×V100 32GB、拓扑和 PyTorch 2.1 均为用户自述待核验。

## 1. 本周真正要回答的问题

Mini-WAM 不是“再加一个会下降的 loss”。本周要构建一个动作条件的未来表征预测器，并用三个等预算实验回答：

1. `A_action_only`：只训练动作头；
2. `B_action_world`：动作头加正确动作条件的未来表征 loss；
3. `C_action_world_shuffle`：与 B 完全相同，但把动作在 batch 内错配。

若 B 只比 A 的辅助 loss 更低，却没有优于 last-frame baseline、没有优于 C、或动作/OOD 指标没有收益，便不能声称 world supervision 有用。

最终产物是：固定的数据清单、可运行的合成数据 correctness 载荷、A/B/C 配置、可恢复 checkpoint、独立评测结果和一个 `PASS-WORLD / REPRESENTATION-ONLY / FAIL-MODEL / FAIL-SYSTEM / INCONCLUSIVE` 标签。它承接 Week 09 的受控目标比较与 Week 10 的视觉动作管线，并为 Week 14 capstone 提供候选载荷。

## 2. 前置知识与环境边界

你应已能解释 supervised loss、AMP/GradScaler、global batch、DDP、episode split 和 checkpoint。不会这些时仍可完成 Lab 的 smoke，但不得把 smoke 当 mastery。

| 环境 | 本周职责 | 允许精度 | 边界 |
|---|---|---|---|
| 公司 Linux 8×V100 | 1 卡 correctness，2/4 卡受控短训；8 卡仅短 profile | FP32 小参考；FP16 AMP 正式 | 结果、日志、trace、checkpoint、数据和拓扑细节全部留公司环境；禁止上传 |
| 个人 5070 Ti | 公开/合成数据的缩小复现 | 以探针为准，可用 FP16 | 不得复制公司产物或数字；公开作品必须本机重新运行 |
| 可选 H100 | 只有预注册的规模或 BF16 对照 | FP16/BF16 需分开标记 | 不与 V100 指标混称同一硬件结果 |

V100 是 Volta SM70：支持 FP16 Tensor Core，通常没有原生 BF16 Tensor Core；官方 FlashAttention-2 CUDA 路径不支持 V100。本文用 PyTorch eager attention/小 MLP，不依赖 FA2、BF16、TF32 或 FP8。

## 3. 数据、模型与许可对象

正式实验前必须把下表写入 `manifests/sources.json`；revision 不可写成 `main`。

| 对象 | 官方来源 | 本手册冻结 revision | 许可与用途 |
|---|---|---|---|
| PushT 机器人数据（可选真实载荷） | [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) | `7628202a2180972f291ba1bc6723834921e72c19` | Hub 卡标记 MIT；使用前仍保存 LICENSE/README 并按公司数据审批 |
| DINOv2-small（可选冻结视觉 target） | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | 模型卡标记 Apache-2.0；固定 `safetensors` 和 processor |
| DINOv2 官方实现 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | 官方 README 声明代码/标准权重 Apache-2.0；特殊衍生权重需另查许可 |
| 本手册 ToyPush-v1 | Lab 中 `make_toy_data.py` | 由本地代码 SHA-256 固定 | 非敏感确定性合成 correctness 数据；不能替代真实机器人结论 |

为何提供 ToyPush-v1：它把“图片变化是否由动作引起”写进生成方程，因此可在无网络、无视频解码器时验证索引、mask、shuffle、梯度和 resume。为何仍登记 PushT：只有在公开机器人 episode 上复测，才能把结论从“代码正确”升级为“真实载荷待评估”。离线环境只接收获批的 snapshot；不得绕过网络管控。

本周主路径从零训练一个小卷积 encoder、action head 与 world head，不依赖预训练权重。DINOv2 只是可选 target encoder；若锁定的 Transformers 与 PyTorch 2.1 不兼容，保留本地 encoder 路径并把真实表征结论标为 INCONCLUSIVE。

## 4. 输入样本与时间对齐

ToyPush-v1 一个 episode 的 schema：

```json
{
  "episode_id": 7,
  "images": "uint8[T,3,32,32]",
  "states": "float32[T,4] = [x,y,vx,vy]",
  "actions": "float32[T,2] = [ax,ay]",
  "split": "train|dev|test"
}
```

窗口在时刻 `t` 取：

```text
image_t        = image[t]
state_t        = state[t]
action_chunk   = action[t : t + H_a]
image_future   = image[t + k]
valid_future   = (t + k < episode_length)
```

worked example：`B=8, T=24, H_a=4, A=2, k=4, H=W=32, D=64`。

| 边界 | shape | dtype |
|---|---|---|
| 当前/未来图像 | `[8,3,32,32]` | 输入 uint8；归一化后 float32/AMP float16 |
| 当前状态 | `[8,4]` | float32 |
| action chunk | `[8,4,2]` | float32 |
| encoder latent | `[8,64]` | 计算可 FP16；loss reduction 转 FP32 |
| 动作预测 | `[8,4,2]` | 同 action |
| future mask | `[8]` | bool，乘 loss 时转 float32 |

绝不允许 `image_future` 进入 predictor 输入，也不允许 `t+k` 跨 episode。若缓存 latent，cache key 至少包含 dataset SHA、episode、frame、encoder revision、preprocess hash。

## 5. 从输入到 loss 的完整数据流

定义冻结 target encoder `E_target`、可训练 context encoder `E_ctx`、动作头 `P` 和世界头 `W`：

\[
z_t=E_{ctx}(I_t),\quad
\hat a_{t:t+H_a}=P(z_t,s_t),
\]

\[
z_t^{*}=\operatorname{stopgrad}(E_{target}(I_t)),\quad
z_{t+k}^{*}=\operatorname{stopgrad}(E_{target}(I_{t+k})),\quad
\hat z_{t+k}=W(z_t,s_t,a_{t:t+H_a},e_k).
\]

动作均方误差按元素归一：

\[
L_{action}=\frac{1}{B H_a A}\sum_{b,h,j}(\hat a_{bhj}-a_{bhj})^2.
\]

future latent 先逐样本求平均，再用 mask 归一：

\[
\ell_b=\frac{1}{D}\lVert\hat z_b-z_b^*\rVert_2^2,\qquad
L_{future}=\frac{\sum_b m_b\ell_b}{\max(1,\sum_b m_b)}.
\]

\[
L_{total}=L_{action}+\lambda_{world}L_{future}.
\]

分母必须是有效样本数，不是固定 B。否则 episode 尾部比例改变会改变 loss 尺度。所有 reduction 用 FP32，避免 FP16 小误差下溢。

### 为什么 stop-gradient

若 target encoder 也接收 `L_future` 梯度，最容易的解可能是所有图像都映射到同一常数，`L_future≈0`，却没有世界信息。冻结 target 只能阻止这条直接塌缩路径；仍需看 target latent 方差、prediction 方差和 last-frame baseline。

### 为什么 last-frame baseline

静态场景里 `z_t` 往往天然接近 `z_{t+k}`。定义：

\[
L_{last}=\operatorname{MSE}(z_t^{*},z_{t+k}^{*}),\quad
RI=\frac{L_{last}-L_{future}}{L_{last}+\epsilon}.
\]

`RI>0` 才代表 predictor 超过“世界不动”的猜法。这里当前帧与未来帧必须经过同一个冻结 `E_target`；若拿训练中不断漂移的 `E_ctx(I_t)` 与 `E_target(I_{t+k})` 比，RI 会混入 encoder drift，失去基线含义。预注册目标 `RI≥10%` 是课程门槛，不是已知结果。

### 为什么 shuffled action 是因果对照

C 保留动作边缘分布、模型容量、优化器和额外 world head，只在训练时破坏动作与未来的配对；独立 eval 时 B/C 都必须收到同一条正确 action，不能让 C 在评测输入上额外吃亏。若 B 与 C 相同，可能是 world head 只靠 `z_t` 预测静态未来；不能说它使用了 action。训练 shuffle 必须无 fixed point，且最好跨 episode。

## 6. 最小可运行数学示例

下面只验证 mask 归一和 shuffle，不依赖 Lab 文件；可复制运行。

```python
import torch

torch.manual_seed(11)
B, D = 4, 3
pred = torch.tensor([[1., 0., 0.], [2., 0., 0.], [9., 9., 9.], [4., 0., 0.]])
target = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
valid = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
per_item = (pred.float() - target.float()).pow(2).mean(dim=1)
loss = (per_item * valid.float()).sum() / valid.sum().clamp_min(1)
perm = torch.roll(torch.arange(B), 1)
assert torch.all(perm != torch.arange(B))
assert abs(loss.item() - (1/3 + 4/3 + 3.0) / 3) < 1e-6
print({"loss": loss.item(), "perm": perm.tolist(), "valid": int(valid.sum())})
```

预计示例输出（不是实测证据）：`loss≈1.5556`、`perm=[3,0,1,2]`、`valid=3`。第三条无效样本即使误差巨大也不能进入分子；三个有效样本的逐样本 MSE 还要再除以有效样本数 3。

## 7. 算法与训练系统如何联动

### 7.1 显存

显存不只看 world head 参数：

```text
模型参数 + 梯度 + Adam 一二阶状态 + activation
+ 当前/未来两份图像 + target/context encoder activation
+ DDP bucket + CUDA workspace + allocator reserved
```

若 target 在 `no_grad()` 且冻结，其 activation 不需要为 backward 保存；若同一图像被两个 encoder 重复处理，计算和显存仍增加。优先离线 cache latent，但 cache 必须绑定 encoder/preprocess hash，不能跨版本复用。

### 7.2 计算与 I/O

视频解码、两个 frame 的预处理和两次视觉 forward 常成为瓶颈。先分别计时 `next(loader)`、H2D、encoder、world head、backward；CUDA 计时前后同步或用 event。不能把 Python 发射耗时当 GPU kernel 耗时。

### 7.3 DDP

`global_batch = micro_batch_per_rank × accumulation × world_size`。A/B/C 必须固定 global batch、successful optimizer steps 和样本 ID。教学采样器每个 global batch 无放回抽样，再按 rank 切片，因此同一步跨 rank 的 sample ID 不得重叠。DDP 默认把各 rank 梯度求和后除以 world size；如果 loss 又手工除 world size，会重复缩小。生产 DataLoader 的 sampler 每 epoch `set_epoch(epoch)`；所有 rank 若出现非有限必须用 collective 达成一致止损，不能部分 rank 更新。

### 7.4 FP16

正确顺序：`autocast(float16)` → FP32 loss reduction → `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → 跨 rank 检查梯度有限 → clip 并跨 rank 检查返回的 total norm 有限 → `scaler.step()` → `scaler.update()` → 跨 rank 核对 skip 决策。本周为保证 update clock 可审计，任一 skip 都立即失败，不把 overflow 当普通完成步。记录 attempted/successful step 与 loss scale；checkpoint 保存 scaler。

## 8. 正确性不变量：失败即停止正式训练

1. `episode(t) == episode(t+k)`，且 `t+k<T`；
2. predictor 的参数/输入图中没有 future image；
3. `z_future.requires_grad == False`；
4. B/C 除 action permutation 外 config hash 相同；
5. C 的 permutation 无 fixed point；batch=1 时禁止伪造 shuffle；
6. A/B/C 使用同一 split、初始化、global batch、successful steps 和 eval；
7. loss 分母是有效 action/target 元素数；空 mask batch 必须跳过或明确拒绝；
8. train/dev/test 按 episode 分割，相邻 frame 不跨 split；
9. checkpoint 包含 model、optimizer、scaler、每-rank RNG、attempted/successful step、resolved precision/world size 和 source/data/config hash；旁车 manifest 最后原子发布，并记录 checkpoint 字节数/SHA-256、代码 SHA、Torch 版本与合同 SHA；任何 `torch.load` 之前必须先校验该旁车，避免先反序列化损坏或错 lineage 的文件。本最小实现显式使用 constant LR、没有 scheduler，若以后增加 scheduler 就必须把其状态纳入合同；resume 只允许改变终止步数与输出目录；
10. 任何 NaN/Inf、重复/漏样、错误 revision 或未获批数据都将状态置为 `FAIL-SYSTEM`。

## 9. 诊断树

| 症状 | 最小检查 | 根因候选 | 修复 | 回归测试 |
|---|---|---|---|---|
| future loss 很快为 0 | 打印 target/pred 每维 std | target encoder 被训练、未来等于当前、全零图 | 冻结 target；修索引；检查渲染 | stop-grad、replace-future、latent-variance 三测 |
| B≈C | 检查 action grad 与 permutation | action 未接入；动作对未来无信息；k 不合适 | assert grad；改 k；运动分桶 | 固定 256 windows correct/zero/shuffle |
| B 优于 C 但差于 last | 直接算 `RI` | predictor 容量/LR 不足；归一错误 | 先小样本过拟合；核对尺度 | 同 dev、同 encoder 重算 |
| action loss 降、rollout 不升 | 检查开放环误差与分桶 | covariate shift；world loss 抢梯度 | 降 λ；看 shared grad cosine；增加 rollout | 固定 seed 的独立 rollout |
| FP16 NaN | 定位首个非有限 tensor | reduction 溢出、LR、坏图像 | reduction FP32；降 LR；过滤坏样本 | 20-step FP32/FP16 A/B 对照 |
| resume 后下一 batch 不同 | 打印 sample IDs/RNG | sampler offset/RNG 未保存 | 保存 epoch、offset、各 RNG | continuous vs resume 固定 next-batch loss |
| 多卡 loss 不等价 | 固定样本 ID/global batch | sampler、额外除 world size、accum no_sync 错 | 修采样/归一/同步 | 1/2/4 卡前 20 successful updates |

## 10. 闭卷自检与 teach-back

不看后文回答：

1. `B=8,D=64`，只有 6 个 future 有效时，为什么分母不能仍是 8？
2. 若 B 和 C 都比 last-frame 好，但 B≈C，你能声称什么、不能声称什么？
3. 写出 `global_batch=64`、4 卡、micro=4 时的 accumulation。
4. target encoder 冻结后，为什么仍可能发生“表征没有动作信息”？
5. 解释一次 FP16 overflow 时 attempted step 与 successful step 的区别。
6. 用 90 秒向工程师讲清：样本索引 → 两个 latent → 两个 loss → DDP 更新 → checkpoint → 独立 eval。

合格 teach-back 必须出现：同 episode 的 `t+k`、stop-gradient、last-frame、shuffled-action、有效 mask 分母、公平预算和“辅助 loss 下降不等于策略变好”。

## 11. 官方资料

- [LeRobotDataset 文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [lerobot/pusht 数据集](https://huggingface.co/datasets/lerobot/pusht)
- [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2)
- [DINOv2-small 模型卡](https://huggingface.co/facebook/dinov2-small)
- [PyTorch AMP 示例](https://pytorch.org/docs/2.1/notes/amp_examples.html)
- [PyTorch DDP 2.1](https://pytorch.org/docs/2.1/generated/torch.nn.parallel.DistributedDataParallel.html)

链接与 revision 于 2026-09-03 核验。实际执行时若远端 HEAD 已变化，仍使用上表 commit；若 commit 不可取，先记录新 resolved commit 和许可差异，再决定是否更新，不得悄悄跟随 `main`。
