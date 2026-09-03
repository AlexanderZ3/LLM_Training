# Week 08 基础篇：FSDP1、Hybrid Shard 与可恢复 checkpoint

> 核验日期：2026-09-03。目标是从张量和状态所有权推导 FSDP，而不是背开关。公司 8×V100/Linux、每卡显存、拓扑及 PyTorch 2.1 均为用户自述，运行前核验。V100 支持 FP16 Tensor Core，通常没有原生 BF16；公司环境不为本周强行升级。所有公司代码、数据、日志、trace、截图、checkpoint 和拓扑资料留在公司批准位置，禁止导出。

## 1. 本周必须回答的问题与最终产物

闭卷后应能回答：DDP、Full Shard、2×4 Hybrid Shard 分别复制/切分什么；一次 FSDP block 的 all-gather 与 reduce-scatter 在何时发生；为什么 FSDP 仍可能 activation OOM；wrap 太粗/太细怎样影响峰值与通信；分片 checkpoint 为什么要所有 rank 参与且最后写完成标志；同 world-size 恢复与换 world-size 迁移为何不是同一问题。

最终产物应含：

1. 0.6B/1.5B/4B 纸面状态预算与假设；
2. 可独立运行的约 0.6B decoder surrogate、tiny CPU 契约测试；
3. DDP、FSDP Full Shard、可用时的 2×4 Hybrid Shard 对照；
4. wrap unit 清单、显存/吞吐/collective 观测；
5. 同 world-size 的 model/optimizer/scaler/RNG/step 完整恢复；
6. incomplete checkpoint 被拒绝的负测；
7. activation checkpointing A/B；
8. 每项标为 `VERIFIED`、`PENDING`、`ENV-BLOCKED` 或 `INCONCLUSIVE`，不得把预期当实测。

## 2. 来源、版本、许可与资源路径

本周核心实验不用外部模型或数据，避免许可证和下载成为变量。token 由确定性公式生成。

| 对象 | 固定版本/身份 | 许可与用途 |
|---|---|---|
| PyTorch FSDP1 / DCP | 公司已批准 PyTorch 2.1.x，记录 patch/CUDA/NCCL | PyTorch BSD-style；使用本机对应 API，不升级去追 FSDP2 |
| synthetic tokens | 本文公式和 seed | 现场生成，无外部数据许可 |
| 教学脚本 | 实践篇全文创建并做 SHA-256 | 项目内部教学代码 |
| CS336 A2 | 阅读时固定具体 commit，实验不依赖 | MIT；仅补充系统训练思路 |
| Picotron | 阅读时固定具体 commit，实验不依赖 | Apache-2.0；不能把 H100 benchmark 当 V100 证据 |
| TorchTitan/Nanotron | 只做架构阅读，不安装/运行 main | 各自仓库许可证；现代/nightly/kernel 路径不作为公司 2.1 依赖 |

官方一手链接：[PyTorch 2.1 FSDP](https://pytorch.org/docs/2.1/fsdp.html)、[PyTorch 2.1 Distributed Checkpoint](https://pytorch.org/docs/2.1/distributed.checkpoint.html)、[activation checkpointing](https://pytorch.org/docs/2.1/checkpoint.html)、[DDP](https://pytorch.org/docs/2.1/generated/torch.nn.parallel.DistributedDataParallel.html)、[FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)、[CS336 A2](https://github.com/stanford-cs336/assignment2-systems)、[Picotron](https://github.com/huggingface/picotron)。链接于 2026-09-03 核验；实际 2.1.x 安装的签名优先。

三条路径：公司 8×V100 是 Full/Hybrid 主证据；个人 5070Ti 只做单卡/至多其真实可用卡数的代码检查，不接收公司产物；可选 H100 可做单独环境对照，不能替代 V100 结论。FlashAttention-2 官方 CUDA 路径面向 Ampere/Ada/Hopper，V100 不支持；本周使用 PyTorch 原生 attention 数学/可用后端，不编译非官方 Volta fork。SDPA 的具体 kernel 选择取决于 PyTorch/CUDA/shape，不能仅看到 API 名就宣称使用 flash kernel。

## 3. 从状态所有权理解 DDP、Full Shard、Hybrid

设模型参数数为 `P`，world size 为 `N`，Hybrid 的 shard 组大小为 `S`、replica 组大小 `R=N/S`。

- DDP：每 rank 保存完整参数、完整梯度和完整 optimizer state；反向用 all-reduce 得到相同梯度。
- Full Shard：稳定态下参数、梯度、optimizer state 理想地按 `N` 分片；进入一个 FSDP unit 前临时 all-gather 完整参数，反向后 reduce-scatter 梯度。
- Hybrid：只在大小为 `S` 的组内分片，同时在 `R` 个组间复制/同步；理想状态节省约 `S` 倍，而不是 `N` 倍。

对两个四卡岛 `{0,1,2,3}`、`{4,5,6,7}`：shard rows 是这两个集合；replica columns 是 `{0,4}`、`{1,5}`、`{2,6}`、`{3,7}`。每个状态 shard 有两个副本，所以是四路切分、两路复制。这个映射必须来自实测拓扑；编号只是候选，不是已核验事实。

### 3.1 一次 block 的时间线

```text
本地持有参数 shard
  -> pre-forward all-gather 得到该 block 完整参数
  -> forward 产生 activation
  -> 可按策略 reshard，释放完整参数
  -> pre-backward 再 all-gather 参数（若已释放）
  -> backward 产生完整梯度贡献
  -> reduce-scatter 得到本 rank 梯度 shard
  -> optimizer 只更新本地状态 shard
```

关键点：activation `[B,S,D]` 并不因参数被分片而自动按 rank 切小；长序列、较大 micro-batch 或保存过多中间值仍可 OOM。activation checkpointing 通过重算前向减少保存的 activation，代价是额外计算。

## 4. 纸面预算：先把假设写在数字前

最简 FP32 AdamW 粗估：参数 4、梯度 4、一阶矩 4、二阶矩 4，共 `16 bytes/parameter`。AMP 实现可能另有 FP16 参数副本、FP32 master、flat parameter、通信 buffer 和 allocator 碎片，因此这是状态下界，不是峰值承诺。

以 `P=0.612B` 为例：

```text
DDP state lower bound       = 16 * 0.612e9 = 9.792 GB/rank
8-way Full Shard ideal     = 9.792 / 8     = 1.224 GB/rank
2×4 Hybrid ideal           = 9.792 / 4     = 2.448 GB/rank
```

同一公式给出纸面表（十进制 GB，尚未含 activation/buffer）：

| P | DDP 状态下界 | /4 shard | /8 shard | 解释 |
|---:|---:|---:|---:|---|
| 0.6B | 9.6 GB | 2.4 GB | 1.2 GB | 本周主实验 |
| 1.5B | 24 GB | 6 GB | 3 GB | 只做预算/可选短 probe |
| 4B | 64 GB | 16 GB | 8 GB | 纸面，不承诺 V100 训练 |

显存不变量是：`peak >= persistent_shard + largest_live_all_gather + live_activations + CUDA_context + allocator_fragmentation`。若实测峰值远高于下界，不能说“FSDP 无效”，要拆分上述项。

## 5. 约 0.6B surrogate 的参数和 shape

实践篇使用 decoder-like encoder block（无因果 mask，只为训练系统载荷，不声称语言建模质量）。设词表 `V=16384`、宽度 `D=1024`、层数 `L=46`、MLP 宽 `4D`、输入/输出 embedding 不共享。

每层主导参数：

```text
Q,K,V,O projections: 4D²
MLP D->4D->D:        8D²
layer total:        ≈12D²
embedding + head:    2VD
P ≈ 12LD² + 2VD
  ≈ 12*46*1024² + 2*16384*1024
  ≈ 612.4M（加 bias/norm 后略高）
```

具体 shape，取 `B=1,S=256,D=1024,H=16,V=16384`：

| 张量 | shape | 典型 dtype | 说明 |
|---|---|---|---|
| token IDs | `[1,256]` | int64 | 每 rank 本地 micro-batch |
| embedding/hidden | `[1,256,1024]` | autocast FP16 | activation 未由 FSDP 自动分片 |
| 分头 Q/K/V | `[1,16,256,64]` | FP16 | `head_dim=64` |
| attention scores | 概念上 `[1,16,256,256]` | 实现相关 | math backend 可能显式物化 |
| logits | `[1,256,16384]` | FP16/部分 FP32 | 展开为 `[256,16384]` 算 CE |
| label | `[1,256]` | int64 | 与 logits 前两维一致 |
| loss | `[]` | 建议 FP32 reduction | 每 rank scalar，需明确定义全局聚合 |

shape 不变量：`D % H == 0`；logits 最后一维等于 `V`；目标落在 `[0,V)`；所有 wrap unit 参数计数加上未包裹根参数应覆盖模型参数且不重复。

## 6. 最小可运行概念例子

下面只验证 FSDP 的调用协议；需要两张可见 GPU 和现有 PyTorch，不能证明正式配置性能。代码是完整的，可复制到空目录运行：

```bash
mkdir -p /tmp/fsdp-concept
cat > /tmp/fsdp-concept/min_fsdp.py <<'PY'
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local)
dist.init_process_group('nccl')
torch.manual_seed(7)
model=nn.Sequential(nn.Linear(32,64), nn.GELU(), nn.Linear(64,32)).cuda(local)
model=FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=local)
opt=torch.optim.AdamW(model.parameters(), lr=1e-3)
x=torch.arange(128, device=f'cuda:{local}', dtype=torch.float32).reshape(4,32)/128
loss=model(x).float().square().mean()
loss.backward(); opt.step()
value=loss.detach(); dist.all_reduce(value); value/=dist.get_world_size()
if rank==0: print({'shape': list(x.shape), 'global_loss': float(value)})
dist.destroy_process_group()
PY
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 /tmp/fsdp-concept/min_fsdp.py
```

输出 loss 只能在实跑后记录。若 2.1.x 的构造签名不同，以本机文档/`inspect.signature(FSDP)` 为准，状态记 `ENV-BLOCKED`，不升级生产栈。

## 7. wrap policy：性能问题也是正确性问题

以 Transformer block 为 unit 通常是合理起点：unit 太大，完整参数 all-gather 的瞬时 buffer 大、重叠机会少；unit 太小，collective 数量碎片化、CPU launch/延迟变大。还要处理：共享/绑定权重不能被不兼容地拆开；冻结与可训练参数混合在某些 FSDP1 配置受限；模型在 wrap 后再搬设备或创建 optimizer 会改变语义。

必须在 wrap 前按参数 identity 建立归属：FSDP unit 全名、原始参数量、trainable 数、最大/最小 unit、根部未包裹参数、总覆盖数；随后核对实际 FSDP wrapper 数。wrap 后 flat shard 的 local `numel` 不是原始覆盖证明。验收不是“看到了 FSDP 字样”，而是唯一归属的 union 等于原模型参数集且无重复。

## 8. 数值等价与全局 loss

若各 rank local loss 是样本均值，且 local batch 等大，则展示用全局均值：

```text
L_global = (1/N) Σ_r L_r
```

FSDP/DDP 的梯度同步由框架完成；不能把仅用于日志的 `all_reduce(loss)` 再参与 backward，否则改变梯度。等价实验固定初始化、全局样本 ID、global batch、optimizer、precision、step 次序；先在 tiny FP32 比较一/数步，再做 FP16 趋势比较。FP16 不应以逐 bit 相等为门槛，但非有限、持续漂移或 rank divergence 都必须停。

AMP 顺序不变量：`autocast -> scaled backward -> unscale -> finite/clip -> optimizer step -> scaler update`。V100 不开启 BF16；不要用 `model.half()` 代替 AMP。

## 9. checkpoint 是一个分布式提交协议

可恢复训练至少包含：

```text
model shard + optimizer shard + scheduler + GradScaler
+ per-rank CPU/CUDA/Python RNG + sampler/global sample offset
+ completed step/tokens + model/data/config revision
+ strategy/wrap/world size/torch version
```

保存阶段所有 rank 以相同调用顺序参加 DCP；各 rank 写各自合法 shard，不可同时覆盖一个普通文件。先写同文件系统的 step-versioned 临时目录，barrier 后生成全文件 hash manifest，原子发布该目录，最后写含格式版本、精确 torch patch 与 manifest hash 的 `COMPLETED`。loader 先由 marker 做版本门，再校验 manifest/文件 hash，任一不符都在 DCP load 前拒绝。

同 world-size resume 要求用独立 uninterrupted control 对照恢复后下一固定 batch 的 loss、model、optimizer、step、scaler 与每-rank RNG；只看到 step 接续不算等价。换 world-size 是迁移：是否支持 optimizer reshard 取决于本机 PyTorch 2.1 API；不能手工拼大型 shard。若版本不支持，明确 `ENV-BLOCKED`，可只在 tiny 模型通过官方 full-state 路径验证。

`torch.distributed.checkpoint` 在版本间的 state-dict 兼容不应默认保证；保存和恢复都记录精确 torch 版本，普通 resume 在任何 DCP I/O 前必须拒绝 patch 不同。跨版本只能走单独、显式验证的 migration。Hybrid Shard 下 DCP 的 process group 语义也要以 2.1 文档/签名核对，不能把所有 replica 重复写同一 shard。

## 10. 算法与 infra 联动

| 变化 | 算法/内存结果 | infra 结果 | 必测证据 |
|---|---|---|---|
| DDP→Full Shard | 状态占用下降 | all-gather/reduce-scatter 增多 | optimizer 初始化后峰值、collective timeline |
| Full→2×4 Hybrid | 每卡状态约翻倍于 8-way ideal | 高频通信限制在四卡组、跨组复制同步 | 真实拓扑、group membership、吞吐 |
| wrap 变小 | live full-param buffer 可能下降 | collective 数增多 | unit 清单、事件数、step time |
| activation checkpoint on | activation 峰值下降 | 重算使 backward 更长 | 同 batch/seq A/B |
| sequence 增长 | attention/activation 增长 | kernel/通信占比变化 | shape、峰值、OOM 首现场 |
| checkpoint 频率上升 | 算法不变 | I/O idle 增加 | steady-state 与端到端吞吐分开 |

正式性能的 `steady_tokens_per_s` 必须来自 profiler-off 且窗口内无日志/保存的预热后 update；profile trace 只解释 collective/重算，不能作为 activation-checkpoint A/B 的速度值。

## 11. 失败模式与诊断树

```text
启动失败/hang
├─ CPU Gloo collective 也失败 -> launcher/rank/端口
├─ Gloo 过、NCCL 失败 -> 驱动/NCCL/可见卡/拓扑
└─ 只 Hybrid 失败 -> 组创建顺序、2.1 API/tuple process_group；标 ENV-BLOCKED

OOM
├─ optimizer 首步暴涨 -> model/grad/Adam 状态主导；检查 sharding 生效
├─ 长序列前向暴涨 -> activation/attention；checkpoint/减 seq
├─ 某 unit 前瞬时峰值 -> wrap 太粗/all-gather buffer
└─ 保存时暴涨 -> 意外 full state dict/CPU-GPU 副本

loss 与 DDP 不一致
├─ global batch/sample ID 不同
├─ loss reduction 或累积除数不同
├─ init/RNG/dropout 不同
├─ AMP scaler skip/clip 顺序不同
└─ shared weight/wrap/use_orig_params 语义不同

resume 失败
├─ 无 COMPLETED -> 正确拒绝，不修补半成品
├─ shard/metadata 缺失 -> 回到上一个完整 checkpoint
├─ world/strategy/wrap 改变 -> 这是迁移，不是 resume
└─ step 对、next loss 错 -> optimizer/scaler/RNG/sample offset 未恢复
```

止损：任一非有限参数/梯度；连续 scaler 回退且伴随非有限梯度；rank hang 超过预设 timeout；checkpoint 缺项；为“跑通”而静默换精度、数据、模型或 world size。

## 12. 自检与 teach-back

闭卷作答：

1. 为什么 2×4 Hybrid 在 8 卡上只有四路状态分片？画出两类 process group。
2. 0.612B、16 bytes/param 时 DDP、/4、/8 的状态下界分别是多少？哪些峰值没算？
3. 给出 `[1,256,1024]` 隐状态、16 头的 Q/K/V 概念 shape；哪一项 FSDP 不会自动切分？
4. 为什么 optimizer 第一次 step 之后才适合比较状态峰值？
5. wrap 太粗和太细的 profiler 特征各是什么？
6. 日志用的 all-reduced loss 为什么不能再参与 backward？
7. 一个可恢复 checkpoint 必须有哪些状态？为什么 `COMPLETED` 最后写？
8. 8 卡保存、4 卡加载为何是迁移而非普通 resume？
9. 为什么调用 SDPA 不等于使用 FlashAttention-2？V100 的边界是什么？
10. teach-back：用两分钟解释“FSDP 解决状态复制，但不自动解决 activation、I/O 与拓扑问题”。

能准确说出所有权、shape、公式、不变量、版本条件与证据，才算掌握；只会复述“FSDP 更省显存”不算。
