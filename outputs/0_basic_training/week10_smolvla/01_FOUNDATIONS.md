# Week 10 基础篇：SmolVLA 的 processor、Flow Action Expert 与兼容边界

> 核验日期：2026-09-03。目标是理解并审计真实 VLA 链路，不是把 Hugging Face 命令当黑盒。公司 8×V100/Linux、PyTorch 2.1、32 GB/卡与拓扑为用户自述，运行前核验。V100 支持 FP16、通常无原生 BF16。公司数据、代码、日志、trace、图片/视频、checkpoint、性能结果和拓扑不得导出。

## 1. 先给可执行结论：三道硬门

本周正式 SmolVLA 结果需同时过三道门：

1. **runtime 门**：锁定 LeRobot v0.4.3（commit `0b067df57d21d3a02d6c511f1609172fa39ac29b`，Git tree `b5891d6333c4eb611bd3e81a4d13ed3c71210a6e`）要求 `torch>=2.2.1,<2.8.0`；公司 PyTorch 2.1 不满足。该 commit 的 `smolvlm_with_expert.py` 还把嵌套 VLM 加载 dtype 写成 BF16，V100 通常无原生 BF16。因此公司 V100 主路径是双重 `ENV-BLOCKED`，不升级环境、也不把未经评审的 dtype 改写冒充官方结果。个人 5070Ti 或组织已有的独立批准兼容 GPU 环境才可继续。
2. **权重许可门**：`lerobot/smolvla_base` 模型卡在核验时未声明 license。LeRobot 代码的 Apache-2.0 不自动覆盖模型权重；在组织/法务明确批准前，不下载、不训练、不发布衍生 checkpoint。
3. **数据/processor 门**：公开数据 `lerobot/svla_so101_pickplace` 是 Apache-2.0，但它的 camera key 是 `up/side`，base config 期望 `image/image2/image3`；必须固定 rename map、stats、action/state 维和 camera order。Task-P 则另需内部许可/revision，绝不搬到个人机器。

因此本文给出完整、可复制的兼容路径，但不会虚构“已在 V100 跑通”。公司 2.1 不兼容时正确产物是证据充分的 `ENV-BLOCKED`，不是偷偷改环境。

## 2. 本周必须回答的问题与最终产物

闭卷应能回答：VLM、视觉编码器、语言上下文、state projection 与 action expert 如何连接；processor 为什么属于模型的一部分；SmolVLA 的 flow path/target/sampling 方向；action chunk padding/mask 如何影响 loss；为什么 8 卡不能修复单进程 kernel/runtime 不兼容；冻结、DDP、AMP、checkpoint 各改变哪些状态。

最终产物：代码/model/data 精确 revision+license 表；安装 dry-run；离线资产 hash；schema/rename/normalization 审计；CPU 合同测试；严格 load + 单 batch forward；batch 1/2/4 memory scan；20-step smoke；200–500-step 短微调；resume 能力清单；独立 base/FT offline eval；有条件的四卡 DDP/profile；Model Card 与 `PASS/INCONCLUSIVE/ENV-BLOCKED/DATA-BLOCKED/LICENSE-BLOCKED` 状态。

## 3. 官方来源、版本、许可

| 对象 | 固定身份 | 许可/已核事实 | 边界 |
|---|---|---|---|
| LeRobot code | v0.4.3 / commit `0b067df57d21d3a02d6c511f1609172fa39ac29b` / tree `b5891d6333c4eb611bd3e81a4d13ed3c71210a6e` | Apache-2.0 | Python>=3.10，torch>=2.2.1,<2.8；另有本周 action-mask 审计补丁，patch SHA 单列 |
| SmolVLA base | repo `lerobot/smolvla_base`，revision `04d96c1f9167360280aaa54de31418f824d1ef48` | 约 0.5B；核验时模型卡 license 字段未声明 | 必须获得使用批准；不假定 Apache-2.0 |
| 嵌套 VLM + processor/tokenizer | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467` | Apache-2.0 | 是 base config 的必需间接资产；必须本地化并把 runtime config 指向该快照，不能让 Transformers 联网解析浮动 `main` |
| 公开样例数据 | `lerobot/svla_so101_pickplace`，revision `f641879e22172be7e8161d5e6c1503c2d2feb657` | Apache-2.0；50 episodes、11,939 frames、30 FPS | action/state 各 6 维，两路 AV1 视频 |
| Task-P | 内部 owner 提供精确 revision/split/schema/license | 当前未知 | 缺任何一项即 `DATA-BLOCKED`；只留公司 |
| SmolVLA paper | arXiv `2506.01844` | 论文版权 | 背景/方法，不等于权重许可 |

官方一手链接：[LeRobot v0.4.3](https://github.com/huggingface/lerobot/tree/v0.4.3)、[精确 SmolVLA 源码](https://github.com/huggingface/lerobot/tree/0b067df57d21d3a02d6c511f1609172fa39ac29b/src/lerobot/policies/smolvla)、[v0.4.3 SmolVLA 文档](https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla)、[固定 SmolVLA base tree](https://huggingface.co/lerobot/smolvla_base/tree/04d96c1f9167360280aaa54de31418f824d1ef48)、[固定 SmolVLM2 tree](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467)、[固定公开数据 tree](https://huggingface.co/datasets/lerobot/svla_so101_pickplace/tree/f641879e22172be7e8161d5e6c1503c2d2feb657)、[论文](https://arxiv.org/abs/2506.01844)。2026-09-03 核验。

base 顶层运行必需文件是 `config.json/model.safetensors/train_config.json`；嵌套 VLM 的 PyTorch 路径固定 `config.json/generation_config.json/model.safetensors/preprocessor_config.json/processor_config.json/tokenizer.json/tokenizer_config.json/special_tokens_map.json/added_tokens.json/chat_template.json/merges.txt/vocab.json`，不下载 ONNX 子树。公开数据固定 parquet、episodes/info/stats/tasks metadata 和两路 MP4。Hub revision 是 commit 身份；下载后还要对每个本地树的全部文件生成 canonical content-tree SHA-256，离线导入既核 revision lock，也核 tree digest，二者不可互换。

三条资源路径：公司 8×V100 只有在组织提供兼容且批准的隔离环境时运行真实模型；公司 PyTorch2.1 主环境停在审计。个人 5070Ti 可在现有兼容环境用公开数据重跑，但须先过权重许可门，且不接收公司材料。可选 H100 单独记录，不替代 V100 兼容/性能结论。

## 4. 架构数据流与 processor 边界

```text
raw dataset frame/chunk
  ├─ cameras: decode -> CHW float -> rename -> resize/pad -> image tokens
  ├─ state[6]: rename/order -> dataset stats normalize -> pad to max_state_dim=32
  ├─ task text: tokenize -> language ids/mask (max length 48)
  └─ action[50,6]: delta timestamps -> stats normalize -> pad to max_action_dim=32
                         ↓
VLM prefix: image(s) + language + state context
Action expert suffix: noisy action tokens + time embedding
                         ↓
velocity[50,32] -> slice first 6 dims -> unnormalize -> physical action chunk[50,6]
```

processor 冻结了 key rename、camera order、pixel range、resize、tokenizer、state/action stats、padding 与反归一化。模型权重相同但 processor 不同，就是不同系统。camera 交换后 loss 仍可能下降，因为网络能拟合错误映射；这不证明语义正确。

固定 base revision 的 `config.json` 已核到：三路期望视觉 key `observation.image/image2/image3`，每路声明 `[3,256,256]`；输入 state 6，输出 action 6；`chunk_size=n_action_steps=50`；内部 `max_state_dim=max_action_dim=32`；resize padding `[512,512]`；`num_steps=10`；`freeze_vision_encoder=true`、`train_expert_only=true`、`train_state_proj=true`。这些是该 revision 的事实，不应推广到未来模型。

公开数据 revision 则提供 `observation.images.up/side`、原视频 `[480,640,3]`、state/action `[6]`。建议固定映射：

```json
{
  "observation.images.up": "observation.image",
  "observation.images.side": "observation.image2"
}
```

该 base 的 `empty_cameras=0`；固定实现会只处理 batch 中存在的两路图像，不合成第三路。不能复制某一相机假装第三视角。train/eval 必须使用同一 map，并用回归样本确认 present/missing keys 与 image mask。

公开仓库只给 `train: 0:50`，不能拿同一 episode 的 frame 同时训练和评估。本周把 episode 当 group，预先固定 `train=0..39/dev=40..44/test=45..49`。只在 train episode 的 parquet 行上重算 state/action mean/std/min/max；发布的全 50-episode `meta/stats.json`只能用于来源审计，不能作为本实验 normalizer。preprocessor、postprocessor、rename map、train-only stats 和 tokenizer config 一起保存、hash、在新进程无 override reload；缺 stats 时上游 normalizer 可能静默 identity，因此本周在加载前显式拒绝缺 key/shape/非有限 stats。

## 5. SmolVLA 的 Flow Matching 公式

v0.4.3 固定源码采用与 Week09 记号方向相反但等价的 path：令 clean action 为 `a`，noise 为 `ε`，`t∈(0,1]`，

```text
x_t = t ε + (1-t) a
u_t = ε - a
L = mean(mask * (v_θ(x_t,t,context)-u_t)²)
```

这里 `t=0` 是 data，`t=1` 是 noise。推理从 noise 开始，`dt=-1/K`，从 `t=1` 积分到 `0`：

```text
x_(k+1) = x_k + dt * v_θ(x_k,t_k,context)
```

若模型完美 `v=ε-a`，`x_1=ε` 经总时长 `-1` 到 `a`。公式方向不变量：path 的导数、target 与积分方向必须同号；不能只改一个。

### 5.1 具体张量 shape

取 `B=2`、公开动作维 `A=6`、chunk `T=50`、内部最大动作维 `Amax=32`：

| 张量 | shape | dtype | 不变量 |
|---|---|---|---|
| raw image up/side | 各 `[2,3,480,640]`（decode 后） | float32 | key/order 固定 |
| resized image tokens 前输入 | 各 `[2,3,512,512]`（实现 padding） | AMP FP16 路径 | 像素预处理同 train/eval |
| raw state | `[2,6]` | float32 | 名称/单位与 stats 对齐 |
| padded state | `[2,32]` | FP16/FP32 | 后 26 维为实现定义 padding |
| raw action chunk | `[2,50,6]` | float32 | episode 边界有 pad mask |
| internal action/noise/target | `[2,50,32]` | noise/time FP32，模型混合精度 | 只前 6 维是真动作 |
| flow time | `[2]`，广播 `[2,1,1]` | float32 | v0.4.3 Beta(1.5,1) 映射到 `[.001,1]` |
| per-element loss | `[2,50,32]` | FP32 output projection 后 MSE | episode padding必须排除 |
| predicted physical actions | `[2,50,6]` | float32 | slice+postprocess 后比较 |

公开数据每帧本身只含 `[6]` action；`LeRobotDataset` 依据 policy 的 `action_delta_indices`/FPS 查询成 50-step chunk，并产出 `action_is_pad:[B,50]`。固定 v0.4.3 policy 却读取了拼错的 `actions_id_pad`，且把置零后的 padding 仍纳入 `.mean()` 分母；这会让 episode 尾部 batch 的监督被稀释。本周维护一个可审计补丁：读取正确 key，按有效时间步×动作维形成分子/分母，DDP 用 global numerator/denominator，并以全零 mask、padding 值不敏感和 rank 不等长测试防回归。补丁 SHA/dirty diff 必须进入 lineage，不能仍称“原版 v0.4.3”。

## 6. 最小公式自检

```bash
python - <<'PY'
import torch
action=torch.tensor([[[1.,2.]]])
noise=torch.tensor([[[5.,-2.]]])
t=torch.tensor([.25])[:,None,None]
x=t*noise+(1-t)*action
target=noise-action
perfect_next=x+(-.25)*target
print({'x_t':x.tolist(),'target':target.tolist(),'next_toward_data':perfect_next.tolist()})
assert torch.allclose(x,torch.tensor([[[2.,1.]]]))
assert torch.allclose(perfect_next,action)
PY
```

这只检验符号。它不加载权重、不证明机器人效果。

## 7. 冻结策略、状态显存与 AMP

固定 base config 已设置 vision encoder frozen、action expert为主训练对象、state projection trainable。实际对象构建后必须逐参数记录 `name/shape/numel/dtype/requires_grad`，对 canonical JSON 求 hash；optimizer 参数集合必须与 manifest 中 trainable 集合精确相等。训练前后逐 tensor 比较所有 frozen 参数 hash，不允许只凭 config 字段或总参数数推断冻结生效。

训练状态粗估对每个 trainable 参数常含权重/梯度/Adam 两矩，混合精度实现还可能保留 FP32 master。冻结会减少梯度与 optimizer state，却不一定消除前向 activation；DDP 仍在每卡复制所有参数。四卡能分摊 global batch/吞吐，不能解决单 rank OOM、unsupported kernel 或 Python/torch 版本冲突。

V100：使用 Accelerator/AMP FP16；不声明 BF16；不安装官方 FA2；不升级 CUDA13；SDPA/eager 后端以 v0.4.3 与本机实际选择为准。`torch_dtype`、vision/VLM/action expert、loss projection 的真实 dtype 都要打印。出现非有限先用单 batch FP32/FP16 对照定位，不能只降 batch 掩盖数值问题。

## 8. checkpoint/resume 的真实版本边界

v0.4.3 checkpoint 目录由官方代码生成：

```text
RUN/checkpoints/000100/
├── pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── train_config.json
│   └── processor/postprocessor files
└── training_state/
    ├── optimizer_param_groups.json
    ├── optimizer_state.safetensors
    ├── rng_state.safetensors
    ├── scheduler_state.json
    └── training_step.json
```

官方 CLI 允许从 `last` symlink 导航，但本周的可复现合同禁止用移动别名作为 parent：首段和恢复段的总目标都是 200，首段仅由 `LEROBOT_STOP_AFTER_COMPLETE_CHECKPOINT_STEP=100` 在完成 `RUN/checkpoints/000100` 后停止，恢复显式使用 `--resume=true --config_path=RUN/checkpoints/000100/pretrained_model/train_config.json --steps=200`，最终固定为 `RUN/checkpoints/000200`。这使“中断在 100”与“训练计划只有 100 step”成为两件可区分的事。

上游 v0.4.3 默认直接写最终目录，没有原子完成 marker。Lab 的可审计 vendor patch 将核心 checkpoint 先写入同文件系统 staging 目录，校验 model/config/optimizer/scheduler/RNG/step，fsync 后写 `CHECKPOINT_COMPLETE.json`，再以同目录 `os.replace` 提交。stop hook 只能在提交、`last` 更新和全 rank barrier 完成后 break。遗留的 `.*.staging-*` 是失败证据，不是可恢复 checkpoint；不得手工 rename 伪造完成。训练后添加的 processor 合同是可验 sidecar，不改核心训练状态。

每个可接受 checkpoint 旁必须有 lineage：base code commit/tree、action-mask/原子保存 patch SHA、三棵 Hub 资产 revision+content-tree digest、episode split hash、train-only stats hash、pre/postprocessor tree hash、完整 trainable manifest hash、训练命令/config、固定编号 parent checkpoint、attempted/successful step。加载 policy 必须 `strict=True`；processor 也从 checkpoint 新进程 reload，不能用当前 dataset stats override 掩盖缺文件。恢复后至少核 step/LR/optimizer/scheduler/RNG 与 processor lineage；两条路径的 fixed-heldout eval 也必须用同 sample/noise 对照。重要审计结论仍不变：v0.4.3 `save_checkpoint` 没有显式保存 `Accelerator` 的 FP16 `GradScaler`，也没有 sampler/dataloader cursor；即使 resume 能跑，也不能声称完全 bitwise 连续，严格验收应标 `RESUME-LIMITATION/INCONCLUSIVE`，不可隐藏。

## 9. 离线指标与 rollout 的边界

独立离线评估只读 held-out test episodes，固定 split hash、sample indices、已保存 processor/train-only stats、initial flow noise、`num_steps=10`、action chunk 50、base/FT dtype/backend。质量循环与 latency 循环分离；latency 至少 5 个 warmup，明确 dataset decode 是否排除、在每个样本计时边界同步，报 p50/p95、flow model calls/chunk、actions/chunk 和仅在完整消费 chunk 时成立的摊销 ms/action。指标可含 masked action MSE、endpoint、saturation、jerk；它们只衡量 held-out 动作拟合/平滑 proxy。

真实/simulator rollout 还受视觉分布、控制频率、延迟、reset、动力学与安全限制影响。没有获批环境就标 `INCONCLUSIVE`；不得用离线 loss 宣称机器人成功率。rollout 至少相同 episode seed、30–50 episode、系统失败单列，并在公司边界内执行。

## 10. 算法与 infra 联动

| 改动 | 算法含义 | infra 后果 | 必须固定/观测 |
|---|---|---|---|
| camera rename/order | 改变视觉语义 | decode/processor key 路径变化 | schema manifest、单 batch key/shape |
| chunk 50 / sampler 10 | 控制 horizon/积分精度 | action token与推理调用成本 | train/eval 相同 config |
| freeze vision/expert-only | 限制可学习模块 | optimizer/通信状态下降 | 实际 trainable manifest |
| batch 1→4 | 统计/全局 batch 变化 | activation/吞吐/显存变化 | 单变量 scan、峰值 reserved |
| 1→4 卡 DDP | 若 global batch 不固定会改优化 | all-reduce/数据 shard | global batch、sample IDs、Week07 invariant |
| processor/stats revision | 改变输入/输出尺度 | cache/文件依赖变化 | hash、base/FT 共用 |

## 11. 失败模式与诊断树

```text
安装/导入失败
├─ torch=2.1 而 v0.4.3 要>=2.2.1 -> ENV-BLOCKED，不升级公司栈
├─ pip 想替换 torch/CUDA -> 停，审计批准环境
├─ flash-attn/SM80+ kernel -> 禁用该 extra/backend；不编译未知 SM70 fork
└─ Python<3.10/系统库/AV1 缺失 -> 交管理员或用批准环境

严格 load/forward 失败
├─ model revision 与 code tag 不匹配
├─ 权重文件/hash/processor 不全
├─ camera keys 未 rename 或全缺
├─ state/action 维与 config 不符
└─ delta timestamps 未生成 50-step action chunk

loss 能降但 eval 差
├─ train/eval stats、camera order、num_steps 不同
├─ 只挑 best train loss
├─ 过拟合 50 episodes
├─ padding loss 分母错误
└─ offline proxy 与闭环控制错位

resume 异常
├─ `config_path` 指向错层级
├─ checkpoint 文件不完整/last symlink 坏
├─ optimizer/scheduler/RNG/step 未恢复
└─ GradScaler 未由 v0.4.3 显式保存 -> 已知限制，不谎称严格连续
```

## 12. 自检与 teach-back

1. 画出 image/language/state prefix 与 noisy-action/time suffix，标注 `[B,50,32]`。
2. 写出 v0.4.3 的 `x_t`、velocity target 和从 `t=1→0` 的 Euler 方向。
3. 为什么公开 dataset 的 `[6]` action 要由 delta timestamps 变成 `[B,50,6]`？
4. base 期望三相机、数据只有两相机时，什么做法可审计，什么做法是伪造？
5. processor 的哪些内容必须随 checkpoint 保存？
6. 为什么四卡 DDP 不能修复 torch 版本/FA2/SM70 不兼容？
7. model card 未声明 license 时，LeRobot Apache-2.0 能否自动授权权重？
8. v0.4.3 resume 明确保存什么？哪项 FP16 状态没有显式证据？
9. offline action MSE 与 rollout success 为何可能相反？
10. teach-back：用两分钟解释为何“成功加载 0.5B 权重”离“可复现的 VLA 微调结果”仍隔着 runtime、license、processor、训练状态和评估五层证据。
