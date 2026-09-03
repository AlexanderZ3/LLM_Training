# 共享实验、日志、证据与安全协议

> 适用于 Week 01–14。  
> 技术链接核验日期：2026-09-03。  
> 本文命令未经用户机器实测；先识别“可执行 / 模板 / 伪代码”标签。

## 1. 三条环境线，不混写结论

### 公司 8×V100 线

`用户自述待核验`：8×V100、既有 Conda、PyTorch 2.1.0 与相应 CUDA，可按公司流程安装包，不新建环境。

- 目标：FP16 AMP、NCCL、DDP/FSDP、profiling、checkpoint/recovery。
- 不能假设：每卡一定 32GB、特定 NVLink/NUMA 图、CUDA 版本、包版本、空闲状态或管理员权限；Week 03 现场探针后确认。
- V100 是 Volta。它支持 FP16 Tensor Core mixed precision；通常不把 BF16、TF32、FP8 或官方 FlashAttention-2 当可用路径。
- 公司所有代码、数据、配置、日志、曲线、trace、checkpoint、图片、拓扑和性能数值留在公司内部。

### 个人 RTX 5070 Ti 16GB 线

`用户自述待核验`：用户有完整权限。

- 目标：公开数据缩小复现、Blackwell 兼容性、公开作品集证据。
- 不能把个人单卡结果描述成公司多卡结果。
- 16GB 需要小模型、短序列、gradient accumulation、activation checkpointing、冻结/PEFT 等手段；先测再决定。
- Python/PyTorch/CUDA 组合须按 NVIDIA 驱动和 PyTorch 官方安装矩阵选择，不在教程中硬编码一个“万能”版本。

### 可选 H100 线

只有同时满足以下条件才租用：

1. 小规模正确性与恢复门已通过；
2. 问题确实需要 H100 的显存、互联或精度能力；
3. 预计 GPU-hours、上限费用和停止条件已写明；
4. 数据许可、镜像、checkpoint、评测和销毁策略已写明；
5. 不用昂贵运行代替实验设计。

## 2. 命令标签

- **可执行**：无占位符，语法完整；仍需确认当前目录与权限。
- **模板**：含 `<...>`、`/path/to/...`、`$VARIABLE` 或项目特定 ID，替换后才可执行。
- **伪代码**：用于解释算法/接口，不能直接运行。

教程若没有标签，按“模板”处理。

## 3. 环境预检：先读取，后安装

公司 Linux 环境的只读探针（可执行；输出必须留在公司）：

```bash
which python
python --version
python -m pip --version
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
nvidia-smi topo -m
```

Python 级探针（可执行）：

```bash
python - <<'PY'
import json, platform, torch
items = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "cudnn": torch.backends.cudnn.version(),
}
if torch.cuda.is_available():
    items["devices"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": torch.cuda.get_device_capability(i),
            "total_memory": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ]
    items["bf16_reported"] = bool(torch.cuda.is_bf16_supported())
print(json.dumps(items, indent=2))
PY
```

个人 Windows PowerShell 可使用（可执行）：

```powershell
python --version
python -m pip --version
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

在确认现状前禁止直接 `pip install -U torch transformers ...`。每周只安装缺失且已审计的最少包：

```bash
# 示例：把这两个变量改成当周教程规定的包与约束；先检查，不执行升级
PACKAGE="datasets"
VERSION_CONSTRAINT="==2.21.0"
python -m pip show "$PACKAGE"
python -m pip index versions "$PACKAGE"
python -m pip install --dry-run "${PACKAGE}${VERSION_CONSTRAINT}"
```

`pip index`/`--dry-run` 依赖 pip 版本和网络策略；不可用时查公司镜像/管理员提供的元数据，不绕过访问控制。决定安装后，将最终 `pip freeze` 或 `conda list` 保存在该环境内部的 run manifest 旁。

## 4. 固定实验目录

下面是模板，不要求复制公司内容回本仓库：

```text
training-lab/
├── README_OFFLINE.md
├── configs/
│   ├── base.yaml
│   └── experiments/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── eval/
│   └── profiling/
├── scripts/
│   ├── probe_env.py
│   ├── dry_run.py
│   ├── train.py
│   ├── eval.py
│   └── profile.py
├── tests/
├── reports/templates/
└── runs/<run_id>/
    ├── config.resolved.yaml
    ├── run_manifest.json
    ├── metrics.jsonl
    ├── checkpoints/
    └── decision.md
```

训练入口统一支持：

```text
--config --seed --max_steps --precision fp32|fp16
--resume_from --output_dir --dry_run --profile_steps
```

分布式入口再支持 `--world_size` 或由 `torchrun` 环境读取 rank/world size。禁止在代码中硬编码 GPU 数、数据路径或敏感名称。

## 5. 每个 run 先写实验合同

```yaml
run_id: <date-topic-variant-seed>
hypothesis: <一个可证伪句子>
independent_variable: <只改一个主变量>
controlled_variables:
  - model architecture and parameter count
  - data split/hash and preprocessing
  - optimizer/lr schedule and token or sample budget
  - global batch or explicitly declared weak scaling
primary_metric: <与任务目标直接相关>
secondary_metrics: [loss, grad_norm, step_time, peak_memory]
correctness_invariants: [shape, no_leakage, finite_params, resume_state]
stop_conditions: [nonfinite_param, repeated_overflow, wrong_evaluator]
decision_rule: <PASS/FAIL-MODEL/FAIL-SYSTEM/INCONCLUSIVE 的判定>
```

先写合同才能区分“结果不好”和“实验不可解释”。

## 6. 训练循环的共同不变量

### batch 与 token

```text
global_batch = micro_batch_per_rank × world_size × gradient_accumulation_steps
tokens_per_update = global_batch × effective_sequence_tokens
```

含 padding 时必须同时报告 nominal tokens 与 non-padding tokens。跨卡等价实验固定 global batch；弱扩展实验固定每卡 micro-batch，并明确 global batch 变化。

### FP16 AMP

PyTorch 2.1 兼容模板（可执行片段，需放入完整训练器）：

```python
scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=torch.float16,
                    enabled=(precision == "fp16")):
    outputs = model(**batch)
    loss = compute_loss(outputs, batch)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)          # 必须先 unscale
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scale_before = scaler.get_scale()
scaler.step(optimizer)              # 非有限梯度时可能跳过 optimizer.step
scaler.update()
scale_after = scaler.get_scale()
optimizer.zero_grad(set_to_none=True)
```

需要记录 `scale_before/after` 和是否跳步。敏感 reduction、normalization、自定义 loss 或 evaluator 可显式在 FP32 中计算，但必须用 FP32 reference 验证，而不是凭感觉到处 `.float()`。官方依据：[PyTorch 2.1 AMP](https://docs.pytorch.org/docs/2.1/amp.html)、[NVIDIA mixed precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)。

### gradient accumulation

- loss 若是 batch mean，通常每个 micro-step 除以 accumulation steps 后 backward；
- scaler 在一个有效 batch 的所有 micro-step 共用，最后一次才 unscale/clip/step/update；
- DDP 非最后 micro-step 可用 `no_sync()`，但先验证数值等价再优化通信；
- scheduler 按 optimizer update 还是 micro-step 前进必须写清。

### checkpoint 必须包含

```text
model / optimizer / scheduler / GradScaler / EMA（如用）
global_step / epoch / sampler position
Python RNG / NumPy RNG / torch CPU RNG / every CUDA RNG
resolved config / code id / data id / world size / precision
```

“权重能加载”不等于 resume 正确。验证方法：从同一 checkpoint 出发，在同一 fixed batch 上比较下一个 update 的 batch 标识、loss、LR、scale、grad norm 和参数差异。完全 bitwise 一致并非所有 kernel/并行路径都保证，因此必须预注册容差和已知 nondeterminism。

## 7. 日志的最小字段

每次 optimizer update 建议记录：

```json
{
  "step": 0,
  "split": "train",
  "loss": 0.0,
  "loss_components": {},
  "lr": 0.0,
  "grad_norm_pre_clip": 0.0,
  "grad_norm_post_clip": 0.0,
  "amp_scale": 0.0,
  "optimizer_step_skipped": false,
  "step_time_ms": 0.0,
  "data_time_ms": 0.0,
  "samples_per_second": 0.0,
  "nonpad_tokens_per_second": 0.0,
  "peak_allocated_bytes": 0,
  "peak_reserved_bytes": 0
}
```

不要每步同步所有 GPU 指标造成测量污染。细粒度统计只在短 profile window 打开；常规训练按间隔记录。

公司环境优先 JSONL/CSV + 本机 TensorBoard，禁用任何默认联网同步工具；W&B、MLflow 远端或外部 API 只有公司明确批准才可用。个人环境可选这些工具，但要记录版本和配置。

TensorBoard 模板：

```bash
# 模板；仅在已安装 tensorboard 且端口使用符合环境政策时
tensorboard --logdir <RUN_ROOT> --port 6006 --bind_all false
```

## 8. 怎么读 loss 曲线

永远把 train loss 与 `LR、grad norm、AMP scale、skipped step、data time、step time、eval` 对齐看。

| 形态 | 第一假设 | 必须区分的替代解释 | 最小实验 |
| --- | --- | --- | --- |
| 从第一步几乎不降 | label/shift/mask 错、LR 太低、参数未更新 | 数据随机、本就到基线、日志位置错误 | 固定 1 batch 反复训练并检查参数 delta |
| 立刻降到异常接近 0 | 泄漏、target 进入输入、padding 主导 | 极小/重复数据确实可记忆 | 打乱/遮蔽输入，逐 token 查看 mask |
| 周期性尖峰 | LR、数据 shard、eval/checkpoint 边界 | AMP overflow、异常 batch | 对齐 batch id、scale、grad norm、时间 |
| train 降、val 升 | 过拟合或 split shift | train/eval preprocessing 不同 | 固定 pipeline，对比按桶指标与去重 |
| 平滑但任务指标不升 | surrogate loss 与目标错配 | evaluator 错、生成/采样参数错 | gold evaluator、反事实输入、task metric |
| FP16 偏离 FP32 | overflow/underflow、敏感 op | seed/kernel 非确定性 | 同 fixed batch、短轨迹、模块级 finite check |
| 多卡与单卡偏离 | global batch/LR、sampler、reduction | dropout/BN/RNG、日志聚合 | 固定 batch/seed，比较一步梯度和参数 |

语言模型初始交叉熵接近 `ln(V)` 只在预测近似均匀、词表大小为 `V`、mask/reduction 正确时成立；它是 sanity heuristic，不是定律。生成质量还受 tokenizer、采样、上下文和数据分布影响。

## 9. 性能测量纪律

稳态 step time 不应包含首轮 CUDA context、编译、数据缓存、checkpoint 或 eval：

```python
# 伪代码：需要在真实训练步骤两侧放置
for _ in range(20):
    train_step()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    train_step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

同时保留端到端吞吐，因为“稳态更快”可能被加载、评测和保存吞掉。每个 A/B 点至少重复 3 次，报告 median、范围/分位数和共享负载条件，不只给最好值。

显存至少区分：

- parameters、gradients、optimizer states；
- activations 与 activation checkpointing；
- temporary buffers / all-gather buckets；
- allocator `allocated` 与 `reserved`；
- 峰值位置与 batch/sequence/image/action horizon。

工具层级：

1. 训练器自己记录 data/forward/backward/optimizer 时间；
2. PyTorch Profiler 看 op、shape、memory 和调用栈；
3. Nsight Systems 看 CPU/CUDA/NCCL 时间线；
4. Nsight Compute 只针对已确认的少数热点 kernel。

不要从 GPU utilization 单指标推断 compute-bound。

## 10. 分布式正确性

正式 scaling 前依次通过：

```text
单卡 tiny overfit
→ 2 卡 fixed-global-batch 一步/短轨迹等价
→ topology pair 对照
→ 4 卡 smoke
→ 另一四卡组对称性（若现场拓扑支持）
→ 8 卡 smoke
→ 1/2/4/8 profile
```

检查：每个 epoch 的唯一 sample ID 数、rank 间重复/缺失、loss reduction、最后不满 batch、`DistributedSampler.set_epoch`、所有 rank 的 optimizer update 次数，以及更新后的参数一致性。

弱扩展效率：

```text
E_N = throughput_N / (N × throughput_1)
```

必须注明 throughput 的单位和 batch 规则。来源计划中的 E2/E4/E8 阈值是该 workload 的课程目标，不是 NVIDIA 硬件规格。

## 11. 失败分类与止损

### 立即停止

- 参数首次出现 NaN/Inf；
- evaluator 的 gold cases 未达到预期正确性；
- 数据泄漏、时间反向、label mask 或单位错误；
- checkpoint 无法恢复完整状态；
- 多 rank 卡死且可能占用共享资源；
- 输出目录或数据权限不符合政策；
- 预算/最大步数未限制的无人值守任务。

### 分类

- `FAIL-SYSTEM`：数据、环境、数值、通信、测量或恢复不可信。
- `FAIL-MODEL`：系统和 evaluator 已通过，但预注册算法假设未成立。
- `INCONCLUSIVE`：方差、样本或预算不足。
- `PASS`：只表示该预注册问题在该范围内通过。

## 12. 证据记录

仓库证据遵循 `schemas/evidence.schema.json`；run 遵循 `schemas/run_manifest.schema.json`。公司内部证据只在公司环境保存，仓库中最多记录获准的抽象结论，不能放真实路径或内部数字。

最小学习证据：

```text
date / week / task / skill
difficulty / A0-A4 / environment scope
result status / scoring basis / error type
K-A-D-E（基线后才可正式计分）
privacy review / regression date
```

## 13. 口试规则

1. `03_ORAL_EXAM.md` 闭卷、限时；先写假设和推理。
2. 不会时只请求一个最小提示，辅助等级从 A0/A1 升到 A2。
3. 完成后才看 `04_REFERENCE_ANSWERS.md`。
4. 参考答案是评分锚点，不是唯一措辞；能给出更严谨的边界、反例和实验同样得分。
5. 24–72 小时后换数据/shape/硬件条件做回归题，避免背答案。

## 14. 官方基线来源

- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [NVIDIA Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html)
- [CUDA 13 Release Notes：Volta 构建支持边界](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
- [PyTorch 2.1 AMP](https://docs.pytorch.org/docs/2.1/amp.html)
- [PyTorch 2.1 DDP](https://docs.pytorch.org/docs/2.1/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch 2.1 FSDP](https://docs.pytorch.org/docs/2.1/fsdp.html)
- [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

使用当天再次核验；本列表不是兼容性保证。
