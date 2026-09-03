# Week 03 基础篇：V100、FP16、拓扑与 NCCL 的可信基线

> 本周所有硬件数量、显存、NVLink关系和CPU affinity均为用户自述假设，必须用本机命令核验；本文不把它们写成已测事实。公司环境固定 PyTorch 2.1，不自行升级。V100（Volta，通常 compute capability 7.0）支持 FP16 Tensor Core，通常不支持原生 BF16；禁用 BF16、TF32、FP8与官方 FlashAttention-2。公司日志、拓扑图、trace、性能数字和代码不得导出。

## 1. 本周问题与最终产物

本周不追“最大 tokens/s”，而是建立后11周可引用的 Hardware Card：

1. 每张可见GPU的FP32/FP16计算正确性；
2. driver、torch build CUDA、runtime、cuDNN、NCCL、SM架构事实；
3. NV2/NV1/SYS pair、两个四卡组和8卡 collective correctness/带宽；
4. 30M/120M synthetic Transformer 的FP32/FP16同工作量基线；
5. 显存包络、计时方法、波动与共享负载；
6. 一次可恢复的FP16 overflow故障注入；
7. 对不可见sysfs/NUMA的“已知未知项”。

```text
只读环境探针 → 每卡独立算术 → 拓扑/P2P事实
→ PyTorch collective correctness → nccl-tests上限
→ 相同Transformer workload FP32/FP16 → Profiler
→ Hardware Card + 后续冻结默认值
```

Week01模型在这里只是受控载荷；Week07/08 的DDP/FSDP会直接引用本周rank mapping与collective上限。

## 2. 平台职责与环境边界

| 平台 | 本周职责 | 不能推出的结论 |
|---|---|---|
| 公司 Linux / 用户自述8×V100 | 完整拓扑、2/4/8卡NCCL和FP16基线 | 未探测前不能声称8卡都可见、32GB或某pair更快 |
| 个人5070 Ti | 单卡correctness、FP16计时脚本的公开复现 | 单卡不能验证公司拓扑/NCCL；不同架构数字不可横比 |
| 可选H100 | 另立环境卡 | BF16/TF32/FA结果不能替代V100结论 |

本周不下载数据或模型权重，正式载荷是固定seed的synthetic token IDs。参考源码只有 `NVIDIA/nccl-tests`；如果公司不能编译，先用自写PyTorch probe做correctness，带宽上限记为缺口。

## 3. 官方来源与版本策略

技术链接核验日期：**2026-09-03**。

| 对象 | 固定方式 | 用途 | 许可/限制 |
|---|---|---|---|
| nccl-tests | commit `b4d5beebca8a76cf01335f724d154b9b9d394d96` | all_reduce/all_gather/reduce_scatter标准测试 | BSD-3-Clause；公司编译工具链需批准 |
| PyTorch | 公司既有 `2.1.x` 实测完整版本/build hash | AMP、distributed、Profiler | 不依据stable/current文档升级；优先2.1文档 |
| CUDA/driver/NCCL | 只读探针解析本机版本 | 兼容性manifest | 不安装CUDA13或替换驱动 |
| synthetic token载荷 | 本教程代码+seed+config hash | 排除数据I/O | 无上游数据许可；不代表真实训练吞吐 |

若 `torch.cuda.get_arch_list()` 不含 `sm_70`，或自定义扩展只含更高架构，V100可能在启动kernel时失败。CUDA toolkit、driver与torch wheel内置runtime是三个不同对象：`nvidia-smi`显示的“CUDA Version”是driver可支持上限，不等于 `torch.version.cuda`，也不证明系统`nvcc`版本。

## 4. 已知拓扑只是待核验假设

用户提供的候选拓扑：GPU0–3和4–7各自形成NVLink四卡组；对照pair `0,1=NV2`、`0,3=NV1`、`0,5=SYS`；跨组存在若干NV1/NV2边。只有 `nvidia-smi topo -m`、P2P探针和实测能确认。

- `NV1/NV2`：两GPU之间有一/两条NVLink聚合路径标记；具体带宽随代际与工具定义，不能只凭名字填数字。
- `PIX`：路径最多穿过一个PCIe bridge；对NIC邻近可能重要。
- `PHB`：穿过PCIe host bridge。
- `NODE/SYS`：跨更多host/NUMA/CPU互连，通常延迟更高；仍不代表NCCL全局ring只能走SYS。
- `CUDA_VISIBLE_DEVICES=4,5` 后，进程内逻辑device 0/1对应物理4/5；rank与物理ID必须同时记录。

一条全NVLink环“存在”只证明图论路径存在。NCCL会综合ring/tree/channel/P2P选择；不要手工强制算法来制造预期排序。

## 5. Collective的数学、shape与指标

all-reduce 输入每rank一个同shape tensor，例如 world size `N=4`，每rank `[1_048_576] float32`（4 MiB）。SUM后每rank得到所有rank逐元素和，shape/dtype不变。

若每rank初始常数为 `rank+1`，期望值：

$$\sum_{r=0}^{N-1}(r+1)=\frac{N(N+1)}2.$$

这是一条比“进程没挂”更强的correctness断言。all-gather把每rank `[M]` 汇成 `[N,M]`；reduce-scatter把按rank组织的总输入规约并切成每rank `[M]`。

`algbw = payload_bytes / time` 描述API视角有效带宽。对ring all-reduce，nccl-tests常用：

$$busbw=algbw\times\frac{2(N-1)}N.$$

all-gather/reduce-scatter因子通常是 `(N-1)/N`。应读取固定nccl-tests版本的实现/输出，不把某公式套给所有collective。

worked example：4 rank、每rank256MiB、平均0.02s，`algbw=12.5GiB/s`；ring factor1.5，对应`busbw≈18.75GiB/s`。这是算术示例，不是V100实测。

需要同时报告：消息size、dtype、world size、映射、warmup、重复次数、algbw/busbw p50与波动、错误数、共享负载。单一最好值不是基线。

## 6. FP16 Tensor Core与GradScaler

FP16约5位指数、10位显式尾数，动态范围远小于FP32/BF16。V100能加速对齐良好的FP16 GEMM，但是否更快取决于矩阵规模、维度对齐、kernel、launch与数据/CPU瓶颈。

AMP训练顺序：

```text
FP32主参数
→ autocast(float16) forward
→ loss/reduction必要时FP32
→ scaler.scale(loss).backward()
→ scaler.unscale_(optimizer)
→ finite/grad-norm与clip
→ scaler.step → scaler.update
```

GradScaler通过放大loss减少小梯度在FP16表示中下溢；它不能修复模型本身的 `exp` overflow、错误normalization或无限logits。scale下降表明检测到inf/NaN gradient并跳过update；只有成功update才推进scheduler和全局训练step。

故障注入应把loss临时乘大系数，仅在独立run中执行。验证：scale回退、参数未变为非有限、移除注入后训练继续。不能让故障配置进入后续周。

## 7. 30M/120M载荷的shape与成本

本周自包含 `nn.TransformerEncoder` causal LM：

| 配置 | V | T | L | D | heads | MLP | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---|
| small | 8192 | 256 | 8 | 512 | 8 | 2048 | 由脚本实算，目标约30M |
| medium | 8192 | 256 | 16 | 768 | 12 | 3072 | 由脚本实算，目标约120M |

`input [B,256] int64 → embedding [B,256,D] → each layer → logits [B,256,8192]`。维度512/768和head dim64对Tensor Core友好。对比FP32/FP16时，model、batch、seed、warmup、timed steps、optimizer与device必须相同。

计算近似：每层参数主项约 `12D²`；attention score激活 `[B,H,T,T]`。从T=256加倍到512，score元素约4倍。模型太小或batch太小时，Python/launch开销会掩盖FP16收益。

## 8. 正确计时、显存与Profiler

CUDA异步执行，`time.perf_counter()`不加`synchronize`会只量到launch时间。可用：

- CUDA events记录GPU区间；
- 或区间前后 `torch.cuda.synchronize()`；
- 前20步warmup，后100步计时；
- 单独报告steady-state，不把初始化、首次compile或checkpoint混入。

`allocated`是tensor实际占用；`reserved`是caching allocator向CUDA保留。OOM要记录发生在forward/backward/optimizer/eval哪个阶段和两者峰值。Profiler只采短窗口并看self CUDA time、调用次数、shape、memory；公司trace只留本机。

## 9. NCCL与训练系统联动

DDP每次backward会对gradient buckets做all-reduce；小bucket多时latency敏感，大bucket/大模型更接近bandwidth上限。FSDP除reduce-scatter外还会all-gather参数，所以本周要额外测这两个collective。

一个SYS pair慢不能直接推出8卡一定慢：全局ring/tree可能绕过或以不同channel利用NVLink边。相反，2卡NV2快也不保证8卡scaling好。必须分层：

```text
每卡算术 → 2卡pair → 4卡组A/B → 8卡collective
→ 真实DDP compute/communication overlap
```

`NCCL_DEBUG=INFO` 只开一次最小case；日志可能含内部host/interface/topology信息，绝不外传。hang先保存最小映射/命令/最后阶段，不通过永久设置`NCCL_P2P_DISABLE=1`掩盖。

## 10. 正确性不变量

1. 8张卡（若实际可见）逐卡FP32/FP16算术和backward通过；
2. 物理GPU ID、逻辑device、rank映射明确；
3. 每个collective不仅退出0，还验证数值期望；
4. nccl-tests错误数为0；
5. NCCL相同mapping完整重复3次并报告波动；Transformer的FP32/FP16两侧各固定100个timed successful updates，若要发布稳定性能结论再把整对实验重复3次；
6. CUDA计时包含同步/warmup；
7. FP32/FP16配置除precision外一致；
8. V100不使用BF16/TF32/FP8/FA2；
9. hwloc/sysfs不可见标 `FAIL-VISIBILITY`，不伪造NUMA；
10. Hardware Card区分用户自述、探针事实、实测结果与推断。

## 11. 失败树

| 症状 | 第一检查 | 候选根因 | 修复/回归 |
|---|---|---|---|
| 某卡matmul错误 | 同seed CPU FP64、device ID | 硬件/ECC、错误容差、kernel | 缩小最小复现并报平台；禁止多卡 |
| `no kernel image` | capability与`get_arch_list` | wheel/extension无sm70 | 使用管理员批准兼容build |
| BF16路径出现 | dtype/assert与config | 代码默认或autocast抄错 | 显式float16；dtype单测 |
| NCCL hang | 两rankprobe+timeout | rank/world/env、P2P、共享故障 | 2卡最小映射；平台升级，不永久禁P2P |
| nccl-tests build失败 | nvcc/make/CUDA_HOME | 工具链缺失/ABI不符 | 管理员binary；PyTorch correctness fallback |
| SYS反而更快 | 3次波动/共享负载 | 消息size、算法、噪声 | 多size重复，标INCONCLUSIVE |
| FP16不快 | GPU利用率与矩阵shape | 载荷太小/不对齐/CPU瓶颈 | 放大受控载荷，Profiler定位 |
| scale连续回退 | FP32同batch、grad首次异常 | overflow/clip顺序/loss放大 | 修复后200-step gate |
| NUMA信息矛盾 | 容器cpuset/sysfs权限 | 可见性限制 | 记录未知，不越权mount |

## 12. Teach-back（先闭卷）

1. 解释driver版本、`torch.version.cuda`和系统nvcc为何可不同。
2. 手算4-rank all-reduce常数输入的期望结果与busbw factor。
3. 解释物理GPU5如何变成进程内device1/rank1。
4. 画出“每卡→pair→4卡→8卡→DDP”的证据链。
5. 写完整FP16 step并解释GradScaler不能修什么。
6. 为什么必须同步CUDA计时？
7. 为什么2卡NV2结果不能预测8卡collective？
8. 什么证据足以将故障标为 `FAIL-SYSTEM`？

## 13. 官方一手资料

技术链接核验日期：**2026-09-03**。

- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [PyTorch 2.1 distributed](https://pytorch.org/docs/2.1/distributed.html)
- [PyTorch 2.1 AMP examples](https://pytorch.org/docs/2.1/notes/amp_examples.html)
- [PyTorch 2.1 Profiler](https://pytorch.org/docs/2.1/profiler.html)
- [CUDA toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [FlashAttention official support statement](https://github.com/Dao-AILab/flash-attention)
