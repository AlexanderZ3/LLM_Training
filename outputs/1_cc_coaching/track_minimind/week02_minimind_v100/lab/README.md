# Week M02 lab — 同一个 64M MiniMind 在 8×V100 上的四种放置方式

DDP / FSDP / 专家并行 / GRPO 角色拆分，四种放置方式共用一套数据切分、一套日志字段、
一套判定阈值，所以它们的 loss 与耗时可以直接对比。

## 0. 两个 torch 版本的关系（先读这一段）

| | 目标机 | 本机 |
| --- | --- | --- |
| 硬件 | 公司 8×V100 32 GB（sm70） | Windows，无 NVIDIA GPU |
| torch | **2.1.0**（不升级、不新建 conda） | 2.14.0+cpu（conda 环境 `ResearchAgentPy310`） |
| 精度 | fp16 + GradScaler；**没有 bf16** | fp32（CPU 上 autocast 是 no-op） |
| 后端 | NCCL | gloo |

本 lab 的代码只调用 **torch 2.1.0 已经存在的 API**：`FullyShardedDataParallel`、
`ShardingStrategy`、`MixedPrecision`、`StateDictType`、`FullStateDictConfig`、
`FSDP.optim_state_dict` / `optim_state_dict_to_load`、
`torch.distributed.checkpoint.save_state_dict` / `load_state_dict` +
`FileSystemWriter` / `FileSystemReader`、`transformer_auto_wrap_policy`、
`apply_activation_checkpointing`、`dist.all_to_all_single`、`dist.new_group`、
`torch.cuda.amp.autocast` / `GradScaler`、`torch.backends.cuda.sdp_kernel`。

**刻意不用**（这些是 2.2+ 才有的东西，写进去就没法在公司跑）：`fully_shard`、
`DTensor`、`torch.distributed.tensor`、`device_mesh` / `init_device_mesh`、
`torch.accelerator`、`torch.get_default_device`、`torch.nn.attention.sdpa_kernel`。

本机的 torch 2.14 能跑上面所有 2.1 的 API（`save_state_dict` 在新版里被标记
deprecated，`ckpt_reshard._dcp_save` 因此带了一个同语义的 fallback），
所以本机验证结果对目标机有意义——但只限于 CPU 能覆盖的那部分，见第 6 节。

## 1. 安装与前置

### 1.1 MiniMind（两台机器都要）

本 lab **不复制** MiniMind 源码，只引用它：

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind && git checkout 7a6fddd63a30c06b2fdd5fac4089922b29bc841b
```

然后设置 `MINIMIND_ROOT`：

```bash
export MINIMIND_ROOT=/abs/path/to/minimind          # Linux / 公司机器
```
```powershell
$env:MINIMIND_ROOT = "D:\abs\path\to\minimind"      # Windows 本机
```

每个入口脚本也都接受 `--minimind-root <path>`，优先级高于环境变量。
没有设置时错误信息里直接给出上面这两条命令。

`mm_dist.common.import_minimind()` 会把 `MINIMIND_ROOT` 插到 `sys.path[0]`。
注意 MiniMind 的顶层包名是通用的 `model` / `dataset` / `trainer`，会遮蔽同名第三方包；
本 lab 自己不使用这三个名字。

### 1.2 公司环境：确认依赖，不安装

公司环境**不新建 conda、不升级 torch、不 pip install**。用两个只读脚本确认现状：

```bash
python lab/scripts/probe_company.py --out <公司内路径>/probe.json
python lab/scripts/audit_minimind_compat.py --out <公司内路径>/compat_audit.md
```

`probe_company.py` 的退出码：全部检查通过是 0，有任一项不符合预期是 1
（例如卡数不是 8、`is_bf16_supported()` 意外为 True、MiniMind commit 不匹配）。
非 0 不代表不能继续，代表任务卡上要记一条偏差。

怎么确认"没有新建环境"：

```bash
conda env list                 # 运行前后行数不变
python -c "import torch, sys; print(sys.prefix, torch.__version__)"
pip freeze > /tmp/before.txt   # 跑完 lab 后再 diff 一次，应当为空
```

### 1.3 本机（Windows / CPU）

```powershell
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" -m pip install `
    torch --index-url https://download.pytorch.org/whl/cpu
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" -m pip install `
    transformers pytest
```

裸 `python` 在本机是 Windows 商店占位入口，不能用；一律走上面的绝对路径，
或项目根的 `.claude/scripts/cc_py.ps1`。

## 2. smoke 命令（从空目录开始的最短路径）

```powershell
# 0) 设 MINIMIND_ROOT（见 1.1）
$env:MINIMIND_ROOT = "<minimind 绝对路径>"
$env:PYTHONPATH = "<lab 绝对路径>\src;$env:PYTHONPATH"

# 1) CPU 单测（约 2 分钟）
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" -m pytest lab/tests -q

# 2) 单进程 DDP 训练 3 步（CPU、fp32）
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" -m mm_dist.train_ddp `
    --config lab/configs/tiny_cpu.json --out-dir .\runs\smoke `
    --force-cpu --dtype float32 --global-batch 8 --accum 4 --seq-len 32 `
    --max-steps 3 --num-samples 256

# 3) EP 的数值等价（不需要多进程）
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" -m mm_dist.ep_moe equiv `
    --config lab/configs/moe_tiny_cpu.json --out-dir .\runs\smoke `
    --force-cpu --dtype float32 --batch 2 --seq-len 64 --tol 1e-5

# 4) 2 进程 gloo 等价 —— 本机走 pytest（它用 mp.spawn，不经过 torchrun）
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" -m pytest `
    lab/tests/test_ddp_gloo_cpu.py -q
```

第 4 步在公司 Linux 机器上是这条命令（本机跑不了，原因见"已知限制"第 7 条）：

```bash
torchrun --nproc_per_node 2 -m mm_dist.train_ddp \
    --config lab/configs/tiny_cpu.json --out-dir ./runs/smoke \
    --backend gloo --force-cpu --dtype float32 \
    --global-batch 8 --accum 2 --seq-len 32 --max-steps 3 \
    --num-samples 256 --equiv-check --log-name equiv_ws2
```

公司机器上的完整流程见同级目录的 `02_LAB_GUIDE.md` 第 3 节。

## 3. 目录说明

```text
lab/
├── README.md                 本文件
├── requirements.txt          公司 / 本机两段依赖
├── configs/
│   ├── tiny_cpu.json         CPU 单测用的 dense 小模型（hidden 64, 2 层）
│   ├── moe_tiny_cpu.json     CPU 单测用的 MoE 小模型（4 experts, top-1）
│   ├── v100_dense.json       公司主线 dense（63.9M，seq 512，G=64）
│   └── v100_moe.json         公司主线 MoE（4 experts；EP=8 时用 --num-experts 8 覆盖）
├── src/mm_dist/
│   ├── common.py             分布式初始化、dtype 守卫、seed、模型/数据、JSONL 日志
│   ├── train_ddp.py          DDP 训练 + fixed-global-batch 等价 + 日志比对
│   ├── train_fsdp.py         FSDP1 三策略 + 集合通信计数 + activation checkpointing
│   ├── ckpt_reshard.py       分片 checkpoint 保存 / 跨 world_size 恢复 / loss 连续 verify
│   ├── ep_moe.py             教学版专家并行（六步 all_to_all）+ 阶段计时 + 等价检查
│   ├── router_stats.py       router bias 注入、每专家负载统计、bias × aux_coef 扫描
│   ├── grpo_roles.py         GRPO 三种角色放置（replicate / policy_fsdp / split_roles）
│   └── faults.py             kill rank + NCCL 超时/错误处理环境助手
├── scripts/
│   ├── probe_company.py          环境探针 -> probe.json（拓扑抽象化，--redact 默认开）
│   ├── audit_minimind_compat.py  静态兼容审计 -> Markdown（只读 MiniMind）
│   ├── run_ddp_equiv.{sh,ps1}    Day 1：1/2/8 卡等价
│   ├── run_fsdp_vs_ddp.{sh,ps1}  Day 2：DDP vs FSDP 三策略
│   ├── run_reshard.{sh,ps1}      Day 2：8 卡存 -> 4 卡恢复 -> verify
│   ├── run_ep_scan.{sh,ps1}      Day 3：EP=1/2/4/8
│   ├── run_moe_sweep.{sh,ps1}    Day 4：bias × aux_coef 扫描 -> CSV
│   ├── run_grpo_roles.{sh,ps1}   Day 5：三种角色放置
│   └── fault_kill_rank.{sh,ps1}  Day 5：kill rank，对比 async 错误处理开/关
└── tests/                    pytest（CPU + gloo，不需要 GPU）
```

所有脚本的产物都写到 `--out-dir` / `-OutDir` 指定的目录，没有隐含的写入位置。

## 4. 已知限制

1. **FSDP 需要 CUDA。** torch 的 `_init_device_handle` 在参数全在 CPU 上时会回落到
   `torch.cuda.current_device()`，没有 GPU 就抛
   `FSDP needs a non-CPU accelerator device`。torch 2.1.0 与本机的 2.14 行为一致。
   因此 `train_fsdp.py`、`ckpt_reshard.py`、`grpo_roles.py --placement policy_fsdp`
   在本机只能验证参数解析与那句可执行的报错（`require_cuda_for_fsdp`），
   真正的运行必须在公司 8×V100 上。
2. **`--dataset real` 需要 `datasets` 包**（MiniMind 的 `dataset/lm_dataset.py` 依赖它）。
   公司环境如果没有，用默认的 `--dataset tiny`：合成 token，按索引可复现，
   fixed-global-batch 等价实验本来就需要这个性质。
3. **教学版 EP 不做 kernel 融合**，dispatch/combine 是两次朴素 `all_to_all_single`，
   本地 expert 用 `index_select` + 逐专家 FFN。它的目的是让通信序列可见可测，
   不是性能基线；与 DeepSpeed-MoE / Tutel 的对照是选做。
4. **GRPO 是同步简化版**：每轮 rollout 之后只做一次 policy 更新，
   所以重要性采样比恒为 1、PPO 的 clip 项不激活，目标退化为 `-A·logπ`；
   KL 项默认关闭（`--kl-coef 0`），打开时会额外拷贝一份冻结的初始 policy 作参考。
   规则奖励只是可优化的信号源，**不是**语言质量指标。
5. **`torch.compile` 本周一律关闭**。Triton 2.1 声称支持 CC≥7.0，V100 满足但未实测。
6. **CPU 上的 gloo 支持 `all_to_all_single`**（本机 torch 2.14 已验证）；
   若在别的环境上不支持，`tests/test_ep_cpu.py` 会 skip 并写明原因，而不是失败。
7. **本机的 `torchrun` 不可用**：这台 Windows 机器上的 torch 2.14 CPU wheel 编译时
   没带 libuv，而 torchrun 的 rendezvous 会创建 libuv 版 TCPStore，直接抛
   `use_libuv was requested but PyTorch was built without libuv support`，
   `USE_LIBUV=0` 也压不住（该开关只影响用户自建的 TCPStore）。
   所以本机所有多进程验证都走 `torch.multiprocessing.spawn`（`tests/conftest.py`
   的 `spawn_dist` fixture 就是干这个的），公司 Linux 机器上照常用 `torchrun`。
   `scripts/*.sh`、`scripts/*.ps1` 里的 `torchrun` 命令是给目标机用的。

## 5. 公司数据与产物边界

- 所有脚本的日志、checkpoint、CSV、probe.json、审计 Markdown 都**只写到 `--out-dir`**。
  在公司机器上请把 `--out-dir` 指向公司内部路径。
- 本 lab **没有任何上传、同步、外发逻辑**，也不建议任何这样的做法。
- 从公司带出的只有**抽象值**：比值（step_time_ratio、mem_ratio、负载比）、
  布尔判定（pass / 是否连续 / checksum 是否一致）、计数（collectives_total、
  skipped_steps、被杀的 step 号）。
- `probe_company.py` 的 `--redact` 默认开启：不写主机名、不写绝对路径；
  `nvidia-smi topo -m` 的**矩阵在任何情况下都不写入 probe.json**，
  只输出连接类型的计数直方图与"是否全对称"。
- 不带出：loss 绝对值曲线、tokens/s 绝对值、显存绝对值、trace、图片、
  拓扑矩阵、数据样本、模型权重。

## 6. 本机验证状态（2026-09-05）

已执行并通过：`lab/tests` 全部用例（CPU + gloo，含 2 进程 DDP 等价、
2 进程 EP 往返、4 进程角色分组、kill rank 真实退出码）；
所有 `.py` 的 `py_compile`。
未在本机执行：任何 CUDA / fp16 / FSDP / 8 卡路径。
逐条状态见 `../02_LAB_GUIDE.md` 第 7 节。
