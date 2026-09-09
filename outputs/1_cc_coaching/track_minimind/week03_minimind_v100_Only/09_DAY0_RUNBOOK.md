# Day 0 操作手册 — 从「东西还在 Windows 上」到「八卡线路验通」

> 生成日期：2026-09-09 · 生成方：cc
> 配套 `03_TASK_CARDS/day0.md`。任务卡讲**为什么**和**判据**；这份手册讲**照着敲什么**。
> 预计：你在场 100 分钟。中间没有需要挂着跑的长任务，Day 0 一次坐完。

## 这份手册解决什么

Day 0 的目标不是训练任何东西，是把四件事一次钉死：

1. 代码和数据**到位了**（路径和这边写的完全对得上）；
2. Python 环境**能用**（不新建 conda，不动 torch 2.1.0）；
3. V100 的**真实能力**（8 卡在不在、bf16 是不是真的没有、SDPA 落哪个后端、int8 能不能调）；
4. **`torchrun` 起不起得来**（1 → 2 → 8 卡）。

第 4 条是全周唯一的早期闸门。写这份周包的机器没有 GPU，`torchrun` 在那边**一次都没跑起来过**（CPU 版 torch 缺 libuv）。所以所有多卡代码都是开环写的。Day 3 的字节账要 8 卡，Day 5 的 kill-rank 恢复要 8 卡——8 卡起不来，后面四天全部建立在一个跑不起来的前提上。**用五分钟在今天发现，不要用两小时在 Day 3 发现。**

---

## 阶段 A — 在 Windows 这边打包（10 分钟）

### A1. 打包周包本体

周包只有 1.2 MB，和 27 GB 的数据比可以忽略。在 Windows PowerShell 里：

```powershell
cd C:\Users\zzz_7893\Desktop\0_Projects\LLM_Training\outputs\1_cc_coaching\track_minimind

# 排除 __pycache__ 和本地跑出来的临时产物
tar -czf week03.tar.gz `
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' `
    week03_minimind_v100_Only

Get-Item week03.tar.gz | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,2)}}
```

预期：`week03.tar.gz`，**0.3–0.5 MB** 左右（压缩后）。
不对时先查：`tar` 在 Windows 10/11 自带（`C:\Windows\System32\tar.exe`）。没有就用 `Compress-Archive -Path week03_minimind_v100_Only -DestinationPath week03.zip`，V100 那边用 `unzip`。

### A2. 决定带多少数据

**跑完 Day 1–5 只需要 mini 层的 4 个文件，2.85 GiB。** 不是 23.83 GiB。

| 层 | 文件 | 体积 | 什么时候要 |
| --- | --- | --- | --- |
| **mini（必带）** | `pretrain_t2t_mini.jsonl`、`sft_t2t_mini.jsonl`、`dpo.jsonl`、`rlaif.jsonl` | **2.85 GiB** | Day 1–5 全部 |
| full（可不带） | 其余 7 个 | 20.98 GiB | 只有想跑长训练时 |
| 奖励模型（可不带） | `internlm2_1_8b_reward/` 11 个文件 | 3.17 GiB | Day 5 默认走规则奖励，用不到 |

**第一次进去只带 mini 层。** 数据搬运在内网是最贵的一步，先用 2.85 GiB 把整条链路跑通，确认路径合同没问题，再决定要不要搬剩下的 21 GiB。

**传输必须是二进制方式**（tar / scp / rsync）。文本模式的 FTP 或某些同步工具会把 jsonl 的 `\n` 改成 `\r\n`，字节数就对不上了——第 B4 步会当场抓到这个。

### A3. 一起带的两个可选文件

- `SHA256SUMS.txt`（如果外网下载那台机器上有）：放进 `MM_DATA_ROOT`，第 B4 步就能做全量哈希比对，而不只是前 16 位。
- MiniMind 仓库：V100 上 `git clone` 可用的话就在那边克隆，不用带。锁定 commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`。**Day 0 不强制要它**——没有它只是跳过第 B7 步的兼容审计。

---

## 阶段 B — 在 V100 上（90 分钟）

### B1. 落地与目录结构（5 分钟）

```bash
# 换成你自己的路径
export WORK=/your/path/to/minimind
mkdir -p $WORK && cd $WORK

tar -xzf week03.tar.gz          # 得到 week03_minimind_v100_Only/
ls week03_minimind_v100_Only/
```

预期：看到 `00_WEEK_CARD.md` … `09_DAY0_RUNBOOK.md`、`lab/`、`datasets/`、`weights/`、`runs/`。
不对时先查：如果只看到一堆文件没有目录，是 tar 时 `cd` 到了周包里面而不是它的父目录。

### B2. 设四个环境变量（5 分钟）

**这四条是全周唯一需要你按机器修改的东西。** 设完之后 `lab/` 里所有脚本都不用动。

```bash
cd $WORK/week03_minimind_v100_Only

export MM_DATA_ROOT=$WORK/data          # 你待会儿把 jsonl 放这儿的父目录
export MM_WEIGHTS_ROOT=$WORK/weights
export MM_RUNS_ROOT=/your/scratch/runs  # 大容量可写分区，不要放 home
export MINIMIND_ROOT=$WORK/MiniMind     # 没克隆就先不设

export HF_HUB_OFFLINE=1                 # 内网必设：让它立刻失败而不是挂着等超时
export PYTHONIOENCODING=utf-8
```

**把这几行写进一个 `env.sh`，以后每次开工 `source env.sh`。** 六天里每天都要。

`MM_RUNS_ROOT` 不要指向周包目录内部——那里面的东西是公司运行产物，放进仓库目录容易连同代码一起被拷出去。

### B3. 确认 Python 环境（5 分钟）

**不要新建 conda 环境，不要升级 torch。** 用公司那个已有的：

```bash
conda activate <你公司那个环境名>

python -c "
import sys, torch, numpy
print('python      ', sys.version.split()[0])
print('torch       ', torch.__version__, '| cuda', torch.version.cuda)
print('numpy       ', numpy.__version__)
print('cuda avail  ', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
"
```

预期：`torch 2.1.0`、`cuda avail True`、`device_count 8`。
**`numpy` 必须是 1.x**——torch ≤2.1 在 numpy 2.x 下 `.numpy()` / `.from_numpy()` 会失效。如果这里打出 2.x，先 `pip install "numpy<2"` 再继续，否则后面每一步都可能踩这个。

不对时先查：`device_count` 不是 8 → `nvidia-smi` 看是不是被 `CUDA_VISIBLE_DEVICES` 限制了，或者卡被别人占着。

主线代码**不需要装任何新包**，只用 `torch` + `numpy` + 标准库。所以这一步之后不该有 `pip install`。

### B4. 数据到位校验（10 分钟）· 这是 Day 0 的第一个证据

先把数据放好。目录名必须**逐字一致**：

```bash
mkdir -p $MM_DATA_ROOT/minimind_dataset
# 把 4 个 jsonl 放进去（用你的传输方式）
ls -l $MM_DATA_ROOT/minimind_dataset/
```

然后校验。**只带了 mini 层就加 `--tier mini`**，否则它会按 11 个文件报，你会看到 7 个本来就不该在那里的 MISSING：

```bash
python lab/scripts/check_data_layout.py --tier mini
```

预期：4 行 `OK`，末尾 `全部通过。这一行可以作为 Day 0 的证据字段 data_layout=PASS。`，退出码 0。

不对时先查：
- `MISSING` → 目录名或文件名不对。对照 `datasets/README.md` 第 2 节的目录树，注意大小写和下划线。
- `SIZE-MISMATCH` → **传输被截断，或者用了文本模式改了换行**。重新用二进制方式拷那几个文件。这一条抓到过就是抓到了，不要绕过去。

想连内容一起验（几分钟，值得做一次）：

```bash
python lab/scripts/check_data_layout.py --tier mini --sha256
```

带了 `SHA256SUMS.txt` 就自动做全量比对，否则比前 16 位。

### B5. CPU 单测：确认代码是完整的（15 分钟）

这一步验的不是 GPU，是**代码在传输过程中没有缺斤少两**，而且和这台机器的 torch 2.1.0 兼容。

```bash
python -m pytest lab/tests -q
```

预期：`368 passed, 6 skipped`（或更多 passed——有 GPU 的机器上原本跳过的 6 条会真跑）。
本机（无 GPU）跑出来是 `368 passed, 6 skipped`，用时 4–11 分钟。

**有一条会 FAIL，那是故意的**：`test_reshard_cpu.py::test_fsdp_sharded_save_load_placeholder` 在检测到 GPU 时会主动 `pytest.fail`，提示「到了有 GPU 的机器上请把这条改成真实的 FSDP 往返测试」。**看到它 FAILED 不是回归**，是一个留给你的施工标记。Day 3 会回来处理它。

不对时先查：大面积 `ImportError` → `conda activate` 忘了，或者激活到了别的环境。`ModuleNotFoundError: mm_v100` → 不在周包根目录下跑。

### B6. 环境探针：把 V100 的真实能力问一遍（10 分钟）

```bash
python lab/scripts/probe_env.py --out-dir $MM_RUNS_ROOT/day0_probe
```

它逐项检查并打印 `[PASS]` / `[FAIL]`。**关注这几项**：

| 项 | 期望 | 不是期望值时意味着什么 |
| --- | --- | --- |
| `torch_version_matches_2.1.0` | PASS | 环境不是你以为的那个 |
| `eight_gpus` | PASS（device_count=8） | 卡被占或被 `CUDA_VISIBLE_DEVICES` 限制 |
| `sm70_in_arch_list` | PASS | **这个 wheel 没为 V100 编译**，后面所有 CUDA kernel 都会有问题 |
| `bf16_absent_as_expected` | PASS（`is_bf16_supported()=False`） | 如果是 True，说明你不在 V100 上 |
| `fsdp_api_complete` / `dcp_importable` | PASS | torch 2.1 的 FSDP / 分布式 checkpoint API 齐不齐 |
| `disk_enough_for_day3` | PASS | `MM_RUNS_ROOT` 空间不够，先换个分区 |

还会记录几项**不阻塞**的信息（无论结果如何都继续）：`int8.callable_on_device`（`torch._int_mm` 在 sm70 上能不能调，预期 `false`——**`false` 是预期结果，不是故障**）、SDPA 三个后端哪个可用（预期 `flash: false`）、NCCL 版本、GradScaler 默认值、GPU 是否独占。

**`gpu_exclusive` 这一项要留意**：如果卡是共享的，Day 3 的所有 `step_time_ratio` 都不可比，只能用 `mem_ratio` 和通信次数下结论。

产物：`$MM_RUNS_ROOT/day0_probe/probe.json`。

### B7. MiniMind 兼容审计（10 分钟，可选）

只有克隆了 MiniMind 才做：

```bash
python lab/scripts/audit_compat.py --out-dir $MM_RUNS_ROOT/day0_probe
```

预期四个计数：默认 `bfloat16` 的脚本数（约 9）、用 autocast 但无 GradScaler 的处数（约 4，含 `train_grpo.py`）、`torch.compile` 出现处（约 8）、torch 2.1 里不存在的 API（应为 **0**）。

最后一项不是 0 就要看清楚是哪个 API——那意味着当前 MiniMind commit 用了 2.1 没有的东西。

没克隆 MiniMind 就跳过这步，不影响 Day 0 通过。

### B8. **wiring smoke — 全周唯一的早期闸门**（25 分钟）· 不可裁

逐级加卡，**一级不过就停下**，不要跳到下一级。

```bash
# 第一级：单进程。验的是模型、数据、训练循环本身
python lab/scripts/wiring_smoke.py \
    --out-dir $MM_RUNS_ROOT/day0_probe \
    --config smoke_v100 --sizes 1 --steps 2 \
    --launcher torchrun --backend nccl
```

```bash
# 第二级：2 卡。验的是 torchrun 的 rendezvous + NCCL 初始化
NCCL_DEBUG=WARN python lab/scripts/wiring_smoke.py \
    --out-dir $MM_RUNS_ROOT/day0_probe \
    --config smoke_v100 --sizes 2 --steps 2 \
    --launcher torchrun --backend nccl
```

```bash
# 第三级：8 卡。验的是全部 8 张卡能不能进同一个通信组
NCCL_DEBUG=WARN python lab/scripts/wiring_smoke.py \
    --out-dir $MM_RUNS_ROOT/day0_probe \
    --config smoke_v100 --sizes 8 --steps 2 \
    --launcher torchrun --backend nccl
```

也可以一条命令跑完三级（它内部按顺序来，失败就停）：

```bash
NCCL_DEBUG=WARN python lab/scripts/wiring_smoke.py \
    --out-dir $MM_RUNS_ROOT/day0_probe \
    --config smoke_v100 --sizes 1,2,8 --steps 2 \
    --launcher torchrun --backend nccl
```

预期：三级都打出 `started=true`，每级 2 步就结束。**每步只跑 2 步是刻意的**——这一步不验训练对不对，只验线路通不通。

**逐级的意义在于失败时能立刻定位**：

| 哪一级失败 | 大概率是什么 | 先查 |
| --- | --- | --- |
| 1 卡就失败 | 不是分布式问题，是模型/数据/环境 | 回到 B5，CPU 单测是不是真的过了 |
| 1 过、**2 失败** | `torchrun` 的 rendezvous 或 NCCL 初始化 | `NCCL_DEBUG=INFO` 重跑看它卡在哪；查 `MASTER_ADDR`/`MASTER_PORT` 是否被占；端口是否被防火墙挡 |
| 2 过、**8 失败** | 拓扑、配额、或某张卡有问题 | `nvidia-smi` 看 8 张卡是否都空闲；试 `--sizes 4` 二分定位 |
| 挂住不动 | NCCL 在等一个永远不来的 peer | 加 `NCCL_TIMEOUT_S=60` 让它超时报错而不是无限等 |

**这一步没过就不要往 Day 1 走。** 记 `BLOCKED`，把失败那一级的报错抽象描述（不含主机名、不含拓扑细节）带出来，我们一起看。

### B9. 生成证据字段（5 分钟）

```bash
python lab/scripts/make_evidence.py \
    --day 0 --skill env_probe_compat_data_contract \
    --run-dir $MM_RUNS_ROOT/day0_probe
```

它从上面几步的产物里聚合出 `evidence.json`，**只含抽象值**（布尔、计数、比值），不含原始数字序列、路径和拓扑。

**生成之后自己再看一眼再带出来——这一眼是你的责任，不是脚本的。**

---

## 今天结束时你应该能回答

1. 这 8 张卡是不是真的 8 张、是不是独占、bf16 是不是真的没有？
2. `torchrun` 在这台机器上起不起得来？到几卡为止？
3. 数据的字节数和这边的清单是不是逐字相等？
4. 这套代码在 torch 2.1.0 上跑单测是什么结果？

## 证据字段（填完给我）

```json
{
  "date": "2026-09-__",
  "week": 3,
  "day": 0,
  "card": "day0",
  "skill": "env_probe_compat_data_contract",
  "env": "v100",
  "ai_level": "A2",
  "primary_artifact": "$MM_RUNS_ROOT/day0_probe/evidence.json",
  "observations": {
    "device_count": 0,
    "capability_uniform": true,
    "bf16_supported": false,
    "sm70_in_arch_list": true,
    "torch_version": "2.1.0",
    "numpy_major": 1,
    "gpu_exclusive": true,
    "int8_matmul_available": false,
    "data_layout": "PASS",
    "pytest_tail": "368 passed, 6 skipped",
    "wiring_ws1_started": true,
    "wiring_ws2_started": true,
    "wiring_ws8_started": true
  },
  "status": "PASS",
  "failure_class": "none",
  "self_check": {"q1": "能", "q2": "能"},
  "time_spent_min": 0,
  "notes": "不含主机名、拓扑与绝对性能数字"
}
```

---

## 关于「机器可以 24 小时挂着」

Day 0 全部是短命令，没有需要挂着跑的东西，一次坐完就行。

从 **Day 1** 开始会有长任务。那时的用法是：

```bash
# 挂上就走，日志落盘，断开 ssh 也不受影响
nohup python lab/scripts/run_day.py --day 1 ... > $MM_RUNS_ROOT/day1_pretrain/nohup.log 2>&1 &
echo $!    # 记下 PID

# 回来之后
tail -50 $MM_RUNS_ROOT/day1_pretrain/nohup.log
```

有 `tmux` 或 `screen` 就更好，可以断开重连看实时输出。

**但今天先不要用这个。** Day 0 的每一步都要你看着输出判断，尤其 B8 的三级 wiring——挂后台跑掉了，你就失去了「哪一级断的」这个信息。
