# Week 03 实践篇：从空目录产出 V100 Hardware Card

> **全部步骤待执行、待验证。** 用户自述公司机器为8×V100 32GB、给定NVLink拓扑；必须以本机结果为准。公司PyTorch 2.1不升级，不创建Conda/venv。所有拓扑、NCCL日志、trace、代码与性能数字只留公司机器，不向个人机、聊天或外部服务导出。

## Day 0：只读环境探针与工程骨架

### 为什么做

冻结软件/硬件事实，确保后续脚本引用的目录先存在，并把“环境问题”与“训练问题”分开。

### 输入与前置检查

估算：无需外部数据；nccl-tests源码/构建通常不足1GB；30M/120M运行输出不足数GB。显存从batch1扫描，不预设能占满。完整8卡实验仅公司Linux适用；个人5070Ti只跑单卡部分。

`[公司 Linux]`：

```bash
which python
python -V
python -m pip --version
python -m pip check
nvidia-smi
nvidia-smi topo -m
lscpu -e=CPU,NODE,SOCKET
numactl -H
sed -n '/Cpus_allowed_list/p' /proc/self/status
git --version
make --version
```

`lscpu/numactl/sysfs`因容器权限失败时记 `FAIL-VISIBILITY`，不mount host sysfs、不提权绕过。

`[个人 PowerShell]`：

```powershell
python -V
python -m pip check
nvidia-smi
Get-Command git
Get-PSDrive -PSProvider FileSystem
```

### 本日要创建/修改的文件

`[公司 Linux]`：

```bash
mkdir -p week03-hardware/{env,topology,src,configs,reports,runs,third_party}
cd week03-hardware
touch env/env_snapshot.txt topology/topology_card.md reports/hardware_card.md reports/decision.md
python -m pip freeze > env/pip-freeze.txt
```

`[个人 PowerShell]`：

```powershell
New-Item -ItemType Directory -Force week03-hardware,week03-hardware/env,week03-hardware/topology,week03-hardware/src,week03-hardware/configs,week03-hardware/reports,week03-hardware/runs,week03-hardware/third_party | Out-Null
Set-Location week03-hardware
New-Item -ItemType File -Force env/env_snapshot.txt,topology/topology_card.md,reports/hardware_card.md,reports/decision.md | Out-Null
python -m pip freeze | Set-Content -Encoding utf8 env/pip-freeze.txt
```

目录：

```text
env/probe_env.py             torch/CUDA/cuDNN/NCCL/SM事实
src/cuda_correctness.py      单卡FP64 oracle、FP32/FP16前反向
src/distributed_probe.py     数值可验的all-reduce
src/transformer_bench.py     自包含30M/120M synthetic LM
topology/topology_card.md    物理ID/rank/CPU/P2P
runs/nccl_matrix.csv         配置×消息size×重复（由parser生成，不预建空证据）
reports/hardware_card.md     最终事实卡
runs/                        本机原始JSONL/摘要
third_party/nccl-tests       固定commit源码
```

### 实现

创建环境探针：

```python
# env/probe_env.py
import json, platform, subprocess, sys, torch
def safe(fn):
    try: return fn()
    except Exception as e: return {"error":type(e).__name__,"message":str(e)[:200]}
def bf16_for(i):
    with torch.cuda.device(i): return torch.cuda.is_bf16_supported()
d={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
   "torch_cuda_build":torch.version.cuda,"cuda_available":torch.cuda.is_available(),
   "cudnn":safe(lambda:torch.backends.cudnn.version()),
   "nccl":safe(lambda:torch.cuda.nccl.version()),"arch_list":safe(torch.cuda.get_arch_list)}
if torch.cuda.is_available():
    d["devices"]=[]
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        d["devices"].append({"logical_id":i,"name":p.name,"total_memory":p.total_memory,
                             "capability":[p.major,p.minor],"bf16":safe(lambda i=i:bf16_for(i))})
print(json.dumps(d,indent=2))
```

### 执行命令

`[公司 Linux]`：

```bash
set -euo pipefail
python -m py_compile env/probe_env.py
python env/probe_env.py | tee env/env_snapshot.txt
python -m torch.utils.collect_env >> env/env_snapshot.txt
nvidia-smi topo -m > topology/nvidia-smi-topo.txt
```

`[个人 PowerShell]`：

```powershell
python -m py_compile env/probe_env.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python env/probe_env.py | Tee-Object env/env_snapshot.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m torch.utils.collect_env | Add-Content env/env_snapshot.txt
nvidia-smi topo -m | Set-Content topology/nvidia-smi-topo.txt
```

### 预期观测（估算/示例，不是实测）

公司若确为V100，应看到capability 7.0、BF16通常false、arch list含sm_70；GPU数量/显存必须抄实值。个人结果另立，不混表。

### 验收条件

探针退出0；torch仍2.1.x；每张可见卡属性被记录；不确定NUMA明确标未知；无环境改动。

### 若失败，按什么顺序查

作业是否分配GPU → `CUDA_VISIBLE_DEVICES` → driver → torch build arch → cuDNN/NCCL import → 平台工单。发现CUDA13或缺sm70时不自行重装。

### 当日证据清单

`env_snapshot.txt`、`pip-freeze.txt`、原始topo文本、可见性限制。

## Day 1：逐卡 FP32/FP16 正确性

### 为什么做

先证明每张卡独立做对计算；任一卡失败都禁止进入多卡性能测试。

### 输入与前置检查

Day0设备清单；每次只暴露一张物理卡，使脚本内部逻辑device固定为0。

### 本日要创建/修改的文件

```python
# src/cuda_correctness.py
import argparse, json, torch
def relerr(a,b): return float((a-b).norm()/b.norm().clamp_min(1e-12))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--size",type=int,default=512); p.add_argument("--steps",type=int,default=20); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    g=torch.Generator().manual_seed(123); x=torch.randn(a.size,a.size,generator=g,dtype=torch.float64); y=torch.randn(a.size,a.size,generator=g,dtype=torch.float64)
    ref=x@y; xf=x.float().cuda(); yf=y.float().cuda(); out32=xf@yf; out16=xf.half()@yf.half()
    e32=relerr(out32.double().cpu(),ref); e16=relerr(out16.float().double().cpu(),ref)
    w=torch.nn.Linear(a.size,a.size).cuda(); opt=torch.optim.SGD(w.parameters(),lr=1e-4)
    finite=True
    for _ in range(a.steps):
        opt.zero_grad(set_to_none=True); loss=w(xf).float().square().mean(); loss.backward(); opt.step()
        finite &= bool(torch.isfinite(loss) and all(torch.isfinite(q).all() for q in w.parameters()))
    result={"visible_name":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0),
            "fp32_rel_l2":e32,"fp16_rel_l2":e16,"backward_finite":finite,"steps":a.steps}
    assert finite and e32<1e-5 and e16<5e-3, result
    print(json.dumps(result))
if __name__=="__main__": main()
```

### 实现

CPU float64是oracle；FP16容差较宽且只作故障筛查，不是数值精度承诺。脚本同时跑20步FP32 backward；后续Transformer再测AMP。

### 执行命令

`[公司 Linux]` 在调度允许时逐卡串行：

```bash
set -euo pipefail
python -m py_compile src/cuda_correctness.py
for gpu in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$gpu python src/cuda_correctness.py --size 512 --steps 20 | tee runs/correctness_gpu${gpu}.json; done
```

若不允许shell循环，就将物理ID逐项代入同一命令。`[个人 PowerShell]`：

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python -m py_compile src/cuda_correctness.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python src/cuda_correctness.py --size 512 --steps 20 | Tee-Object runs/correctness_gpu0.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

### 预期观测（估算/示例，不是实测）

每张卡输出finite=true，FP32误差显著小于FP16；具体误差不预填。阈值只是硬件/软件粗筛。

### 验收条件

所有实际计划参与训练的卡通过；名称/capability一致或差异有解释；无NaN/Inf。

### 若失败，按什么顺序查

复跑同物理卡 → 缩小size → 比FP32/FP16 → 查ECC/Xid（只读）→ 换进程/节点 → 最小复现交平台。不能跳过坏卡后伪称8卡基线。

### 当日证据清单

逐卡JSON、失败物理ID、容差与seed、平台工单号（如有）。

## Day 2：拓扑、P2P 与 PyTorch collective correctness

### 为什么做

验证rank映射和collective语义，再使用nccl-tests量上限。

### 输入与前置检查

Day1所有目标卡通过。候选映射：`0,1` NV2、`0,3` NV1、`0,5` SYS、四卡A `0,1,2,3`、四卡B `4,5,6,7`、八卡全体；每项以topo实值确认。

### 本日要创建/修改的文件

```python
# src/distributed_probe.py
import json, os, time, torch, torch.distributed as dist
def main():
    rank=int(os.environ["RANK"]); local=int(os.environ["LOCAL_RANK"]); world=int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local); dist.init_process_group("nccl")
    x=torch.full((1024,),rank+1,dtype=torch.float32,device=local); dist.all_reduce(x)
    expected=world*(world+1)/2; torch.testing.assert_close(x,torch.full_like(x,expected))
    n=16*1024*1024//4; z=torch.zeros(n,dtype=torch.float32,device=local)
    if rank==0:z.fill_(1)
    elif rank==1:z.fill_(-1)
    for _ in range(10): dist.all_reduce(z)
    torch.cuda.synchronize(); dist.barrier(); t0=time.perf_counter()
    for _ in range(50): dist.all_reduce(z)
    torch.cuda.synchronize(); dist.barrier(); elapsed=time.perf_counter()-t0
    if rank==0:
        algbw=(z.numel()*z.element_size()*50)/elapsed/1e9
        print(json.dumps({"world":world,"payload_bytes":z.numel()*z.element_size(),"iters":50,
                          "elapsed_s":elapsed,"algbw_GBps":algbw,"correct":True}))
    dist.destroy_process_group()
if __name__=="__main__": main()
```

在 `topology/topology_card.md` 手工写：物理ID、bus ID、CPU affinity、pair link、逻辑device、rank。不要复制用户自述表而不核验。

### 实现

每rank常数输入让错误映射/规约能被数值断言发现；16MiB payload只是smoke，不代表峰值。

### 执行命令

`[公司 Linux]`：

```bash
set -euo pipefail
python -m py_compile src/distributed_probe.py
python -c "import torch; print([[torch.cuda.can_device_access_peer(i,j) for j in range(torch.cuda.device_count())] for i in range(torch.cuda.device_count())])" > topology/p2p.txt
CUDA_VISIBLE_DEVICES=0,1 timeout 120s torchrun --standalone --nproc_per_node=2 src/distributed_probe.py | tee runs/probe_01.json
CUDA_VISIBLE_DEVICES=0,3 timeout 120s torchrun --standalone --nproc_per_node=2 src/distributed_probe.py | tee runs/probe_03.json
CUDA_VISIBLE_DEVICES=0,5 timeout 120s torchrun --standalone --nproc_per_node=2 src/distributed_probe.py | tee runs/probe_05.json
CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 120s torchrun --standalone --nproc_per_node=4 src/distributed_probe.py | tee runs/probe_4a.json
CUDA_VISIBLE_DEVICES=4,5,6,7 timeout 120s torchrun --standalone --nproc_per_node=4 src/distributed_probe.py | tee runs/probe_4b.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 120s torchrun --standalone --nproc_per_node=8 src/distributed_probe.py | tee runs/probe_8.json
```

`[个人5070 Ti]`：不适用多卡矩阵；只保留Day1。若个人确有多卡，另建hardware card，不能映射到公司结果。

### 预期观测（估算/示例，不是实测）

每个run有`correct:true`且world匹配；带宽只作PyTorch smoke。SYS/NV排序不是PASS条件。

### 验收条件

所有计划映射数值正确、无hang；物理→逻辑→rank可追溯；timeout不会遗留作业进程（按调度器确认）。

### 若失败，按什么顺序查

单卡 → 2卡同组 → rank/env打印 → rendezvous端口 → NCCL error → 最小pair → 平台。永久禁P2P/IB不是修复。

### 当日证据清单

`p2p.txt`、拓扑卡、六组probe JSON、hang最小复现。

## Day 3：固定 nccl-tests，测 pair/4/8 卡上限

### 为什么做

用标准工具区分训练框架开销与collective上限，并为FSDP补all-gather/reduce-scatter。

### 输入与前置检查

PyTorch probe全过；CUDA_HOME/make/compiler来自批准环境。固定commit `b4d5beebca8a76cf01335f724d154b9b9d394d96`。

### 本日要创建/修改的文件

`third_party`已存在；git clone将创建`third_party/nccl-tests`。把下面代码块保存为 `src/parse_nccl.py`；它从保留的原始 stdout 生成逐次 CSV 和中位数/波动汇总，不允许人工抄带宽数字。

```python
# src/parse_nccl.py
import argparse, csv, re, statistics
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r"^(all_reduce|all_gather|reduce_scatter)__([0-9]+)__g([0-9]+)__r([0-9]+)\.log$")
FIELDS = ["collective", "mapping", "gpus", "size", "dtype", "repeat", "time_us", "algbw", "busbw", "errors", "shared_load", "notes"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-repeats", type=int, default=3)
    p.add_argument("--expected-cases", type=int, default=8)
    p.add_argument("--shared-load", choices=["idle", "shared", "unknown"], default="unknown")
    a = p.parse_args()
    rows = []
    for log in sorted(Path(a.input).glob("*.log")):
        m = NAME.match(log.name)
        if not m:
            raise ValueError(f"unexpected log name: {log.name}")
        collective, mapping, gpus, repeat = m.groups()
        parsed = 0
        for line in log.read_text(errors="replace").splitlines():
            cols = line.split()
            if len(cols) < 13 or not cols[0].isdigit():
                continue
            try:
                size, dtype, time_us = int(cols[0]), cols[2], float(cols[5])
                algbw, busbw = float(cols[6]), float(cols[7])
                errors = int(float(cols[8]))
            except ValueError:
                continue
            rows.append({"collective": collective, "mapping": mapping, "gpus": int(gpus), "size": size,
                         "dtype": dtype, "repeat": int(repeat), "time_us": time_us,
                         "algbw": algbw, "busbw": busbw, "errors": errors,
                         "shared_load": a.shared_load, "notes": "out_of_place"})
            parsed += 1
        if parsed == 0:
            raise RuntimeError(f"no nccl-tests data rows parsed from {log}")
    if not rows:
        raise RuntimeError("no nccl-tests logs found")
    cases = {(r["collective"], r["mapping"], r["gpus"]) for r in rows}
    if len(cases) != a.expected_cases:
        raise RuntimeError(f"case count {len(cases)} != {a.expected_cases}: {sorted(cases)}")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["collective"], r["mapping"], r["gpus"], r["size"], r["dtype"])].append(r)
    expected = set(range(1, a.expected_repeats + 1))
    for key, group in groups.items():
        repeats = {r["repeat"] for r in group}
        if repeats != expected:
            raise RuntimeError(f"{key}: repeats {sorted(repeats)} != {sorted(expected)}")
        if any(r["errors"] != 0 for r in group):
            raise RuntimeError(f"{key}: nccl-tests reported errors")
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    summary = out.with_name(out.stem + "_summary.csv")
    names = ["collective", "mapping", "gpus", "size", "dtype", "repeats", "time_us_median", "algbw_median", "busbw_median", "busbw_min", "busbw_max", "busbw_cv", "errors_sum"]
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names); w.writeheader()
        for key, group in sorted(groups.items()):
            bw = [r["busbw"] for r in group]; mean = statistics.mean(bw)
            row = dict(zip(names[:5], key)); row.update({"repeats": len(group),
                "time_us_median": statistics.median(r["time_us"] for r in group),
                "algbw_median": statistics.median(r["algbw"] for r in group),
                "busbw_median": statistics.median(bw), "busbw_min": min(bw), "busbw_max": max(bw),
                "busbw_cv": statistics.pstdev(bw) / mean if mean else 0.0,
                "errors_sum": sum(r["errors"] for r in group)})
            w.writerow(row)
    print({"raw_rows": len(rows), "groups": len(groups), "raw_csv": str(out), "summary_csv": str(summary)})

if __name__ == "__main__":
    main()
```

### 实现

使用官方build，不修改算法。单机`MPI=0`；如果Makefile探测CUDA/NCCL失败，记录原错而非切换未知编译器。

### 执行命令

`[公司 Linux]`：

```bash
set -euo pipefail
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/nccl-tests.git third_party/nccl-tests
git -C third_party/nccl-tests checkout b4d5beebca8a76cf01335f724d154b9b9d394d96
git -C third_party/nccl-tests rev-parse HEAD | tee runs/nccl-tests-commit.txt
test "$(cat runs/nccl-tests-commit.txt)" = "b4d5beebca8a76cf01335f724d154b9b9d394d96"
make -C third_party/nccl-tests MPI=0 -j2 2>&1 | tee runs/nccl-build.log
python -m py_compile src/parse_nccl.py
```

先启用 `pipefail`，再让每个映射完整重复 3 次。每次 stdout 都保存在本机；任何一个 `nccl-tests` 失败都会让整段命令失败，而不会被 `tee` 掩盖：

```bash
set -euo pipefail
mkdir -p runs/nccl
run_case () {
  collective="$1"; mapping="$2"; devices="$3"; gpus="$4"; max_bytes="$5"; repeat="$6"
  log="runs/nccl/${collective}__${mapping}__g${gpus}__r${repeat}.log"
  timeout 300s env CUDA_VISIBLE_DEVICES="$devices" "third_party/nccl-tests/build/${collective}_perf" -b 8M -e "$max_bytes" -f 2 -g "$gpus" 2>&1 | tee "$log"
}
for repeat in 1 2 3; do
  run_case all_reduce 01 0,1 2 1G "$repeat"
  run_case all_reduce 03 0,3 2 1G "$repeat"
  run_case all_reduce 05 0,5 2 1G "$repeat"
  run_case all_reduce 0123 0,1,2,3 4 1G "$repeat"
  run_case all_reduce 4567 4,5,6,7 4 1G "$repeat"
  run_case all_reduce 01234567 0,1,2,3,4,5,6,7 8 1G "$repeat"
  run_case all_gather 01234567 0,1,2,3,4,5,6,7 8 256M "$repeat"
  run_case reduce_scatter 01234567 0,1,2,3,4,5,6,7 8 256M "$repeat"
done
python src/parse_nccl.py --input runs/nccl --output runs/nccl_matrix.csv --expected-repeats 3 --expected-cases 8 --shared-load unknown
```

只有一次graph诊断：

```bash
timeout 300s env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 third_party/nccl-tests/build/all_reduce_perf -b 256M -e 256M -f 2 -g 8 > runs/nccl_graph_local.log 2>&1
```

若 `-e 1G` 造成内存风险，按smoke结果降到256M并记录，不硬跑OOM。个人单卡不做NCCL对照。

### 预期观测（估算/示例，不是实测）

错误数0；各size有time/algbw/busbw。预期NV2可能优于SYS只是假设；共享机器波动可使顺序反转。

### 验收条件

六组 all-reduce correctness、每组 3 个不同 repeat；`nccl_matrix.csv` 与 `nccl_matrix_summary.csv` 均生成，汇总含中位数、min/max、CV 和错误总数；8 卡 all-gather/reduce-scatter 至少一个中等 size；commit/hash 记录。

### 若失败，按什么顺序查

build日志 → CUDA_HOME/NCCL headers → 2卡8M → P2P probe → 单次INFO日志 → 平台。标准工具不可得则PyTorch probe通过correctness，带宽栏写`MISSING/INCONCLUSIVE`。

### 当日证据清单

`nccl-tests-commit.txt`、`nccl-build.log`、24 个原始日志、`nccl_matrix.csv`、`nccl_matrix_summary.csv`、本机graph日志、共享负载标记。

## Day 4：自包含 30M/120M FP32/FP16 基线

### 为什么做

在无I/O载荷上比较相同模型/批次的精度、吞吐与显存，不把NCCL带宽当训练性能。

### 输入与前置检查

Day1单卡正确；选一张获批空闲卡；本日不需要Week01文件，避免“读者应该还留着”的隐含依赖。

### 本日要创建/修改的文件

```python
# src/transformer_bench.py
import argparse, json, math, statistics, time, torch
from torch import nn
import torch.nn.functional as F
class LM(nn.Module):
    def __init__(self,size,T,V=8192):
        super().__init__(); d,L,h=(512,8,8) if size=="small" else (768,16,12); self.T=T
        self.emb=nn.Embedding(V,d); self.pos=nn.Embedding(T,d)
        layer=nn.TransformerEncoderLayer(d,h,4*d,dropout=0.0,batch_first=True,norm_first=True,activation="gelu")
        self.blocks=nn.TransformerEncoder(layer,L); self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,V,bias=False); self.head.weight=self.emb.weight
        self.register_buffer("mask",torch.triu(torch.ones(T,T,dtype=torch.bool),1),persistent=False)
    def forward(self,x,y):
        z=self.emb(x)+self.pos(torch.arange(x.size(1),device=x.device))[None]
        logits=self.head(self.norm(self.blocks(z,mask=self.mask[:x.size(1),:x.size(1)])))
        return F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),y.reshape(-1))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--size",choices=["small","medium"],required=True); p.add_argument("--precision",choices=["fp32","fp16"],required=True)
    p.add_argument("--batch",type=int,default=1); p.add_argument("--seq",type=int,default=256); p.add_argument("--warmup",type=int,default=20); p.add_argument("--steps",type=int,default=100)
    p.add_argument("--fault-step",type=int,default=-1); p.add_argument("--output",required=True); a=p.parse_args()
    torch.manual_seed(7); torch.cuda.manual_seed_all(7); dev=torch.device("cuda"); m=LM(a.size,a.seq).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
    scaler=torch.cuda.amp.GradScaler(enabled=a.precision=="fp16"); initial_scale=float(scaler.get_scale()); min_scale=initial_scale; g=torch.Generator().manual_seed(8)
    attempted=successful=skipped=0; attempted_ms=[]; successful_ms=[]; losses=[]; grad=float("nan")
    fault_skipped=None; recovered_after_fault=False
    target_success=a.warmup+a.steps; max_attempts=target_success+max(20,a.steps//5); torch.cuda.reset_peak_memory_stats()
    while successful<target_success:
        timed_phase=successful>=a.warmup; attempt_index=attempted
        z=torch.randint(0,8192,(a.batch,a.seq+1),generator=g).to(dev); x,y=z[:,:-1],z[:,1:]; opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0=time.perf_counter()
        with torch.cuda.amp.autocast(enabled=a.precision=="fp16",dtype=torch.float16): loss=m(x,y)
        if attempt_index==a.fault_step: loss=loss*torch.tensor(float("inf"),device=loss.device)
        scaler.scale(loss).backward(); scaler.unscale_(opt); grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.0))
        if not scaler.is_enabled() and not math.isfinite(grad): raise FloatingPointError(f"FP32 nonfinite grad at attempt {attempt_index}")
        before=float(scaler.get_scale()); scaler.step(opt); scaler.update(); min_scale=min(min_scale,float(scaler.get_scale())); success=math.isfinite(grad) and float(scaler.get_scale())>=before
        torch.cuda.synchronize(); ms=1000*(time.perf_counter()-t0)
        attempted+=1; skipped+=int(not success)
        if attempt_index==a.fault_step: fault_skipped=not success
        if a.fault_step>=0 and attempt_index>a.fault_step and success: recovered_after_fault=True
        if timed_phase: attempted_ms.append(ms)
        if success:
            successful+=1
            if timed_phase: successful_ms.append(ms); losses.append(float(loss.detach()))
        if attempted>=max_attempts and successful<target_success: raise RuntimeError("too many AMP overflows to reach requested successful updates")
        if not torch.isfinite(loss.detach()) and attempt_index!=a.fault_step: raise FloatingPointError(attempt_index)
    parameters_finite=all(bool(torch.isfinite(p).all()) for p in m.parameters())
    if not parameters_finite: raise FloatingPointError("nonfinite parameter")
    elapsed=sum(attempted_ms)/1000
    result={"size":a.size,"precision":a.precision,"parameters":sum(p.numel() for p in m.parameters()),"batch":a.batch,"seq":a.seq,
            "attempted_total":attempted,"successful_total":successful,"timed_attempts":len(attempted_ms),"timed_successful":len(successful_ms),
            "timed_skipped":len(attempted_ms)-len(successful_ms),"requested_timed_successful":a.steps,"warmup_successful":a.warmup,
            "fault_attempt":a.fault_step,"fault_skipped":fault_skipped,"recovered_after_fault":recovered_after_fault,
            "initial_scale":initial_scale,"minimum_scale":min_scale,"successful_step_ms_p50":statistics.median(successful_ms),
            "successful_step_ms_p95":sorted(successful_ms)[max(0,int(.95*len(successful_ms))-1)],
            "attempted_tokens_per_s":a.batch*a.seq*len(attempted_ms)/elapsed,
            "effective_tokens_per_s":a.batch*a.seq*len(successful_ms)/elapsed,"peak_allocated":torch.cuda.max_memory_allocated(),
            "peak_reserved":torch.cuda.max_memory_reserved(),"final_scale":float(scaler.get_scale()),"skipped":skipped,"final_loss":losses[-1],"final_grad_norm":grad,
            "parameters_finite":parameters_finite}
    open(a.output,"w").write(json.dumps(result,indent=2)+"\n"); print(json.dumps(result))
if __name__=="__main__": main()
```

### 实现

synthetic 序列固定 seed，移除 I/O；计时包含 forward/backward/update 并显式同步；参数量实算。循环以 **successful update** 达到 `warmup+steps` 才停止，同时报告 attempted/successful、浪费在 skip 上的时间、attempted throughput 与 effective throughput；fault 只用于 Day5 独立 run。

### 执行命令

先1-step/20-step smoke：

`[公司 Linux]`：

```bash
python -m py_compile src/transformer_bench.py
CUDA_VISIBLE_DEVICES=0 python src/transformer_bench.py --size small --precision fp32 --batch 1 --seq 256 --warmup 2 --steps 20 --output runs/small_fp32_smoke.json
CUDA_VISIBLE_DEVICES=0 python src/transformer_bench.py --size small --precision fp16 --batch 1 --seq 256 --warmup 2 --steps 20 --output runs/small_fp16_smoke.json
```

smoke通过后同workload基线：

```bash
CUDA_VISIBLE_DEVICES=0 python src/transformer_bench.py --size small --precision fp32 --batch 4 --seq 256 --warmup 20 --steps 100 --output runs/small_fp32.json
CUDA_VISIBLE_DEVICES=0 python src/transformer_bench.py --size small --precision fp16 --batch 4 --seq 256 --warmup 20 --steps 100 --output runs/small_fp16.json
CUDA_VISIBLE_DEVICES=0 python src/transformer_bench.py --size medium --precision fp32 --batch 1 --seq 256 --warmup 20 --steps 100 --output runs/medium_fp32.json
CUDA_VISIBLE_DEVICES=0 python src/transformer_bench.py --size medium --precision fp16 --batch 1 --seq 256 --warmup 20 --steps 100 --output runs/medium_fp16.json
CUDA_VISIBLE_DEVICES=0 python src/transformer_bench.py --size medium --precision fp16 --batch 1 --seq 256 --warmup 20 --steps 200 --output runs/medium_fp16_stability200.json
```

前四条是可直接复制的同 workload 对照；最后一条 200-successful-update 命令只作 FP16 stability run，不进入速度比。`[个人 PowerShell]` 只跑 small 公开脚本：

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python src/transformer_bench.py --size small --precision fp16 --batch 1 --seq 256 --warmup 20 --steps 100 --output runs/personal_small_fp16.json
```

### 预期观测（估算/示例，不是实测）

参数量接近目标但以脚本值为准；FP16 可能更快但不是 PASS 硬条件；200-successful-update run 无非有限、skip 低。若发生 skip，`attempted_tokens_per_s` 描述实际尝试工作，`effective_tokens_per_s` 才是成功更新口径。

### 验收条件

相同 workload 对照；warmup 排除；100 个 timed successful updates 的比较口径一致；额外 200-successful-update FP16 finite、warmup 后 skip `<1%`；显存峰值记录；若 FP16 不快能用 Profiler/利用率/shape 解释。

### 若失败，按什么顺序查

small FP32 → small FP16 → loss/scale/grad → medium batch1 → OOM阶段 → torch profiler短表。先降batch，不改模型混淆比较。

### 当日证据清单

四组同 workload JSON、一个独立 stability JSON、参数量、attempted/successful 计时口径、峰值显存、FP16 skip、共享负载。

## Day 5：故障注入、Profiler 与 Hardware Card

### 为什么做

验证overflow能被检测并恢复，将所有事实/推断/未知项整理成后续周可复用基线。

### 输入与前置检查

Day4正常FP16 run通过；故障run使用新输出文件，不覆盖基线。

### 本日要创建/修改的文件

填写已创建的 `reports/hardware_card.md`：环境、可见GPU、禁用特性、逐卡正确性、topology/rank映射、NCCL矩阵、Transformer对照、显存包络、可见性限制、默认配置、未知项。每项标 `[探针事实]`、`[实测]`、`[用户自述]` 或 `[推断]`。

### 实现

故障只在step25把loss放大；GradScaler应回退/skip，后续恢复。Profiler用内置API的短段表，不输出到外部。

### 执行命令

`[公司 Linux]`：

```bash
CUDA_VISIBLE_DEVICES=0 python src/transformer_bench.py --size small --precision fp16 --batch 1 --seq 256 --warmup 20 --steps 60 --fault-step 25 --output runs/fp16_fault.json
python -c "import json; r=json.load(open('runs/fp16_fault.json')); assert r['fault_skipped'] and r['recovered_after_fault'] and r['minimum_scale']<r['initial_scale'] and r['timed_successful']==60; print('AMP_FAULT_PASS',r['attempted_total'],r['successful_total'])"
CUDA_VISIBLE_DEVICES=0 python -c "import torch; from torch.profiler import profile,ProfilerActivity; x=torch.randn(4096,4096,device='cuda',dtype=torch.float16); p=profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True); p.__enter__(); y=x@x; torch.cuda.synchronize(); p.__exit__(None,None,None); print(p.key_averages().table(sort_by='self_cuda_time_total',row_limit=15))" > runs/profiler_table_local.txt
```

`[个人 PowerShell]` 可跑同一fault脚本验证个人AMP，但不写入公司Hardware Card。

### 预期观测（估算/示例，不是实测）

fault附近scale下降/skip至少一次，后续正常step继续且参数finite。若1e20没有触发，可在独立run增大；不得事后修改正常baseline。

### 验收条件

Hardware Card十项齐全；禁用BF16/FA2/CUDA13专用路径；collective correctness全过或缺口明确；故障被检测且恢复；Profiler表解释主要算子。

### 若失败，按什么顺序查

确认fault step在warmup+timed总区间 → 检查loss是否真的非有限/grad是否inf → scaler是否enabled → 参数是否在skip后未更新 → 移除fault重跑normal smoke。

### 当日证据清单

`fp16_fault.json`、本机Profiler表、完成的Hardware Card、`decision.md`。任何未执行项明确写 `INCONCLUSIVE`。

## 最终发布门

逐卡算术、rank映射、collective数值、nccl-tests错误数、重复波动、同步计时、同workload FP32/FP16、200-step stability、故障恢复与事实标签缺一不可。本文没有声称任何硬件或性能已经实测。

技术链接核验日期：**2026-09-03**：[nccl-tests](https://github.com/NVIDIA/nccl-tests)、[PyTorch 2.1 distributed](https://pytorch.org/docs/2.1/distributed.html)、[PyTorch 2.1 AMP](https://pytorch.org/docs/2.1/notes/amp_examples.html)、[PyTorch 2.1 Profiler](https://pytorch.org/docs/2.1/profiler.html)。
