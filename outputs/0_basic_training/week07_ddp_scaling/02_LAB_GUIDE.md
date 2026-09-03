# Week 07 实践篇：从空目录复现 DDP 正确性与扩展曲线

> 核验日期：2026-09-03。本文所有数值均是“预期/示例”，命令尚未在目标机器实跑。公司 8×V100、网络与 PyTorch 2.1 均来自用户自述，先核验再运行。公司机器产生的代码副本、数据、日志、trace、截图、checkpoint 和拓扑信息只留在公司批准位置，禁止外传。

## 0. 本周交付与判定口径

从一个空目录创建唯一核心脚本 `src/ddp_lab.py`，依次完成 CPU 单测、Gloo collective、单卡基线、2 卡 smoke、断点续训、1/2/4/8 卡强扩展、弱扩展、拓扑对照、独立评估和重复采样故障注入。最终证据是命令、退出码、环境清单和机器可读 JSONL；不是挑选过的终端截图。

三条资源路径必须分开：

| 路径 | 本周用途 | 硬边界 |
|---|---|---|
| 公司 Linux、8×V100（待核验） | 主结果：1/2/4/8 卡 DDP | PyTorch 2.1 是约束；V100 用 FP16，不把 BF16 当可用；产物不出公司 |
| 个人 5070Ti（待核验） | 单卡代码冒烟与日志格式检查 | 不伪造多卡结论；不接收公司产物 |
| 可选 H100（待核验） | 同脚本 FP16 对照，可另做 BF16 | 不能替代 V100 主证据；硬件与软件栈单独记录 |

止损规则：数据 ID 重复、任一 rank 非有限 loss/梯度、参数校验和跨 rank 不一致、通信 hang、OOM 或断点恢复 step 不连续时，立即停正式实验；只保留现场证据并转入最小复现。

## 1. 来源、版本、许可与离线替代

本实验不下载模型或数据；token 由确定性公式在内存生成，因此不存在数据许可证或缓存污染。唯一运行依赖是目标环境已有的 Python 与 PyTorch。

| 对象 | 固定方式 | 许可/状态 | 获取与完整性 |
|---|---|---|---|
| PyTorch 2.1 | 公司现有安装，记录完整 `torch.__version__` | BSD-style；以公司批准镜像为准 | 不升级；保存 `pip freeze` 与导入路径 |
| `torch.distributed` / DDP | 随上述 PyTorch 安装 | 同 PyTorch | 运行 collective 自检 |
| 合成 token | `sample_id` 与位置的纯函数 | 本教程现场生成，无外部许可 | 日志保存首批 ID 与摘要 |
| 本文脚本 | 文内完整创建 | 项目内部教学代码 | `sha256sum src/ddp_lab.py` |

官方一手资料：[DDP](https://pytorch.org/docs/2.1/generated/torch.nn.parallel.DistributedDataParallel.html)、[`torchrun`](https://pytorch.org/docs/2.1/elastic/run.html)、[PyTorch 2.1 distributed/NCCL 环境变量](https://pytorch.org/docs/2.1/distributed.html#other-nccl-environment-variables)、[AMP 示例](https://pytorch.org/docs/2.1/notes/amp_examples.html)。链接于 2026-09-03 核验；实际安装的 2.1.x 文档/源码优先。

离线时无需联网。若公司镜像缺少 PyTorch，不在本周私自 `pip install`；把预检输出交给管理员。不要把个人 wheel、公司日志或 trace 在两条路径间搬运。

## 2. 环境预检：不新建 conda/venv

在目标机器已有、获批环境中执行：

```bash
which python
python -VV
python -m pip --version
python -m pip freeze > /tmp/week07-pip-freeze.txt
python - <<'PY'
import os, socket, torch
print({
    "torch": torch.__version__, "torch_file": torch.__file__,
    "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(), "host": socket.gethostname(),
    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    "nccl_available": torch.distributed.is_nccl_available(),
    "gloo_available": torch.distributed.is_gloo_available(),
})
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory, p.major, p.minor)
PY
nvidia-smi -L
nvidia-smi topo -m
```

公司路径若 `torch.__version__` 不是已批准的 2.1.x，状态记 `ENV-MISMATCH`，先确认而非升级。V100 的 compute capability 通常为 7.0，原生 BF16 不应假设可用；本手册显式用 FP16。依赖审计没有安装动作，故回滚为零；本文不需要额外 profiler 包。

资源估算（不是实测）：正式约 3.3 亿参数；FP16 参数约 0.66 GB，但 Adam 状态、FP32 master/梯度和激活使单卡占用显著增加。V100 16 GB 先用 `micro-batch=1, seq=256`；超过显存的 85% 或出现 OOM 就降 `seq`/层数并标记配置变化，绝不悄悄改变。

## 3. 目录树与唯一实现

最终目录：

```text
week07-ddp/
├── src/ddp_lab.py
├── artifacts/checkpoints/
├── logs/
├── traces/
└── evidence/
```

下面的 Day 1 会先创建所有目录与脚本。后续没有凭空出现的 `src/` 路径。

## Day 1：创建实现并完成静态/CPU 契约检查

### 为什么做

把采样、DDP、梯度累积、检查点和评估固定在一个可审计实现中，先排除语法和 shape 错误。

### 输入与前置检查

输入仅为预检通过的现有 Python/PyTorch；当前目录应是允许写入但不含公司敏感材料的实验位置。

### 本日要创建/修改的文件

创建 `week07-ddp/src/ddp_lab.py` 以及产物目录。

### 实现

```bash
mkdir -p week07-ddp/{src,artifacts/checkpoints,logs,traces,evidence}
cd week07-ddp
cat > src/ddp_lab.py <<'PY'
#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["unit", "collective", "train", "eval", "compare"], default="train")
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--vocab", type=int, default=2048)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--amp", choices=["none", "fp16"], default="fp16")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--profile-steps", type=int, default=0)
    p.add_argument("--benchmark-warmup", type=int, default=0)
    p.add_argument("--output", default="artifacts/checkpoints/smoke")
    p.add_argument("--log", default="logs/train.jsonl")
    p.add_argument("--trace-dir", default="traces")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--inject-duplicate-data", action="store_true")
    p.add_argument("--inject-nan-rank", type=int, default=-1)
    p.add_argument("--checkpoint-a", default="")
    p.add_argument("--checkpoint-b", default="")
    p.add_argument("--compare-atol", type=float, default=1e-6)
    p.add_argument("--compare-rtol", type=float, default=1e-5)
    return p.parse_args()


def env_int(name, default):
    return int(os.environ.get(name, default))


def distributed_context():
    world = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=5))
        if dist.get_world_size() != world or dist.get_rank() != rank:
            raise RuntimeError("launcher and process-group ranks disagree")
    return rank, local_rank, world, device


def seed_all(seed, rank=0):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


class TinyLM(nn.Module):
    def __init__(self, vocab, dim, heads, layers):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, token_ids):
        h = self.embed(token_ids)
        h = self.blocks(h)
        return self.head(self.norm(h))


def config_dict(a, world):
    keys=("seed","micro_batch","accum","seq","vocab","dim","heads","layers","lr","amp")
    return {**{k:getattr(a,k) for k in keys},"world_size":world}


def sample_ids(step, micro_step, rank, world, micro_batch, accum, duplicate=False):
    global_micro = step * accum + micro_step
    base = global_micro * world * micro_batch
    owner = 0 if duplicate and rank > 0 else rank
    return torch.arange(base + owner * micro_batch, base + (owner + 1) * micro_batch, dtype=torch.long)


def make_batch(ids, seq, vocab, device):
    pos = torch.arange(seq, dtype=torch.long).view(1, seq)
    x = (ids.view(-1, 1) * 104729 + pos * 1543 + 17) % vocab
    y = (x * 31 + pos * 7 + 3) % vocab
    return x.to(device), y.to(device)


def audit_ids(ids, world):
    local = ids.tolist()
    gathered = [None for _ in range(world)]
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    flat = [x for part in gathered for x in part]
    if len(flat) != len(set(flat)):
        raise RuntimeError(f"DATA_DUPLICATION ids={gathered}")
    return gathered


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def checksum(model, device, world):
    value = torch.zeros((), dtype=torch.float64, device=device)
    for p in unwrap(model).parameters():
        value += p.detach().double().sum()
    lo, hi = value.clone(), value.clone()
    if world > 1:
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    return float(value.item()), float((hi - lo).abs().item())


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_completed(path):
    path = Path(path)
    if (path / "COMPLETED").is_file():
        return path
    candidates = sorted(p for p in path.glob("ckpt_step*") if (p / "COMPLETED").is_file())
    if not candidates:
        raise RuntimeError(f"no completed checkpoint under {path}")
    return candidates[-1]


def validate_completed(path):
    target = resolve_completed(path)
    marker = json.loads((target / "COMPLETED").read_text(encoding="utf-8"))
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != marker.get("manifest_sha256"):
        raise RuntimeError(f"checkpoint manifest is missing or unbound: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_torch = str(torch.__version__)
    if marker.get("format_version") != 1 or manifest.get("format_version") != 1:
        raise RuntimeError("unsupported checkpoint format_version")
    if marker.get("step") != manifest.get("step"):
        raise RuntimeError("checkpoint marker/manifest step mismatch")
    if marker.get("torch_version") != current_torch or manifest.get("torch_version") != current_torch:
        raise RuntimeError(
            f"torch version mismatch before state load: saved={manifest.get('torch_version')} "
            f"current={current_torch}"
        )
    files = manifest.get("files")
    hashes = manifest.get("sha256")
    if not isinstance(files, list) or not isinstance(hashes, dict) or set(files) != set(hashes):
        raise RuntimeError("checkpoint manifest files/hash set is incomplete")
    saved_world = manifest.get("world_size")
    if not isinstance(saved_world, int) or saved_world < 1:
        raise RuntimeError(f"invalid checkpoint world_size: {saved_world}")
    expected_files = {"shared.pt"} | {f"rng_rank{r}.pt" for r in range(saved_world)}
    if set(files) != expected_files:
        raise RuntimeError(f"checkpoint manifest file set differs: got={set(files)} expected={expected_files}")
    for rel, expected in hashes.items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise RuntimeError(f"unsafe checkpoint manifest path: {rel}")
        file_path = target / rel
        if not file_path.is_file() or sha256_file(file_path) != expected:
            raise RuntimeError(f"checkpoint file missing/corrupt: {file_path}")
    return target, manifest


def save_checkpoint(a, model, optim, scaler, step, attempted, successful, rank, world):
    root = Path(a.output)
    target = root / f"ckpt_step{step:06d}"
    temporary = root / f".tmp_ckpt_step{step:06d}"
    conflict = torch.tensor(int(rank == 0 and (target.exists() or temporary.exists())), device=next(model.parameters()).device)
    if world > 1:
        dist.broadcast(conflict, src=0)
    if int(conflict):
        raise RuntimeError(f"refuse to overwrite checkpoint step {step}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        temporary.mkdir(parents=False, exist_ok=False)
    if world > 1:
        dist.barrier()
    if rank == 0:
        shared = {
            "step": step, "attempted_updates": attempted, "successful_updates": successful,
            "config": config_dict(a, world), "model": unwrap(model).state_dict(),
            "optimizer": optim.state_dict(), "scaler": scaler.state_dict(),
            "torch_version": str(torch.__version__),
        }
        atomic_torch_save(shared, temporary / "shared.pt")
    rng = {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device=torch.cuda.current_device()) if torch.cuda.is_available() else None,
    }
    atomic_torch_save(rng, temporary / f"rng_rank{rank}.pt")
    if world > 1:
        dist.barrier()
    if rank == 0:
        files = ["shared.pt"] + [f"rng_rank{r}.pt" for r in range(world)]
        manifest = {
            "format_version": 1,
            "step": step, "attempted_updates": attempted, "successful_updates": successful,
            "world_size": world, "torch_version": str(torch.__version__), "files": files,
            "sha256": {name: sha256_file(temporary / name) for name in files},
        }
        atomic_text(temporary / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        manifest_sha = sha256_file(temporary / "manifest.json")
        os.replace(temporary, target)
        atomic_text(target / "COMPLETED", json.dumps({
            "format_version": 1, "step": step, "torch_version": str(torch.__version__),
            "manifest_sha256": manifest_sha,
        }, sort_keys=True) + "\n")
    if world > 1:
        dist.barrier()
    return target


def load_checkpoint(a, model, optim, scaler, rank, world, device):
    target, manifest = validate_completed(a.output)
    shared = torch.load(target / "shared.pt", map_location=device)
    if shared.get("torch_version") != str(torch.__version__):
        raise RuntimeError(f"torch version mismatch: saved={shared.get('torch_version')} current={torch.__version__}")
    if shared["config"] != config_dict(a, world):
        raise RuntimeError(f"resume config mismatch: saved={shared['config']} current={config_dict(a, world)}")
    if (shared["step"] != manifest["step"] or
            shared["attempted_updates"] != manifest["attempted_updates"] or
            shared["successful_updates"] != manifest["successful_updates"] or
            shared["successful_updates"] != shared["step"] or
            shared["attempted_updates"] < shared["successful_updates"]):
        raise RuntimeError("checkpoint clocks disagree")
    unwrap(model).load_state_dict(shared["model"])
    optim.load_state_dict(shared["optimizer"])
    scaler.load_state_dict(shared["scaler"])
    rng_file = target / f"rng_rank{rank}.pt"
    if not rng_file.is_file():
        raise RuntimeError(f"missing {rng_file}; world-size-changing resume is not supported")
    rng = torch.load(rng_file, map_location="cpu")
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if device.type == "cuda" and rng["cuda"] is not None:
        torch.cuda.set_rng_state(rng["cuda"], device=device)
    if world > 1:
        dist.barrier()
    return int(shared["successful_updates"]), int(shared["attempted_updates"])


def compare_nested(left, right, path="root", atol=1e-6, rtol=1e-5):
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"{path}: tensor metadata differs")
        if left.is_floating_point():
            if not torch.allclose(left, right, atol=atol, rtol=rtol):
                raise AssertionError(f"{path}: max_abs={float((left.double()-right.double()).abs().max())}")
            return float((left.double()-right.double()).abs().max()) if left.numel() else 0.0
        if not torch.equal(left, right):
            raise AssertionError(f"{path}: integer/RNG tensor differs")
        return 0.0
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: keys differ")
        return max((compare_nested(left[k], right[k], f"{path}.{k}", atol, rtol) for k in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: length differs")
        return max((compare_nested(x, y, f"{path}[{i}]", atol, rtol)
                    for i, (x, y) in enumerate(zip(left, right))), default=0.0)
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")
    return 0.0


def compare_checkpoints(a):
    if not a.checkpoint_a or not a.checkpoint_b:
        raise ValueError("--mode compare requires --checkpoint-a and --checkpoint-b")
    path_a, manifest_a = validate_completed(a.checkpoint_a)
    path_b, manifest_b = validate_completed(a.checkpoint_b)
    state_a = torch.load(path_a / "shared.pt", map_location="cpu")
    state_b = torch.load(path_b / "shared.pt", map_location="cpu")
    current_torch = str(torch.__version__)
    if state_a.get("torch_version") != current_torch or state_b.get("torch_version") != current_torch:
        raise RuntimeError("comparison checkpoint torch version differs from the current interpreter")
    if state_a["config"] != state_b["config"]:
        raise AssertionError("trajectory configs differ")
    if manifest_a["files"] != manifest_b["files"]:
        raise AssertionError("checkpoint file sets differ")
    max_abs = compare_nested(
        {k: state_a[k] for k in ("model", "optimizer", "scaler", "step", "attempted_updates", "successful_updates")},
        {k: state_b[k] for k in ("model", "optimizer", "scaler", "step", "attempted_updates", "successful_updates")},
        atol=a.compare_atol, rtol=a.compare_rtol,
    )
    for name in sorted(x for x in manifest_a["files"] if x.startswith("rng_rank")):
        if name not in manifest_b["files"]:
            raise AssertionError(f"missing {name} in second checkpoint")
        compare_nested(torch.load(path_a/name, map_location="cpu"), torch.load(path_b/name, map_location="cpu"),
                       path=name, atol=a.compare_atol, rtol=a.compare_rtol)
    cfg = state_a["config"]
    ids = torch.arange(10_000_000, 10_000_000 + cfg["micro_batch"])
    train_id_max = state_a["step"] * cfg["accum"] * cfg["world_size"] * cfg["micro_batch"] - 1
    if int(ids[0]) <= train_id_max:
        raise ValueError("fixed comparison IDs overlap the training trajectory")
    x, y = make_batch(ids, cfg["seq"], cfg["vocab"], torch.device("cpu"))
    losses = []
    for state in (state_a, state_b):
        model = TinyLM(cfg["vocab"], cfg["dim"], cfg["heads"], cfg["layers"])
        model.load_state_dict(state["model"], strict=True); model.eval()
        with torch.no_grad():
            logits = model(x)
            losses.append(float(nn.functional.cross_entropy(logits.reshape(-1,cfg["vocab"]), y.reshape(-1))))
    if not math.isclose(losses[0], losses[1], abs_tol=a.compare_atol, rel_tol=a.compare_rtol):
        raise AssertionError(f"next fixed-batch losses differ: {losses}")
    print(json.dumps({"status":"RESUME_EQUIVALENT","step":state_a["step"],"max_state_abs":max_abs,
                      "next_loss_a":losses[0],"next_loss_b":losses[1]}))


def write_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def unit_test(a):
    ids = sample_ids(0, 0, 0, 1, 2, 1)
    x, y = make_batch(ids, 7, 101, torch.device("cpu"))
    model = TinyLM(101, 32, 4, 2)
    z = model(x)
    loss = nn.functional.cross_entropy(z.reshape(-1, 101), y.reshape(-1))
    loss.backward()
    assert x.shape == (2, 7) and z.shape == (2, 7, 101)
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(json.dumps({"status": "UNIT_OK", "shape": list(z.shape), "loss_finite": True}))


def collective_test(rank, world, device):
    value = torch.tensor(float(rank + 1), device=device)
    if world > 1:
        dist.all_reduce(value)
    expected = world * (world + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all_reduce={value.item()} expected={expected}")
    print(json.dumps({"rank": rank, "world": world, "collective": "OK", "sum": value.item()}))


def train(a, rank, world, device):
    conflict = torch.tensor(int(Path(a.output).exists() or (rank == 0 and Path(a.log).exists())), device=device)
    if world > 1: dist.all_reduce(conflict, op=dist.ReduceOp.MAX)
    if not a.resume and int(conflict):
        raise FileExistsError("non-resume run refuses an existing checkpoint directory or log")
    seed_all(a.seed, 0)
    model = TinyLM(a.vocab, a.dim, a.heads, a.layers).to(device)
    if world > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr)
    amp_enabled = a.amp == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if a.resume:
        start, attempted = load_checkpoint(a, model, optim, scaler, rank, world, device)
    else:
        start, attempted = 0, 0
    if a.steps <= start:
        raise ValueError(f"training target --steps={a.steps} must exceed resume step={start}")
    criterion = nn.CrossEntropyLoss()
    profiler = None
    if a.profile_steps > 0:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=max(1, a.profile_steps - 2), repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(Path(a.trace_dir) / f"rank{rank}")),
            record_shapes=True, profile_memory=True, with_stack=False,
        )
        profiler.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    last = begin
    successful = start
    last_logged_step = start
    attempt_limit = attempted + max(16, 4 * (a.steps - start))
    benchmark_start = None
    benchmark_start_step = max(start, a.benchmark_warmup)
    benchmark_elapsed = None
    while successful < a.steps:
        step = successful
        if benchmark_start is None and step == benchmark_start_step and a.profile_steps == 0:
            if world > 1: dist.barrier()
            if device.type == "cuda": torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
        attempted += 1
        optim.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        first_ids = None
        for micro in range(a.accum):
            ids = sample_ids(step, micro, rank, world, a.micro_batch, a.accum, a.inject_duplicate_data)
            if step < start + 2:
                gathered = audit_ids(ids, world)
                if first_ids is None:
                    first_ids = gathered
            x, y = make_batch(ids, a.seq, a.vocab, device)
            sync_context = model.no_sync() if isinstance(model, DDP) and micro < a.accum - 1 else contextlib.nullcontext()
            with sync_context:
                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
                    logits = model(x)
                    raw_loss = criterion(logits.reshape(-1, a.vocab), y.reshape(-1))
                    if a.inject_nan_rank == rank and step == start and micro == 0:
                        raw_loss = raw_loss * torch.tensor(float("nan"), device=device)
                    loss = raw_loss / a.accum
                loss_ok = torch.tensor(int(torch.isfinite(raw_loss).item()), device=device)
                if world > 1:
                    dist.all_reduce(loss_ok, op=dist.ReduceOp.MIN)
                if not int(loss_ok):
                    raise FloatingPointError(f"non-finite loss on at least one rank at step={step} micro={micro}")
                scaler.scale(loss).backward()
                loss_sum += raw_loss.detach()
        scaler.unscale_(optim)
        local_grad_ok = all(
            p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.parameters()
        )
        grad_ok = torch.tensor(int(local_grad_ok), device=device)
        if world > 1:
            dist.all_reduce(grad_ok, op=dist.ReduceOp.MIN)
        if not int(grad_ok):
            if not amp_enabled:
                raise FloatingPointError(f"non-finite gradient on at least one rank at step={step}")
            new_scale = max(float(scaler.get_scale()) / 2.0, 1.0)
            scaler.update(new_scale=new_scale)
            row={"attempted":attempted,"successful_step":successful,"step_skipped":True,
                 "skip_reason":"cross_rank_non_finite_gradient","loss_scale":new_scale,
                 "first_ids":first_ids}
            if rank == 0:
                print(json.dumps(row,sort_keys=True),flush=True); write_jsonl(a.log,row)
            if attempted >= attempt_limit:
                raise FloatingPointError("too many AMP overflows; stop and debug FP32/fixed batch")
            continue
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        norm_ok = torch.tensor(int(math.isfinite(grad_norm)), device=device)
        if world > 1:
            dist.all_reduce(norm_ok, op=dist.ReduceOp.MIN)
        if not int(norm_ok):
            raise FloatingPointError(f"non-finite global grad norm before optimizer step at step={step}")
        scale_before = float(scaler.get_scale())
        scaler.step(optim); scaler.update()
        scale_after = float(scaler.get_scale())
        skipped = scale_after < scale_before
        skip_min = skip_max = torch.tensor(int(skipped), device=device)
        if world > 1:
            skip_min, skip_max = skip_min.clone(), skip_max.clone()
            dist.all_reduce(skip_min, op=dist.ReduceOp.MIN); dist.all_reduce(skip_max, op=dist.ReduceOp.MAX)
        if int(skip_min) != int(skip_max):
            raise RuntimeError("AMP skip decision differs across ranks")
        if profiler is not None:
            profiler.step()
        if skipped:
            row={"attempted":attempted,"successful_step":successful,"step_skipped":True,
                 "loss_scale":scale_after,"grad_norm":grad_norm,"first_ids":first_ids}
            if rank == 0:
                print(json.dumps(row,sort_keys=True),flush=True); write_jsonl(a.log,row)
            if attempted >= attempt_limit:
                raise FloatingPointError("too many AMP overflows; stop and debug FP32/fixed batch")
            continue
        if not math.isfinite(grad_norm):
            raise FloatingPointError("optimizer step succeeded with non-finite grad norm")
        successful += 1
        if benchmark_start is not None and successful == a.steps:
            if device.type == "cuda": torch.cuda.synchronize(device)
            if world > 1: dist.barrier()
            benchmark_elapsed = time.perf_counter() - benchmark_start
        report_loss = loss_sum / a.accum
        if world > 1:
            dist.all_reduce(report_loss)
            report_loss /= world
        completed = successful
        if completed % a.log_every == 0 or completed == a.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            interval = completed - last_logged_step
            tokens = interval * world * a.micro_batch * a.accum * a.seq
            chk, spread = checksum(model, device, world)
            row = {
                "step": completed, "attempted": attempted, "world_size": world, "loss": float(report_loss.item()),
                "tokens_per_s": tokens / max(now - last, 1e-9), "parameter_checksum": chk,
                "checksum_spread": spread, "first_ids": first_ids,
                "loss_scale": scale_after, "step_skipped": False, "grad_norm": grad_norm,
                "max_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            }
            if rank == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
                write_jsonl(a.log, row)
            if spread > 1e-7:
                raise RuntimeError(f"parameter divergence spread={spread}")
            last = now; last_logged_step = completed
        if completed % a.save_every == 0 or completed == a.steps:
            save_checkpoint(a, model, optim, scaler, completed, attempted, successful, rank, world)
    if profiler is not None:
        profiler.stop()
    if rank == 0:
        total_tokens = (a.steps - start) * world * a.micro_batch * a.accum * a.seq
        benchmark_steps = max(0, a.steps - benchmark_start_step)
        benchmark_tokens = benchmark_steps * world * a.micro_batch * a.accum * a.seq
        print(json.dumps({"status": "TRAIN_OK", "steps": a.steps, "attempted":attempted,
            "successful_updates":successful, "world": world,
            "run_tokens_per_s_including_aux": total_tokens / max(time.perf_counter() - begin, 1e-9),
            "steady_start_step":benchmark_start_step,"steady_steps":benchmark_steps,
            "steady_seconds":benchmark_elapsed,
            "steady_tokens_per_s":benchmark_tokens / benchmark_elapsed if benchmark_elapsed else None}))


@torch.no_grad()
def evaluate(a, device):
    target, manifest = validate_completed(a.output)
    shared = torch.load(target / "shared.pt", map_location=device)
    if shared.get("torch_version") != torch.__version__ or manifest.get("torch_version") != torch.__version__:
        raise RuntimeError("evaluation torch version differs from checkpoint")
    cfg = shared["config"]
    model = TinyLM(cfg["vocab"], cfg["dim"], cfg["heads"], cfg["layers"]).to(device)
    model.load_state_dict(shared["model"])
    model.eval()
    if a.eval_batches < 1:
        raise ValueError("--eval-batches must be >= 1")
    eval_id_min = 10_000_000
    train_id_max = shared["step"] * cfg["accum"] * cfg["world_size"] * cfg["micro_batch"] - 1
    if eval_id_min <= train_id_max:
        raise ValueError(f"eval IDs overlap training IDs: eval_start={eval_id_min}, train_end={train_id_max}")
    total = 0.0
    count = 0
    for i in range(a.eval_batches):
        ids = torch.arange(10_000_000 + i * a.micro_batch, 10_000_000 + (i + 1) * a.micro_batch)
        x, y = make_batch(ids, cfg["seq"], cfg["vocab"], device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, cfg["vocab"]), y.reshape(-1))
        total += float(loss.item())
        count += 1
    row = {"status": "EVAL_OK", "checkpoint":str(target), "checkpoint_step": shared["step"],
           "eval_loss": total / count, "eval_batches": count, "train_id_max":train_id_max,
           "eval_id_min":eval_id_min, "eval_id_max":eval_id_min + count * a.micro_batch - 1,
           "disjoint_from_train":True}
    print(json.dumps(row, sort_keys=True))
    write_jsonl(a.log, row)


def main():
    a = parse_args()
    if a.dim % a.heads:
        raise ValueError("dim must be divisible by heads")
    if a.mode == "unit":
        unit_test(a)
        return
    if a.mode == "compare":
        compare_checkpoints(a)
        return
    rank, local_rank, world, device = distributed_context()
    try:
        if a.mode == "collective":
            collective_test(rank, world, device)
        elif a.mode == "train":
            train(a, rank, world, device)
        elif a.mode == "eval":
            if world != 1:
                raise RuntimeError("independent eval must run as one process")
            evaluate(a, device)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
chmod +x src/ddp_lab.py
```

### 执行命令

```bash
python -m py_compile src/ddp_lab.py
python src/ddp_lab.py --mode unit --amp none
sha256sum src/ddp_lab.py | tee evidence/source.sha256
```

### 预期观测（估算/示例，不是实测）

最后一行形如 `{"status":"UNIT_OK","shape":[2,7,101],"loss_finite":true}`；具体 loss 不设硬编码值。

### 验收条件

静态检查与 CPU 单测退出码均为 0；输出 shape `[2,7,101]`，loss 和梯度有限；源码哈希已保存。

### 若失败，先看什么，再改什么

先看 Python/PyTorch 导入路径与 traceback；再核对 `dim % heads == 0`。不得先改分布式参数，因为此日尚未进入 DDP。

### 当日证据清单

`evidence/source.sha256`、环境清单、两条命令及退出码；公司环境的 `/tmp/week07-pip-freeze.txt` 仍留公司内部。

## Day 2：Gloo/NCCL collective 与单卡基线

### 为什么做

先证明 launcher/rank 映射与 collective 可工作，再建立没有 DDP 通信的性能/数值基线。

### 输入与前置检查

Day 1 通过；公司先确认 `CUDA_VISIBLE_DEVICES`、`nvidia-smi topo -m` 与调度器实际分配一致。

### 本日要创建/修改的文件

不改源码；创建 `logs/single.jsonl` 和单卡 versioned checkpoint。

### 实现

collective 模式会校验 rank `1..N` 的和；train 模式按全局 sample ID 生成数据并写 JSONL。

### 执行命令

```bash
torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode collective
CUDA_VISIBLE_DEVICES=0 python src/ddp_lab.py --mode train --steps 20 --save-every 10 --benchmark-warmup 5 \
  --micro-batch 2 --accum 2 --seq 64 --vocab 2048 --dim 256 --heads 8 --layers 4 \
  --amp fp16 --output artifacts/checkpoints/single --log logs/single.jsonl
```

无两张 GPU 的个人路径只运行单卡；CPU collective 可用以下替代检查 launcher，但不能作为 NCCL 证据：

```bash
CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode collective
```

### 预期观测（估算/示例，不是实测）

两 rank 都打印 `collective:"OK", sum:3.0`。单卡每 5 step 一行，`checksum_spread=0`、loss 有限、`TRAIN_OK`；吞吐与显存只记录不预设。

### 验收条件

collective 60 秒内完成；单卡 20 step 完成，并在 `ckpt_step000010/`、`ckpt_step000020/` 内存在经 hash 绑定的 `shared.pt`、`rng_rank0.pt`、`manifest.json`、`COMPLETED`。

### 若失败，先看什么，再改什么

hang 先看 rank 数、端口占用和可见卡，再以 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 重跑 collective；NCCL 失败才检查驱动/NCCL/拓扑。单卡 OOM 则降 `micro-batch`，并在配置名记录变化。

### 当日证据清单

collective 全 rank 输出、`logs/single.jsonl`、checkpoint manifest、拓扑文本（仅公司内部）。

## Day 3：2 卡 DDP smoke、保存与严格恢复

### 为什么做

确认 `no_sync` 只跳过非末次 micro-step、全局样本无重复、参数保持一致，并验证恢复从已完成 step 接续。

### 输入与前置检查

Day 2 通过；两卡型号一致且空闲。删除旧实验应另建新 run 名，本文不要求破坏性删除。

### 本日要创建/修改的文件

创建 control 与 resume 两个独立根目录/日志；resume 根目录内按 step 发布新版本，旧版本不可覆盖。

### 实现

脚本首两个 step `all_gather_object` 审计 sample ID；loss 在 backward 前、梯度与 grad norm 在 optimizer step 前分别做跨 rank finite gate。AMP 梯度溢出时所有 rank 一起跳过、attempted 加一而 successful 不变；保存双时钟、共享状态和每 rank RNG 到临时目录，核完整文件集/manifest/hash 后发布 version 目录并最后写含格式版本、torch patch、manifest hash 的 `COMPLETED`。另跑 uninterrupted control 比较完整状态和 next loss。

### 执行命令

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode train \
  --steps 20 --save-every 20 --log-every 1 --micro-batch 2 --accum 2 --seq 64 --vocab 2048 \
  --dim 256 --heads 8 --layers 4 --amp fp16 \
  --output artifacts/checkpoints/ddp2_control --log logs/ddp2_control.jsonl

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode train \
  --steps 10 --save-every 10 --log-every 1 --micro-batch 2 --accum 2 --seq 64 --vocab 2048 \
  --dim 256 --heads 8 --layers 4 --amp fp16 \
  --output artifacts/checkpoints/ddp2_resume --log logs/ddp2_resume.jsonl

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode train \
  --steps 20 --save-every 10 --log-every 1 --micro-batch 2 --accum 2 --seq 64 --vocab 2048 \
  --dim 256 --heads 8 --layers 4 --amp fp16 --resume \
  --output artifacts/checkpoints/ddp2_resume --log logs/ddp2_resume.jsonl

python src/ddp_lab.py --mode compare --checkpoint-a artifacts/checkpoints/ddp2_control \
  --checkpoint-b artifacts/checkpoints/ddp2_resume | tee evidence/ddp2-resume-equivalence.json
```

### 预期观测（估算/示例，不是实测）

control 到 step20；resume 先到10再接续到20，第二段 attempted 从10继续。`--log-every 1` 保留首批 ID 证据；各 rank 不重叠，`checksum_spread` 接近0；compare 输出 `RESUME_EQUIVALENT`。

### 验收条件

三次训练与 compare 均退出0；恢复不重跑 step0；attempted/successful 连续；两个根目录的 step20 manifest/hash/marker 全匹配；control 与 resume 的 model/optimizer/scaler/RNG/step 和 next loss 在容差内一致。

### 若失败，先看什么，再改什么

先检查 `COMPLETED` 与每个 `rng_rank*.pt`；world size 变化会被拒绝，这是设计行为。参数极差异常先禁用 AMP 对照，再检查是否有 rank 跳过 optimizer step，禁止提高容差掩盖问题。

### 当日证据清单

三次完整训练命令、两份日志、各 version manifest/完成标志及 `evidence/ddp2-resume-equivalence.json`。

## Day 4：正式强/弱扩展与拓扑对照

### 为什么做

在固定全局工作量下测强扩展，在固定每卡工作量下测弱扩展，并区分卡数效应与 GPU 对/NUMA/PCIe 拓扑效应。

### 输入与前置检查

Day 3 通过；确认正式模型单卡 5-step probe 峰值显存低于预算。以下每个 run 都用新 checkpoint/log 路径。

### 本日要创建/修改的文件

只产生按实验命名的日志与 checkpoint；不改源码。

### 实现

正式配置约 3.3 亿参数：`vocab=16384, dim=1024, heads=16, layers=24, seq=256`。强扩展保持全局 batch 8：1 卡 `micro=1,accum=8`，2 卡 `1,4`，4 卡 `1,2`，8 卡 `1,1`。弱扩展保持每卡 batch 1 且 `accum=1`。

### 执行命令

先做单卡 5-step 资源探针：

```bash
CUDA_VISIBLE_DEVICES=0 python src/ddp_lab.py --mode train --steps 5 --save-every 5 \
  --micro-batch 1 --accum 8 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 24 \
  --amp fp16 --output artifacts/checkpoints/probe --log logs/probe.jsonl
```

探针通过后，分别执行强扩展（每条都是独立 run）：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 src/ddp_lab.py --mode train --steps 100 --benchmark-warmup 20 --log-every 100 --micro-batch 1 --accum 8 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 24 --amp fp16 --save-every 100 --output artifacts/checkpoints/strong1 --log logs/strong1.jsonl
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode train --steps 100 --benchmark-warmup 20 --log-every 100 --micro-batch 1 --accum 4 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 24 --amp fp16 --save-every 100 --output artifacts/checkpoints/strong2 --log logs/strong2.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 src/ddp_lab.py --mode train --steps 100 --benchmark-warmup 20 --log-every 100 --micro-batch 1 --accum 2 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 24 --amp fp16 --save-every 100 --output artifacts/checkpoints/strong4 --log logs/strong4.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 src/ddp_lab.py --mode train --steps 100 --benchmark-warmup 20 --log-every 100 --micro-batch 1 --accum 1 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 24 --amp fp16 --save-every 100 --output artifacts/checkpoints/strong8 --log logs/strong8.jsonl
```

弱扩展把每卡 batch 固定为 1：

```bash
for n in 1 2 4 8; do
  ids=$(seq -s, 0 $((n-1)))
  CUDA_VISIBLE_DEVICES="$ids" torchrun --standalone --nproc-per-node="$n" src/ddp_lab.py --mode train \
    --steps 100 --benchmark-warmup 20 --log-every 100 --micro-batch 1 --accum 1 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 24 \
    --amp fp16 --save-every 100 --output "artifacts/checkpoints/weak${n}" --log "logs/weak${n}.jsonl"
done
```

拓扑对照从 `nvidia-smi topo -m` 选一对近邻卡和一对远端卡，先把真实编号写进命令；下面的 `0,1`/`0,4` 仅是待核验占位，未核验不得执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode train --steps 50 --benchmark-warmup 10 --log-every 50 --micro-batch 1 --accum 4 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 24 --amp fp16 --save-every 50 --output artifacts/checkpoints/topo_near --log logs/topo_near.jsonl
CUDA_VISIBLE_DEVICES=0,4 torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode train --steps 50 --benchmark-warmup 10 --log-every 50 --micro-batch 1 --accum 4 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 24 --amp fp16 --save-every 50 --output artifacts/checkpoints/topo_far --log logs/topo_far.jsonl
```

### 预期观测（估算/示例，不是实测）

吞吐一般随卡数增长但不会线性；强扩展通信占比会增大。弱扩展每卡吞吐若明显下降，优先怀疑通信/拓扑。绝不预填具体加速比。

### 验收条件

每个成功 run 配置、100 step、`steady_tokens_per_s`、峰值显存、loss 与 checkpoint 都齐全；正式窗口无 profiler/中途日志/中途保存，强扩展比较的全局 batch/序列/模型/step/steady steps 相同；拓扑编号有原始证据。

### 若失败，先看什么，再改什么

OOM 先减 `seq` 或层数并给所有卡数统一新配置；hang 用 Day 2 collective 和 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 缩小；性能异常先排除不同功耗/占用/拓扑，最后才调 bucket。变更必须产生新 run 名。

### 当日证据清单

8 个扩展日志、各 manifest、GPU/驱动/拓扑清单、配置差异表和所有退出码；均留公司内部。

## Day 5：短 profiler 与 NCCL 诊断

### 为什么做

解释扩展曲线中的计算、通信和空洞，而不是仅凭吞吐猜原因。

### 输入与前置检查

Day 4 至少有单卡和 2 卡结果；trace 可能包含算子 shape/路径，按公司敏感产物管理。

### 本日要创建/修改的文件

创建 `traces/rank*/` 与 `logs/profile2.jsonl`。

### 实现

脚本使用 PyTorch profiler 的 1 wait + 1 warmup + active 窗口，每 rank 分目录，避免覆盖。

### 执行命令

```bash
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode train \
  --steps 8 --profile-steps 6 --log-every 2 --save-every 8 \
  --micro-batch 1 --accum 4 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 24 --amp fp16 \
  --output artifacts/checkpoints/profile2 --log logs/profile2.jsonl --trace-dir traces

find traces -type f -maxdepth 3 -printf '%p %s bytes\n' | sort
```

### 预期观测（估算/示例，不是实测）

每个 rank 至少一个非空 trace；时间线可见前后向计算及 NCCL all-reduce。短窗口吞吐不用于正式性能结论。

### 验收条件

两个 rank 的 trace 均非空且时间窗口相同；能从事件说明主要瓶颈候选，并与 Day 4 曲线一致或明确标记矛盾待查。

### 若失败，先看什么，再改什么

无 trace 先核对 schedule 是否走满；单 rank 有 trace 先查目录权限/进程退出；trace 过大先缩 active step，不把 trace 拷到个人电脑。

### 当日证据清单

trace 文件清单/哈希、短 run 日志、NCCL 日志与一页内部结论。原 trace 不外传。

## Day 6：独立评估、故障注入与最终验收

### 为什么做

把“训练跑完”与“checkpoint 可独立读取、数据防重机制有效”分开验收。

### 输入与前置检查

选择一个 `COMPLETED` checkpoint；评估使用从未参与训练的 sample ID `>=10,000,000`。

### 本日要创建/修改的文件

创建 `logs/eval.jsonl` 和故障注入终端记录；不修改 checkpoint。

### 实现

eval 模式从 checkpoint 保存的模型配置重建模型并严格加载；故障注入让 rank 1 故意复用 rank 0 ID，所有 rank 在 collective 后一致失败。

### 执行命令

```bash
CUDA_VISIBLE_DEVICES=0 python src/ddp_lab.py --mode eval --eval-batches 20 --micro-batch 1 \
  --output artifacts/checkpoints/strong1 --log logs/eval.jsonl

set +e
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode train \
  --steps 2 --micro-batch 2 --accum 1 --seq 32 --vocab 256 --dim 64 --heads 4 --layers 2 \
  --amp fp16 --inject-duplicate-data --output artifacts/checkpoints/injected --log logs/injected.jsonl \
  > evidence/injected.stdout 2>&1
code=$?
set -e
test "$code" -ne 0
grep -q DATA_DUPLICATION evidence/injected.stdout
printf 'EXPECTED_FAILURE exit=%s\n' "$code" | tee evidence/injected.result

set +e
timeout 60s env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/ddp_lab.py --mode train \
  --steps 2 --micro-batch 1 --accum 1 --seq 32 --vocab 256 --dim 64 --heads 4 --layers 2 \
  --amp fp16 --inject-nan-rank 1 --output artifacts/checkpoints/injected_nan --log logs/injected_nan.jsonl \
  > evidence/injected-nan.stdout 2>&1
nan_code=$?
set -e
test "$nan_code" -ne 0
test "$nan_code" -ne 124
grep -q "non-finite loss on at least one rank" evidence/injected-nan.stdout
printf 'EXPECTED_SYNCHRONIZED_FAILURE exit=%s\n' "$nan_code" | tee evidence/injected-nan.result

python - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('logs').glob('strong*.jsonl')):
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    assert rows and all(r.get('checksum_spread',0.0) <= 1e-7 for r in rows)
    print(p, 'last_step', rows[-1].get('step'))
PY
```

### 预期观测（估算/示例，不是实测）

eval 输出 `EVAL_OK`、有限 loss、正确 version checkpoint step、训练 ID 上界、评估 ID 区间及 `disjoint_from_train=true`。重复数据注入必须含 `DATA_DUPLICATION`；rank1 NaN 注入必须在60秒内由所有 rank 同步失败且不能是 timeout 124。

### 验收条件

独立评估严格加载成功；负测按预期失败；1/2/4/8 强扩展至少各有一个可审计结果，否则结论标 `INCONCLUSIVE`；不把单卡个人结果包装成公司多卡结果。

### 若失败，先看什么，再改什么

eval 先查 `COMPLETED`、保存配置和权重 key；注入 hang 先确认所有 rank 都走到 ID collective。汇总脚本失败则定位具体 JSONL，不手工删坏行。

### 当日证据清单

`logs/eval.jsonl`、`evidence/injected.*`、扩展汇总输出、最终配置/环境/退出码表，以及“已验证/待验证/环境阻塞”状态表。

## 4. 最终报告最低字段

逐 run 记录：run ID、资源路径、日期、host（内部）、GPU 型号/数量、拓扑摘要、torch/CUDA/NCCL、模型参数量、序列长度、micro/accum/global batch、precision、step 区间、正式 `steady_tokens_per_s`、含辅助操作的 run 吞吐、峰值显存、loss、checkpoint 状态、退出码。强扩展计算 `S_N=T_N/T_1`、`E_N=S_N/N`；弱扩展比较每卡吞吐。只有实际日志存在才能填“已验证”，本文出现的所有预期均不得直接抄入结果栏。

## 5. 清理与保密

不要求删除任何材料。若磁盘配额需要清理，先列出精确 run 目录、确认它位于 `week07-ddp/artifacts/checkpoints/` 或 `traces/` 且已按公司政策归档，再由获授权人员执行可恢复操作。公司日志、NCCL 信息、trace、截图、checkpoint 和拓扑表始终留公司环境。
