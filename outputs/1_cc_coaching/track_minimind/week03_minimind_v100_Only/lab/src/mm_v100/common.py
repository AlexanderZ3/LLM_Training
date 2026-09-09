"""公共设施：dtype 守卫、scaler 适配、分布式初始化、seed、数据切分、JSONL 日志。

来源
----
从 week02 lab/src/mm_dist/common.py 复制并裁剪；裁掉了 MiniMind 仓库导入
（模型改由本包 model.py 自带）、MoE 相关字段、RealJsonlDataset（数据契约
改由 data_contract.py 负责）。ScalerAdapter 与 PurePythonScaler 是本周新增。

版本约定
--------
目标机 = 公司 8xV100 + PyTorch 2.1.0。凡是 2.1 之后才有的 API，这里都只在
getattr 存在时才用，并保留 2.1 的路径。不使用 FSDP2 / DTensor / device_mesh /
torch.accelerator / torch.get_default_device。

为什么要自己写一个 scaler
-------------------------
torch 2.1 只有 torch.cuda.amp.GradScaler，在没有 CUDA 的机器上构造时会自动把
enabled 改成 False。也就是说"GradScaler 持续跳步"这档故障在 CPU 上根本无法用
官方 scaler 复现，Day 2 的三档判别就少了一档的本机验证。所以这里定义一个纯
Python 的等价实现（同样的 init_scale / backoff / growth_interval 语义），
CPU 与 V100 共用 ScalerAdapter 接口，last_step_skipped 这个不变量在两台机器上
含义完全一致。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

TARGET_TORCH_VERSION = "2.1.0"
MINIMIND_COMMIT = "7a6fddd63a30c06b2fdd5fac4089922b29bc841b"

# GradScaler 的 torch 默认值。写死在这里是为了让 CPU 的纯 Python 实现与 V100 上
# 的官方实现在同一组参数下比较；Day 0 探针会把官方实际默认值读出来核对这四个数。
DEFAULT_INIT_SCALE = 65536.0
DEFAULT_GROWTH_FACTOR = 2.0
DEFAULT_BACKOFF_FACTOR = 0.5
DEFAULT_GROWTH_INTERVAL = 2000

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

BF16_REFUSED = (
    "拒绝 bfloat16。本 lab 的目标机是公司 8xV100（compute capability 7.0）：\n"
    "  torch.cuda.is_bf16_supported() 要求 major>=8，V100 不满足；\n"
    "  autocast(dtype=bfloat16) 会抛 "
    "'Current CUDA Device does not support bfloat16'。\n"
    "下一步：把命令里的 --dtype 改成 float16（会自动配 GradScaler），"
    "或者 --dtype float32 做数值参考。"
)


# ---------------------------------------------------------------------------
# dtype 守卫
# ---------------------------------------------------------------------------


def device_capability(device: str) -> Optional[Tuple[int, int]]:
    """返回 (major, minor)；CPU 或无 CUDA 时返回 None。"""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    idx = 0
    if ":" in str(device):
        idx = int(str(device).split(":")[1])
    return torch.cuda.get_device_capability(idx)


def assert_dtype_supported(dtype: str, device: str = "cpu") -> torch.dtype:
    """检查 dtype 在当前设备上可用，返回 torch.dtype；bfloat16 一律拒绝。

    守卫的价值在报错的位置：放任 bfloat16 传到 autocast，错误会在训练第一步的
    CUDA 里出现，那时 8 个进程已经建好、数据已经加载、显存已经占上。
    在参数解析阶段就拒绝，公司机器 10 分钟的运行预算不会被浪费。
    """
    key = str(dtype).lower()
    if key not in _DTYPES:
        raise ValueError(
            "不认识的 dtype=" + repr(dtype) + "；可选：float32 / float16 / bfloat16"
        )
    td = _DTYPES[key]
    if td is torch.bfloat16:
        cap = device_capability(device)
        where = ("当前设备不是 CUDA（device=" + str(device) + "）"
                 if cap is None
                 else "设备 " + str(device) + " 的 compute capability 是 "
                      + str(cap[0]) + "." + str(cap[1]))
        if cap is None or cap[0] < 8:
            raise RuntimeError(where + "。\n" + BF16_REFUSED)
    return td


def autocast_context(dtype: torch.dtype, device: str, enabled: bool = True):
    """torch 2.1 语法的 autocast 上下文。

    CPU 分支存在的唯一理由是本机能跑 fp16 数值对照；V100 上永远走 cuda 分支。
    """
    if dtype is torch.float32 or not enabled:
        return nullcontext()
    if str(device).startswith("cuda"):
        return torch.cuda.amp.autocast(dtype=dtype)
    return torch.autocast(device_type="cpu", dtype=dtype)


# ---------------------------------------------------------------------------
# GradScaler
# ---------------------------------------------------------------------------


class PurePythonScaler:
    """与 torch GradScaler 语义一致的纯 Python 实现（CPU 上唯一可用的那个）。

    只实现训练循环真正用到的部分。inf 检查用 torch.isfinite，没有 fused kernel，
    所以慢——但 CPU 上本来就只跑几步。
    """

    def __init__(self, init_scale: float = DEFAULT_INIT_SCALE,
                 growth_factor: float = DEFAULT_GROWTH_FACTOR,
                 backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
                 growth_interval: int = DEFAULT_GROWTH_INTERVAL,
                 enabled: bool = True) -> None:
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._growth_tracker = 0
        self._enabled = bool(enabled)
        self._found_inf = False
        self._unscaled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def get_scale(self) -> float:
        return self._scale if self._enabled else 1.0

    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return outputs
        return outputs * self._scale

    def unscale_(self, optimizer) -> None:
        if not self._enabled or self._unscaled:
            return
        inv = 1.0 / self._scale
        found = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.mul_(inv)
                if not bool(torch.isfinite(p.grad).all().item()):
                    found = True
        self._found_inf = found
        self._unscaled = True

    def step(self, optimizer) -> bool:
        """返回 optimizer 是否真的走了这一步（False = 本步被跳过）。"""
        if not self._enabled:
            optimizer.step()
            return True
        if not self._unscaled:
            self.unscale_(optimizer)
        if self._found_inf:
            return False
        optimizer.step()
        return True

    def update(self, new_scale: Optional[float] = None) -> None:
        if not self._enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale)
        elif self._found_inf:
            self._scale = max(self._scale * self._backoff_factor, 1e-4)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
        self._found_inf = False
        self._unscaled = False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "_growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._scale = float(state["scale"])
        self._growth_factor = float(state.get("growth_factor",
                                              DEFAULT_GROWTH_FACTOR))
        self._backoff_factor = float(state.get("backoff_factor",
                                               DEFAULT_BACKOFF_FACTOR))
        self._growth_interval = int(state.get("growth_interval",
                                              DEFAULT_GROWTH_INTERVAL))
        self._growth_tracker = int(state.get("_growth_tracker", 0))


def new_torch_scaler(device_type: str, enabled: bool):
    """按 torch 版本选择官方 GradScaler 的构造方式；构造不出来返回 None。

    torch >= 2.3：torch.amp.GradScaler(device_type, ...)；
    torch 2.1（目标机）：只有 torch.cuda.amp.GradScaler，且只服务 CUDA。
    """
    amp = getattr(torch, "amp", None)
    cls = getattr(amp, "GradScaler", None) if amp is not None else None
    if cls is not None:
        try:
            return cls(device_type, enabled=enabled)
        except (TypeError, RuntimeError, ValueError):
            pass
    if device_type == "cuda":
        return torch.cuda.amp.GradScaler(enabled=enabled)
    return None


class ScalerAdapter:
    """统一 CPU / CUDA 两条路径的 scaler 接口，并把"这一步跳了吗"变成显式字段。

    官方 GradScaler 不直接告诉你某一步有没有被跳过，只能靠"scale 变小了"间接
    推断。Day 2 的故障 B 需要一个确定的布尔量，所以这里在 step() 前后自己记一份。
    """

    def __init__(self, dtype: torch.dtype, device: str,
                 force_pure_python: bool = False,
                 init_scale: float = DEFAULT_INIT_SCALE) -> None:
        self.dtype = dtype
        self.device = device
        self.enabled = dtype is torch.float16
        self.last_step_skipped = False
        self.skipped_steps = 0
        self.applied_steps = 0
        is_cuda = str(device).startswith("cuda")
        impl = None
        # CPU 上一律用纯 Python 实现，理由有两条，缺一不可：
        # 1) 目标机的 torch 2.1 根本没有 CPU GradScaler（torch.amp.GradScaler 是
        #    2.3 才有的），所以 V100 上的 CPU 路径只能用我们自己的；
        # 2) 本机的新版 torch 有 CPU GradScaler，如果这里用了它，同一份代码在两台
        #    机器上的跳步语义就不一样，Day 2 故障 B 的本机验证也就没有迁移价值。
        if self.enabled and is_cuda and not force_pure_python:
            impl = new_torch_scaler("cuda", True)
            if impl is not None and not impl.is_enabled():
                # torch 2.1 在没有可用 CUDA 时会把 enabled 悄悄改成 False。
                impl = None
        if impl is None:
            impl = PurePythonScaler(init_scale=init_scale, enabled=self.enabled)
            self.backend = "pure_python"
        else:
            self.backend = "torch"
        self.impl = impl

    def is_enabled(self) -> bool:
        return bool(self.impl.is_enabled())

    def get_scale(self) -> float:
        return float(self.impl.get_scale()) if self.is_enabled() else 1.0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return self.impl.scale(loss)

    def unscale_(self, optimizer) -> None:
        if self.is_enabled():
            self.impl.unscale_(optimizer)

    def step_and_update(self, optimizer) -> bool:
        """一次完整的 step+update，返回 optimizer 是否真的更新了参数。"""
        if not self.is_enabled():
            optimizer.step()
            self.last_step_skipped = False
            self.applied_steps += 1
            return True
        before = float(self.impl.get_scale())
        result = self.impl.step(optimizer)
        self.impl.update()
        after = float(self.impl.get_scale())
        # torch 的 GradScaler.step 返回 optimizer.step() 的返回值（通常 None），
        # 判不了跳步；scale 变小是官方唯一可见的信号，两条路径都认它。
        skipped = (result is False) or (after < before)
        self.last_step_skipped = bool(skipped)
        if skipped:
            self.skipped_steps += 1
        else:
            self.applied_steps += 1
        return not skipped

    def state_dict(self) -> Dict[str, Any]:
        return {
            "impl": self.impl.state_dict(),
            "backend": self.backend,
            "skipped_steps": self.skipped_steps,
            "applied_steps": self.applied_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.impl.load_state_dict(state["impl"])
        self.skipped_steps = int(state.get("skipped_steps", 0))
        self.applied_steps = int(state.get("applied_steps", 0))

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "enabled": self.is_enabled(),
            "scale": self.get_scale(),
            "skipped_steps": self.skipped_steps,
            "applied_steps": self.applied_steps,
        }


# ---------------------------------------------------------------------------
# 分布式
# ---------------------------------------------------------------------------


@dataclass
class DistInfo:
    """一次运行的分布式上下文。没有 RANK 环境变量时 initialized=False。"""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "none"
    initialized: bool = False
    device: str = "cpu"

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "backend": self.backend,
            "initialized": self.initialized,
            "device": self.device,
        }


def pick_backend(explicit: Optional[str] = None) -> str:
    if explicit and explicit != "auto":
        return explicit
    return "nccl" if torch.cuda.is_available() else "gloo"


def init_dist(backend: str = "auto", timeout_s: int = 600,
              force_cpu: bool = False) -> DistInfo:
    """读取 RANK / LOCAL_RANK / WORLD_SIZE 初始化进程组。"""
    rank_env = int(os.environ.get("RANK", -1))
    if rank_env == -1:
        device = "cpu"
        if torch.cuda.is_available() and not force_cpu:
            device = "cuda:0"
        return DistInfo(0, 0, 1, "none", False, device)

    be = pick_backend(backend)
    if be == "nccl" and (force_cpu or not torch.cuda.is_available()):
        be = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=be, timeout=timedelta(seconds=timeout_s))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = "cpu"
    if be == "nccl" and torch.cuda.is_available() and not force_cpu:
        torch.cuda.set_device(local_rank)
        device = "cuda:" + str(local_rank)
    return DistInfo(rank, local_rank, world_size, be, True, device)


def cleanup_dist(info: Optional[DistInfo] = None) -> None:
    """barrier + destroy_process_group；未初始化时是 no-op。"""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()


def is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def log_main(msg: str) -> None:
    if is_main_process():
        print(msg, flush=True)


def free_port() -> int:
    """让内核挑一个空闲端口，避免多个本地实验用同一个 MASTER_PORT 撞车。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def set_worker_env(rank: int, world_size: int, port: int) -> None:
    """spawn 出来的子进程在 init_process_group 之前必须设的几个变量。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(int(port))
    os.environ["RANK"] = str(int(rank))
    os.environ["LOCAL_RANK"] = str(int(rank))
    os.environ["WORLD_SIZE"] = str(int(world_size))
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


def set_seed(seed: int, per_rank: bool = False) -> int:
    """设置 random / numpy / torch 的种子，返回实际使用的种子。

    per_rank=True 复刻 MiniMind 的 setup_seed(42+rank)；做 fixed-global-batch
    等价实验时必须 per_rank=False，否则两个 world_size 的初始化不同，
    loss 天然对不上，等价结论就成了噪声。
    """
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    actual = int(seed) + rank if per_rank else int(seed)
    random.seed(actual)
    np.random.seed(actual % (2 ** 31 - 1))
    torch.manual_seed(actual)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return actual


def rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def configs_dir() -> str:
    """lab/configs 的绝对路径（从本文件位置推出来，不写字面路径）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "configs")


def resolve_config(name_or_path: str) -> str:
    """接受 tiny_cpu / tiny_cpu.json / 完整路径，返回存在的绝对路径。"""
    candidate = str(name_or_path)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    stem = candidate if candidate.endswith(".json") else candidate + ".json"
    guess = os.path.join(configs_dir(), stem)
    if os.path.isfile(guess):
        return guess
    available = sorted(n for n in os.listdir(configs_dir()) if n.endswith(".json"))
    raise FileNotFoundError(
        "找不到配置 " + repr(name_or_path) + "。\n"
        "下一步：用下面这些名字之一（不带目录、可省略 .json）：\n  "
        + "\n  ".join(available)
    )


def load_config(name_or_path: str) -> Dict[str, Any]:
    path = resolve_config(name_or_path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    for key in ("name", "model", "train"):
        if key not in cfg:
            raise ValueError("配置 " + path + " 缺少 " + repr(key) + " 段")
    cfg["_path"] = path
    return cfg


# ---------------------------------------------------------------------------
# 数据切分
# ---------------------------------------------------------------------------


class TinyDataset(Dataset):
    """合成 token 数据集：第 i 条样本只由 (seed, i) 决定。

    存在的理由：fixed-global-batch 等价实验要求"第 s 步消费的样本集合与
    world_size 无关"。用可按索引复现的合成数据，任何 world_size 都取到同一条
    样本，而且不依赖 23 GB 真语料——公司机器上的排队时间很贵。
    """

    def __init__(self, num_samples: int = 4096, seq_len: int = 64,
                 vocab_size: int = 6400, seed: int = 1234,
                 label_ignore_prefix: int = 0) -> None:
        if num_samples <= 0 or seq_len <= 1 or vocab_size <= 1:
            raise ValueError("TinyDataset 需要 num_samples>0, seq_len>1, vocab_size>1")
        self.num_samples = int(num_samples)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        # >0 时模拟 SFT：前 k 个位置当作 prompt，不进 loss
        self.label_ignore_prefix = int(label_ignore_prefix)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        idx = int(index) % self.num_samples
        gen = torch.Generator()
        gen.manual_seed((self.seed * 1000003 + idx) % (2 ** 62))
        ids = torch.randint(0, self.vocab_size, (self.seq_len,),
                            generator=gen, dtype=torch.long)
        labels = ids.clone()
        if self.label_ignore_prefix > 0:
            labels[: min(self.label_ignore_prefix, self.seq_len - 1)] = -100
        return ids, labels


def collate_indices(dataset: Dataset, indices: Sequence[int], device: str):
    """按给定的全局索引取样本并 stack 成 batch。"""
    n = len(dataset)  # type: ignore[arg-type]
    items = [dataset[int(i) % n] for i in indices]
    input_ids = torch.stack([it[0] for it in items]).to(device)
    labels = torch.stack([it[1] for it in items]).to(device)
    return input_ids, labels


def micro_batch_indices(step: int, global_batch: int, world_size: int,
                        accum: int, rank: int, acc_idx: int) -> List[int]:
    """fixed-global-batch 的确定性数据切分。

    第 step（1-based）个 optimizer step 消费全局索引区间
    [(step-1)*G, step*G)；该区间被切成 world_size*accum 个等长 chunk，
    chunk 编号 rank*accum + acc_idx。

    因此对任意 (world_size, accum) 组合，只要 G 与 micro 相同，第 step 步消费的
    样本集合完全相同——这是等价实验的数据侧前提。
    """
    denom = int(world_size) * int(accum)
    if int(global_batch) % denom != 0:
        raise ValueError(
            "global_batch=" + str(global_batch) + " 不能被 world_size*accum="
            + str(world_size) + "*" + str(accum) + "=" + str(denom) + " 整除"
        )
    micro = int(global_batch) // denom
    chunk_id = int(rank) * int(accum) + int(acc_idx)
    start = (int(step) - 1) * int(global_batch) + chunk_id * micro
    return list(range(start, start + micro))


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------


class JsonlLogger:
    """每步一行 JSON。文件都写在调用方给的目录下（公司内路径由用户指定）。"""

    def __init__(self, out_dir: str, name: str, rank: int = 0,
                 rank_in_name: bool = True, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.path: Optional[str] = None
        self._fh = None
        if not self.enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        fname = ((name + "_rank" + str(rank) + ".jsonl") if rank_in_name
                 else (name + ".jsonl"))
        self.path = os.path.join(out_dir, fname)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        if not self.enabled or self._fh is None:
            return
        rec = dict(record)
        rec.setdefault("rank", self.rank)
        self._fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# 仪表
# ---------------------------------------------------------------------------


def peak_mem_mb(device: str) -> float:
    """CUDA 峰值显存 MB；CPU 返回 0.0（本机没有可比的等价量，0 表示没测）。"""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        return torch.cuda.max_memory_allocated(idx) / (1024.0 * 1024.0)
    return 0.0


def reset_peak_mem(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        torch.cuda.reset_peak_memory_stats(idx)


def grad_global_norm(params: Iterable[torch.nn.Parameter],
                     norm_type: float = 2.0) -> float:
    """所有含梯度参数的全局范数（与 clip_grad_norm_ 的 total_norm 同定义）。"""
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    norms = torch.stack([torch.linalg.vector_norm(g.float(), norm_type)
                         for g in grads])
    return float(torch.linalg.vector_norm(norms, norm_type).item())


def param_global_norm(params: Iterable[torch.nn.Parameter]) -> float:
    values = [p.detach().float().flatten() for p in params]
    if not values:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(values), 2.0).item())


def cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    """与 MiniMind trainer_utils.get_lr 完全一致的调度。"""
    total = max(int(total_steps), 1)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * int(step) / total)))


def sdpa_context(backend: str, device: str):
    """torch 2.1 的 torch.backends.cuda.sdp_kernel 上下文。

    只在 CUDA 上生效。V100(sm70) 在 torch 2.1 下没有 flash 后端，只有
    mem_efficient / math——所以 --sdpa-backend flash 在目标机上会失败，
    这本身就是 Day 0 要拿到的一条证据。
    """
    if backend == "auto" or not str(device).startswith("cuda"):
        return nullcontext()
    flags = {
        "math": dict(enable_flash=False, enable_mem_efficient=False,
                     enable_math=True),
        "mem_efficient": dict(enable_flash=False, enable_mem_efficient=True,
                              enable_math=False),
        "flash": dict(enable_flash=True, enable_mem_efficient=False,
                      enable_math=False),
    }
    if backend not in flags:
        raise ValueError(
            "不认识的 sdpa 后端 " + repr(backend)
            + "；可选 auto/math/mem_efficient/flash"
        )
    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        print("[warn] 本机 torch " + torch.__version__
              + " 没有 torch.backends.cuda.sdp_kernel，--sdpa-backend 被忽略；"
                "目标机 torch 2.1.0 有该 API。", flush=True)
        return nullcontext()
    return torch.backends.cuda.sdp_kernel(**flags[backend])


def env_summary() -> Dict[str, Any]:
    """写进日志头部的环境摘要。不含主机名、路径、用户名，可以直接贴出来。"""
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "target_torch": TARGET_TORCH_VERSION,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """所有入口脚本共享的参数。每个都有默认值，--help 能独立看懂。"""
    parser.add_argument("--config", type=str, default="tiny_cpu",
                        help="configs/ 下的名字或完整路径")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="产物目录；默认 $MM_RUNS_ROOT/<run-tag>")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="run 目录名（day0_probe / day1_pretrain / ...）")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16"],
                        help="混合精度；bfloat16 在 V100 上会被显式拒绝")
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子")
    parser.add_argument("--seed-per-rank", type=int, default=0, choices=[0, 1],
                        help="1=复刻 MiniMind 的 42+rank；等价实验必须是 0")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="optimizer step 上限（公司机器每次运行 <=10 分钟）")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "nccl", "gloo"])
    parser.add_argument("--nccl-timeout-s", type=int, default=600,
                        help="进程组超时秒数；默认 600 而非 torch 的 1800")
    parser.add_argument("--force-cpu", action="store_true",
                        help="即使有 CUDA 也跑 CPU（本机 gloo 验证用）")
    parser.add_argument("--sdpa-backend", type=str, default="auto",
                        choices=["auto", "math", "mem_efficient", "flash"],
                        help="只在 CUDA 生效；V100 没有 flash 后端")
    parser.add_argument("--dry-run", action="store_true",
                        help="只做配置解析、模型构建与一次前向，不进训练循环")
