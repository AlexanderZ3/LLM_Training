"""公共设施：分布式初始化、dtype 守卫、seed、模型与数据构建、JSONL 日志。

版本约定
--------
目标机 = 公司 8x V100 + PyTorch 2.1.0。本文件只调用 torch 2.1 已存在的 API：
``torch.cuda.amp.autocast`` / ``GradScaler`` / ``dist.init_process_group`` /
``dist.all_to_all_single`` / ``dist.new_group``。不使用 FSDP2、DTensor、device_mesh、
``torch.accelerator``、``torch.get_default_device`` 等 2.1 之后才有的东西。

MiniMind 约定
-------------
MiniMind 是外部仓库，不复制进 lab。通过 ``MINIMIND_ROOT`` 环境变量或
``--minimind-root`` 参数引用，锁定 commit 7a6fddd63a30c06b2fdd5fac4089922b29bc841b。
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

TARGET_TORCH_VERSION = "2.1.0"
MINIMIND_COMMIT = "7a6fddd63a30c06b2fdd5fac4089922b29bc841b"

# ---------------------------------------------------------------------------
# MiniMind 定位与导入
# ---------------------------------------------------------------------------


def resolve_minimind_root(explicit: Optional[str] = None) -> str:
    """解析 MiniMind 仓库根目录。

    优先级：显式参数 > 环境变量 ``MINIMIND_ROOT``。两者都没有时抛出可执行的错误。
    """
    root = explicit or os.environ.get("MINIMIND_ROOT")
    if not root:
        raise RuntimeError(
            "未找到 MiniMind 仓库。请先执行：\n"
            "  git clone https://github.com/jingyaogong/minimind.git\n"
            "  cd minimind && git checkout " + MINIMIND_COMMIT + "\n"
            "再设置环境变量 MINIMIND_ROOT=<该目录绝对路径>，"
            "或给脚本传 --minimind-root <该目录绝对路径>。"
        )
    root = os.path.abspath(os.path.expanduser(root))
    probe = os.path.join(root, "model", "model_minimind.py")
    if not os.path.isfile(probe):
        raise RuntimeError(
            "MINIMIND_ROOT=" + root + " 下没有 model/model_minimind.py；"
            "该路径不是 MiniMind 仓库根目录。"
        )
    return root


def minimind_commit(root: str) -> str:
    """返回 MiniMind 仓库当前 commit（40 位）；取不到时返回 unknown。"""
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    head = os.path.join(root, ".git", "HEAD")
    try:
        with open(head, "r", encoding="utf-8") as fh:
            line = fh.read().strip()
        if line.startswith("ref: "):
            ref = os.path.join(root, ".git", line[5:])
            with open(ref, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        return line
    except OSError:
        return "unknown"


def import_minimind(minimind_root: Optional[str] = None) -> SimpleNamespace:
    """把 MiniMind 根目录加入 sys.path 并导入模型符号。

    注意：MiniMind 的包名是通用的 ``model`` / ``dataset`` / ``trainer``，插入 sys.path
    后会遮蔽同名第三方包。lab 自身不使用这三个名字。
    """
    root = resolve_minimind_root(minimind_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from model.model_minimind import (  # type: ignore  # noqa: E402
        Attention,
        FeedForward,
        MiniMindBlock,
        MiniMindConfig,
        MiniMindForCausalLM,
        MiniMindModel,
        MOEFeedForward,
        RMSNorm,
    )

    return SimpleNamespace(
        root=root,
        commit=minimind_commit(root),
        Attention=Attention,
        FeedForward=FeedForward,
        MiniMindBlock=MiniMindBlock,
        MiniMindConfig=MiniMindConfig,
        MiniMindForCausalLM=MiniMindForCausalLM,
        MiniMindModel=MiniMindModel,
        MOEFeedForward=MOEFeedForward,
        RMSNorm=RMSNorm,
    )


# ---------------------------------------------------------------------------
# 分布式
# ---------------------------------------------------------------------------


@dataclass
class DistInfo:
    """一次运行的分布式上下文。world_size==1 且未 init 时 initialized=False。"""

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
    """auto: 有 CUDA 用 nccl，否则 gloo。"""
    if explicit and explicit != "auto":
        return explicit
    return "nccl" if torch.cuda.is_available() else "gloo"


def init_dist(backend: str = "auto", timeout_s: int = 1800,
              force_cpu: bool = False) -> DistInfo:
    """读取 RANK / LOCAL_RANK / WORLD_SIZE 初始化进程组。

    没有 RANK 环境变量时按单进程处理（不建进程组），这与 MiniMind 的
    ``init_distributed_mode()`` 行为一致。
    """
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


# ---------------------------------------------------------------------------
# dtype 守卫
# ---------------------------------------------------------------------------

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def device_capability(device: str) -> Optional[Tuple[int, int]]:
    """返回 (major, minor)；CPU 或无 CUDA 时返回 None。"""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    idx = 0
    if ":" in str(device):
        idx = int(str(device).split(":")[1])
    return torch.cuda.get_device_capability(idx)


def assert_dtype_supported(dtype: str, device: str = "cpu") -> torch.dtype:
    """检查请求的 dtype 在当前设备上可用，返回对应 torch.dtype。

    本周硬约束：目标机是 V100（sm70），**没有** BF16。任何 bfloat16 请求都在这里
    显式报错，而不是等到 autocast(dtype=bfloat16) 在 CUDA 里抛
    RuntimeError: Current CUDA Device does not support bfloat16。
    """
    key = str(dtype).lower()
    if key not in _DTYPES:
        raise ValueError(
            "不认识的 dtype=" + repr(dtype) + "；可选：float32 / float16 / bfloat16"
        )
    td = _DTYPES[key]
    if td is torch.bfloat16:
        cap = device_capability(device)
        if cap is None:
            raise RuntimeError(
                "拒绝 bfloat16：当前设备不是 CUDA（device=" + str(device) + "）。"
                "本 lab 的目标机是公司 8x V100（compute capability 7.0），"
                "torch.cuda.is_bf16_supported() 要求 major>=8，V100 不满足；"
                "请改用 --dtype float16（配 GradScaler）或 --dtype float32。"
            )
        if cap[0] < 8:
            raise RuntimeError(
                "拒绝 bfloat16：设备 " + str(device) + " 的 compute capability 是 "
                + str(cap[0]) + "." + str(cap[1]) + " (<8.0)。"
                "V100 是 sm70，torch.cuda.is_bf16_supported() 返回 False，"
                "autocast(dtype=bfloat16) 会抛 "
                "'Current CUDA Device does not support bfloat16'。"
                "请改用 --dtype float16（配 GradScaler）或 --dtype float32。"
            )
    return td


def autocast_context(dtype: torch.dtype, device: str):
    """torch 2.1 语法的 autocast 上下文；CPU 上返回 nullcontext（与 MiniMind 一致）。"""
    if not str(device).startswith("cuda"):
        return nullcontext()
    return torch.cuda.amp.autocast(dtype=dtype)


def make_grad_scaler(dtype: torch.dtype, device: str):
    """torch 2.1 语法的 GradScaler；仅 CUDA + fp16 时 enabled。"""
    enabled = bool(str(device).startswith("cuda") and dtype is torch.float16)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


def set_seed(seed: int, per_rank: bool = False) -> int:
    """设置 random / numpy / torch 的种子，返回实际使用的种子。

    ``per_rank=True`` 复刻 MiniMind train_pretrain.py L112 的 setup_seed(42+rank)；
    做 fixed-global-batch 等价实验时必须 ``per_rank=False``。
    """
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    actual = seed + rank if per_rank else seed
    random.seed(actual)
    np.random.seed(actual % (2 ** 31 - 1))
    torch.manual_seed(actual)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual)
        torch.cuda.manual_seed_all(actual)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return actual


# ---------------------------------------------------------------------------
# 配置与模型
# ---------------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    """读取 configs/*.json；必须含 model 与 train 两段。"""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    for key in ("model", "train"):
        if key not in cfg:
            raise ValueError("配置 " + path + " 缺少 '" + key + "' 段")
    return cfg


def build_model(config_json: str, minimind_root: Optional[str] = None,
                device: str = "cpu", override: Optional[Dict[str, Any]] = None):
    """按配置 JSON 构造 MiniMindForCausalLM。

    返回 ``(model, lm_config, cfg_dict, mm)``，其中 ``mm`` 是 :func:`import_minimind`
    的命名空间（后续模块需要 MiniMindBlock / MOEFeedForward 类对象）。
    """
    mm = import_minimind(minimind_root)
    cfg = load_config(config_json)
    model_kwargs = dict(cfg["model"])
    if override:
        model_kwargs.update(override)
    hidden_size = model_kwargs.pop("hidden_size")
    num_hidden_layers = model_kwargs.pop("num_hidden_layers")
    use_moe = bool(model_kwargs.pop("use_moe", False))
    lm_config = mm.MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
        **model_kwargs,
    )
    model = mm.MiniMindForCausalLM(lm_config)
    model = model.to(device)
    return model, lm_config, cfg, mm


def count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------


class TinyDataset(Dataset):
    """合成 token 数据集：第 i 条样本只由 (seed, i) 决定。

    存在的理由：fixed-global-batch 等价实验要求"第 s 步消费的样本集合与 world_size
    无关"。用可按索引复现的合成数据，任何 world_size 都能取到同一条样本，
    并且不依赖 GB 级的 MiniMind 预训练语料。
    """

    def __init__(self, num_samples: int = 4096, seq_len: int = 64,
                 vocab_size: int = 6400, seed: int = 1234):
        if num_samples <= 0 or seq_len <= 1 or vocab_size <= 1:
            raise ValueError("TinyDataset 需要 num_samples>0, seq_len>1, vocab_size>1")
        self.num_samples = int(num_samples)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        idx = int(index) % self.num_samples
        gen = torch.Generator()
        gen.manual_seed((self.seed * 1000003 + idx) % (2 ** 62))
        ids = torch.randint(0, self.vocab_size, (self.seq_len,),
                            generator=gen, dtype=torch.long)
        labels = ids.clone()
        return ids, labels


class RealJsonlDataset(Dataset):
    """走 MiniMind dataset/lm_dataset.py 的 PretrainDataset。

    需要 MiniMind 的 tokenizer（默认 ``<MINIMIND_ROOT>/model``）与 ``datasets`` 包。
    公司环境里数据路径由用户指定，本 lab 不下载、不写出任何数据文件。
    """

    def __init__(self, data_path: str, minimind_root: Optional[str] = None,
                 tokenizer_path: Optional[str] = None, max_length: int = 512):
        root = resolve_minimind_root(minimind_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                "数据文件不存在：" + data_path + "。MiniMind 自带的最小语料是 "
                "<MINIMIND_ROOT>/dataset/pretrain_t2t_mini.jsonl。"
            )
        try:
            from transformers import AutoTokenizer  # noqa: E402
            from dataset.lm_dataset import PretrainDataset  # type: ignore  # noqa: E402
        except ImportError as exc:  # pragma: no cover - 取决于目标机依赖
            raise RuntimeError(
                "导入 MiniMind 的 dataset/lm_dataset.py 失败：" + str(exc)
                + "。它需要 transformers 与 datasets 包；"
                "公司环境若缺少 datasets，请改用 TinyDataset（--dataset tiny）。"
            ) from exc
        tok_path = tokenizer_path or os.path.join(root, "model")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.inner = PretrainDataset(data_path, self.tokenizer, max_length=max_length)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        return self.inner[int(index) % len(self.inner)]


def build_dataset(kind: str, seq_len: int, vocab_size: int,
                  minimind_root: Optional[str] = None,
                  data_path: Optional[str] = None,
                  num_samples: int = 4096, seed: int = 1234) -> Dataset:
    """``kind`` 取 tiny 或 real。"""
    if kind == "tiny":
        return TinyDataset(num_samples=num_samples, seq_len=seq_len,
                           vocab_size=vocab_size, seed=seed)
    if kind == "real":
        if not data_path:
            raise ValueError("--dataset real 必须给 --data-path <jsonl>")
        return RealJsonlDataset(data_path, minimind_root=minimind_root,
                                max_length=seq_len)
    raise ValueError("不认识的 dataset=" + repr(kind) + "；可选 tiny / real")


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

    第 ``step``（1-based）个 optimizer step 消费全局索引区间
    ``[(step-1)*G, step*G)``；该区间被切成 ``world_size*accum`` 个等长 chunk，
    chunk 编号 ``rank*accum + acc_idx``。

    因此对任意 (world_size, accum) 组合，只要 G 与 micro 相同，
    第 step 步消费的**样本集合完全相同**——这正是不变量 I1 的数据侧条件。
    """
    if global_batch % (world_size * accum) != 0:
        raise ValueError(
            "global_batch=" + str(global_batch) + " 不能被 world_size*accum="
            + str(world_size) + "*" + str(accum) + "=" + str(world_size * accum) + " 整除"
        )
    micro = global_batch // (world_size * accum)
    chunk_id = rank * accum + acc_idx
    start = (step - 1) * global_batch + chunk_id * micro
    return list(range(start, start + micro))


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------


class JsonlLogger:
    """每步一行 JSON。所有文件都写在 ``--out-dir`` 下（公司内路径由用户指定）。"""

    def __init__(self, out_dir: str, name: str, rank: int = 0,
                 rank_in_name: bool = True, enabled: bool = True):
        self.enabled = enabled
        self.rank = rank
        self.path: Optional[str] = None
        self._fh = None
        if not enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        fname = (name + "_rank" + str(rank) + ".jsonl") if rank_in_name else (name + ".jsonl")
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
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


def peak_mem_mb(device: str) -> float:
    """CUDA 峰值显存 MB；CPU 返回 0.0（本机没有可比的等价量）。"""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        return torch.cuda.max_memory_allocated(idx) / (1024.0 * 1024.0)
    return 0.0


def reset_peak_mem(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        torch.cuda.reset_peak_memory_stats(idx)


def cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    """与 MiniMind trainer_utils.get_lr 完全一致的调度。"""
    total = max(int(total_steps), 1)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))


def sdpa_context(backend: str, device: str):
    """torch 2.1 的 torch.backends.cuda.sdp_kernel 上下文。

    仅在 CUDA 上生效；backend=auto 或非 CUDA 时返回 nullcontext。
    V100(sm70) 在 torch 2.1 下没有 flash 后端，只有 mem_efficient / math。
    """
    if backend == "auto" or not str(device).startswith("cuda"):
        return nullcontext()
    flags = {
        "math": dict(enable_flash=False, enable_mem_efficient=False, enable_math=True),
        "mem_efficient": dict(enable_flash=False, enable_mem_efficient=True,
                              enable_math=False),
        "flash": dict(enable_flash=True, enable_mem_efficient=False, enable_math=False),
    }
    if backend not in flags:
        raise ValueError(
            "不认识的 sdpa 后端 " + repr(backend) + "；可选 auto/math/mem_efficient/flash"
        )
    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        # torch 2.1 一定有 sdp_kernel；本机 2.14 若已移除则退化为 auto 并提示。
        print("[warn] 本地 torch " + torch.__version__
              + " 没有 torch.backends.cuda.sdp_kernel，--sdpa-backend 被忽略；"
              "目标机 torch 2.1.0 有该 API。", flush=True)
        return nullcontext()
    return torch.backends.cuda.sdp_kernel(**flags[backend])


def env_summary() -> Dict[str, Any]:
    """写进日志头部的无敏感信息环境摘要。"""
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "target_torch": TARGET_TORCH_VERSION,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def add_common_args(parser) -> None:
    """所有入口脚本共享的参数。"""
    parser.add_argument("--config", type=str, default=None,
                        help="configs/*.json 路径")
    parser.add_argument("--minimind-root", type=str, default=None,
                        help="MiniMind 仓库根目录（默认读环境变量 MINIMIND_ROOT）")
    parser.add_argument("--out-dir", type=str, default="./runs",
                        help="日志/产物目录；公司环境里请指定公司内部路径")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16"],
                        help="混合精度；V100 上 bfloat16 会被显式拒绝")
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子")
    parser.add_argument("--seed-per-rank", type=int, default=1, choices=[0, 1],
                        help="1=复刻 MiniMind 的 42+rank；等价实验必须用 0")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="optimizer step 上限（公司环境每次运行 <=10 分钟）")
    parser.add_argument("--nccl-timeout-s", type=int, default=600,
                        help="进程组超时秒数；默认 600 而不是 torch 默认的 1800")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "nccl", "gloo"])
    parser.add_argument("--force-cpu", action="store_true",
                        help="即使有 CUDA 也跑 CPU（本机 gloo 验证用）")
