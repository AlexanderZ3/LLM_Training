"""Day 2/3 通信账：一个 step 内的集合通信，类型、次数、字节量。

来源
----
从 week02 lab/src/mm_dist/train_fsdp.py 里的 CollectiveCounter 复制并扩展：
原版只数次数，这里补上每次调用的**字节量**和张量 dtype，并加了一张
「手算期望值」表供 Day 3 的闭卷题对照。

做法
----
临时替换 torch.distributed 模块上的公共函数名。torch 2.1 的 DDP reducer 与 FSDP
(torch/distributed/fsdp/_flat_param.py) 都是通过 dist.all_reduce /
dist.all_gather_into_tensor / dist.reduce_scatter_tensor 这样的**模块属性**调用的，
所以替换模块属性能拦到。

只包公共名，不包 _all_gather_base / _reduce_scatter_base：后者在 2.1 里是前者的
deprecated 包装，两个都包会重复计数。

一个诚实的限制
--------------
DDP 的 gradient allreduce 有一部分在 C++ reducer 里直接走 ProcessGroup，
不经过 Python 的 dist.all_reduce。所以本计数器在 DDP 下**可能少数**。
判断办法：如果 all_reduce 次数为 0 但 loss 确实在多卡间同步，那就是走了 C++ 路径。
Day 3 的通信账因此以 FSDP 路径为主（FSDP 的 all_gather / reduce_scatter 是
Python 侧调用，能数全），DDP 侧用「跨 rank 梯度一致性」这个不变量来验证，
而不是靠次数。这一点写进 02_LAB_GUIDE 的故障树。
"""

from __future__ import annotations

import functools
import json
import math
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

# 被包住的集合通信名字，以及「payload 张量是第几个位置参数 / 哪个关键字」
# 0 表示第一个位置参数，1 表示第二个。payload 取的是**本 rank 发出去的那份**，
# 因为通信账关心的是每张卡的出口带宽，不是聚合后的总量。
WRAPPED: Dict[str, Dict[str, Any]] = {
    "all_reduce":            {"payload_pos": 0, "payload_kw": "tensor"},
    "broadcast":             {"payload_pos": 0, "payload_kw": "tensor"},
    "all_gather":            {"payload_pos": 1, "payload_kw": "tensor"},
    "all_gather_into_tensor": {"payload_pos": 1, "payload_kw": "input_tensor"},
    "reduce_scatter":        {"payload_pos": 1, "payload_kw": "input_list"},
    "reduce_scatter_tensor": {"payload_pos": 1, "payload_kw": "input"},
    "all_to_all_single":     {"payload_pos": 1, "payload_kw": "input"},
    "barrier":               {"payload_pos": None, "payload_kw": None},
}


def _tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return int(obj.numel()) * int(obj.element_size())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(x) for x in obj)
    return 0


def _tensor_dtype(obj: Any) -> Optional[str]:
    if isinstance(obj, torch.Tensor):
        return str(obj.dtype)
    if isinstance(obj, (list, tuple)) and obj:
        return _tensor_dtype(obj[0])
    return None


class CollectiveMeter:
    """统计一段代码里各类集合通信的次数与字节量。

    用法::

        with CollectiveMeter() as meter:
            ...一个 step...
        print(meter.summary())
    """

    def __init__(self, record_calls: bool = True, max_records: int = 4096) -> None:
        self.counts: Dict[str, int] = {n: 0 for n in WRAPPED}
        self.bytes: Dict[str, int] = {n: 0 for n in WRAPPED}
        self.records: List[Dict[str, Any]] = []
        self.record_calls = bool(record_calls)
        self.max_records = int(max_records)
        self._orig: Dict[str, Any] = {}
        self._active = False

    def reset(self) -> None:
        for n in WRAPPED:
            self.counts[n] = 0
            self.bytes[n] = 0
        self.records.clear()

    def total_calls(self) -> int:
        return sum(self.counts.values())

    def total_bytes(self) -> int:
        return sum(self.bytes.values())

    def summary(self) -> Dict[str, Any]:
        used = {n: self.counts[n] for n in WRAPPED if self.counts[n]}
        used_bytes = {n: self.bytes[n] for n in WRAPPED if self.counts[n]}
        return {
            "counts": used,
            "bytes": used_bytes,
            "total_calls": self.total_calls(),
            "total_bytes": self.total_bytes(),
            "total_mib": self.total_bytes() / (1024.0 * 1024.0),
        }

    def _wrap(self, name: str, fn):
        spec = WRAPPED[name]

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            payload: Any = None
            pos = spec["payload_pos"]
            kw = spec["payload_kw"]
            if kw is not None and kw in kwargs:
                payload = kwargs[kw]
            elif pos is not None and len(args) > pos:
                payload = args[pos]
            nbytes = _tensor_bytes(payload)
            self.counts[name] += 1
            self.bytes[name] += nbytes
            if self.record_calls and len(self.records) < self.max_records:
                self.records.append({
                    "op": name,
                    "bytes": nbytes,
                    "dtype": _tensor_dtype(payload),
                })
            return fn(*args, **kwargs)

        return wrapper

    def __enter__(self) -> "CollectiveMeter":
        if self._active:
            return self
        for name in WRAPPED:
            fn = getattr(dist, name, None)
            if fn is None:
                continue
            self._orig[name] = fn
            setattr(dist, name, self._wrap(name, fn))
        self._active = True
        return self

    def __exit__(self, *exc) -> bool:
        for name, fn in self._orig.items():
            setattr(dist, name, fn)
        self._orig.clear()
        self._active = False
        return False


# ---------------------------------------------------------------------------
# 手算期望
# ---------------------------------------------------------------------------


def ddp_expected(n_params: int, grad_dtype_bytes: int = 4,
                 bucket_cap_mb: float = 25.0) -> Dict[str, Any]:
    """DDP 一个 optimizer step 的手算通信量。

    DDP 把梯度装进 bucket，装满一个就发一次 allreduce。默认 bucket 是 25 MB
    （torch 2.1 的 DistributedDataParallel(bucket_cap_mb=25)）。
    所以次数 = ceil(梯度总字节 / bucket 字节)，每次的量 = 一个 bucket。

    单次 allreduce 的实际线上流量是 2*(W-1)/W * bytes（ring 实现：
    reduce-scatter 一圈 + all-gather 一圈），这里 wire_bytes_factor 就是那个系数。
    """
    grad_bytes = int(n_params) * int(grad_dtype_bytes)
    bucket_bytes = int(bucket_cap_mb * 1024 * 1024)
    n_calls = max(1, int(math.ceil(grad_bytes / float(bucket_bytes))))
    return {
        "mode": "ddp",
        "grad_bytes_per_rank": grad_bytes,
        "bucket_bytes": bucket_bytes,
        "expected_all_reduce_calls": n_calls,
        "expected_all_reduce_bytes": grad_bytes,
        "wire_bytes_formula": "2*(W-1)/W * grad_bytes（ring allreduce）",
        "note": ("部分 allreduce 在 C++ reducer 里直接走 ProcessGroup，"
                 "Python 侧的 CollectiveMeter 可能数不到；次数只作参考，"
                 "DDP 的判据用跨 rank 梯度一致性。"),
    }


def fsdp_expected(n_params: int, n_units: int, world_size: int,
                  param_dtype_bytes: int = 2, grad_dtype_bytes: int = 2,
                  sharding: str = "full_shard") -> Dict[str, Any]:
    """FSDP 一个 step 的手算通信量（每 rank）。

    FULL_SHARD：
      forward   每个 FSDP 单元一次 all_gather（把参数分片拼成整份）
      backward  每个单元再一次 all_gather（重新拼参数）
                + 每个单元一次 reduce_scatter（把梯度切回分片）
      合计 2*n_units 次 all_gather + n_units 次 reduce_scatter。
    SHARD_GRAD_OP：
      参数在 forward 后不释放，backward 不再 all_gather。
      合计 n_units 次 all_gather + n_units 次 reduce_scatter。
    NO_SHARD：
      退化成 DDP，只有梯度 allreduce。

    每次 all_gather 本 rank 发出的是自己那一片：unit_params/world_size 个元素。
    """
    if world_size <= 0:
        raise ValueError("world_size 必须 > 0")
    per_unit_params = int(n_params) / max(int(n_units), 1)
    ag_payload = per_unit_params / world_size * param_dtype_bytes
    rs_payload = per_unit_params * grad_dtype_bytes
    table = {
        "full_shard": (2 * n_units, n_units),
        "shard_grad_op": (n_units, n_units),
        "no_shard": (0, 0),
    }
    if sharding not in table:
        raise ValueError("sharding 只能是 " + " / ".join(sorted(table)))
    n_ag, n_rs = table[sharding]
    return {
        "mode": "fsdp:" + sharding,
        "n_units": int(n_units),
        "world_size": int(world_size),
        "expected_all_gather_calls": int(n_ag),
        "expected_reduce_scatter_calls": int(n_rs),
        "all_gather_payload_bytes_each": int(ag_payload),
        "reduce_scatter_payload_bytes_each": int(rs_payload),
        "expected_total_bytes": int(n_ag * ag_payload + n_rs * rs_payload),
        "note": ("no_shard 退化成 DDP，只剩梯度 allreduce；"
                 "reduce_scatter 的输入是整份梯度，输出才是分片。"),
    }


def compare_to_expected(measured: Dict[str, Any],
                        expected: Dict[str, Any],
                        tol_ratio: float = 0.25) -> Dict[str, Any]:
    """把实测 summary 与手算表并排，给出偏差比与判定。

    tol_ratio 默认 0.25：FSDP 会为 buffer / 根单元多发几次，25% 以内属正常；
    超过说明 wrap 粒度或 sharding 策略与你以为的不一样。
    """
    counts = measured.get("counts", {})
    rows: List[Dict[str, Any]] = []
    pairs = [
        ("all_gather_into_tensor", "expected_all_gather_calls"),
        ("reduce_scatter_tensor", "expected_reduce_scatter_calls"),
        ("all_reduce", "expected_all_reduce_calls"),
    ]
    for op, key in pairs:
        if key not in expected:
            continue
        exp = float(expected[key])
        act = float(counts.get(op, 0))
        ratio = (act / exp) if exp else (0.0 if act == 0 else float("inf"))
        rows.append({
            "op": op,
            "expected": exp,
            "measured": act,
            "ratio": ratio,
            "within_tol": bool(exp == 0 and act == 0)
                          or bool(exp > 0 and abs(ratio - 1.0) <= tol_ratio),
        })
    return {
        "rows": rows,
        "all_within_tol": all(r["within_tol"] for r in rows) if rows else False,
        "tol_ratio": tol_ratio,
    }


def render_text(measured: Dict[str, Any],
                expected: Optional[Dict[str, Any]] = None) -> str:
    lines = ["== 通信账 =="]
    lines.append("%-24s %8s %14s" % ("op", "calls", "bytes"))
    for op, n in sorted(measured.get("counts", {}).items()):
        lines.append("%-24s %8d %14d" % (op, n, measured["bytes"].get(op, 0)))
    lines.append("%-24s %8d %14d" % ("TOTAL", measured.get("total_calls", 0),
                                     measured.get("total_bytes", 0)))
    if expected is not None:
        cmp = compare_to_expected(measured, expected)
        lines.append("")
        lines.append("手算对照（" + str(expected.get("mode")) + "）:")
        lines.append("%-24s %10s %10s %8s %6s"
                     % ("op", "expected", "measured", "ratio", "ok"))
        for row in cmp["rows"]:
            lines.append("%-24s %10.0f %10.0f %8.2f %6s"
                         % (row["op"], row["expected"], row["measured"],
                            row["ratio"], "yes" if row["within_tol"] else "no"))
    return "\n".join(lines)


def to_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
