"""Day 3 字节账：param / grad / optim / activation 四项，每卡各占多少字节。

来源
----
新写。week02 只有「跑一遍看 peak_mem」，本周要求先闭卷手算再实测对照，
所以手算侧必须是一个可以逐项核对的公式表，而不是一个数。

四项各是什么（这是 Gate 的 A0 题）
----------------------------------
param      参数本体。AMP 训练时参数留在 fp32（autocast 只影响算子的输入输出，
           不改 nn.Parameter 的 dtype），所以是 4 字节/元素，不是 2。
grad       梯度。与参数同 dtype 同形状，所以 DDP 下也是 4 字节/元素。
optim      优化器状态。AdamW 每个参数两份 fp32（exp_avg、exp_avg_sq），
           所以是 8 字节/元素。这一项通常比参数本身还大，是「模型明明只有
           几十 M 却塞不下」的主要来源。
activation 为反向保存的中间张量。它是唯一随 batch 和 seq 线性增长的一项，
           也是唯一能靠 activation checkpointing 换回来的一项。

三种放置下每卡的分子（W = world_size）
--------------------------------------
DDP / FSDP NO_SHARD   param=4P    grad=4P    optim=8P
FSDP SHARD_GRAD_OP    param=4P    grad=4P/W  optim=8P/W
FSDP FULL_SHARD       param=4P/W  grad=4P/W  optim=8P/W
                      + 瞬时的 all_gather buffer = 单个 FSDP 单元的参数量
                        × param_dtype 字节（MixedPrecision 下是 2）

FULL_SHARD 省的是常驻，付的是通信与那个瞬时 buffer。如果 auto_wrap 退化成
「只有一个根单元」，那个 buffer 就等于整份模型，FSDP 白做——这是失败模式之一，
所以 n_units 是手算表里的显式输入，不是隐含假设。

activation 的手算是估算，不是恒等式
-----------------------------------
autograd 到底存了哪些中间张量，取决于算子实现（尤其是 SDPA 走哪个后端）。
所以本文件的 activation 手算是一个**带命名系数的模型**，只声明到「估算」级别；
真正可验证的是它的**标度律**：对 B 严格线性，对 T 至少线性。
measure_activation_bytes() 用 saved_tensors_hooks 实测，两者并排给出比值。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from . import common as C
from . import model as M

BYTES_FP32 = 4
BYTES_FP16 = 2
# AdamW 每个参数张量额外存的 step 标量：torch 里是 0 维 float32 张量。
ADAM_STEP_BYTES = 4

# activation 手算的系数：每层每个 token 保存多少个 hidden 宽度 / intermediate 宽度
# 的张量。数字来自逐行数 model.py 里 Block.forward 会被 autograd 保存的张量：
#   input_layernorm 的输入与 rsqrt 中间量、q/k/v 三个投影的输出、
#   rope 后的 q/k、attention 输出、o_proj 输入、残差、post norm、
#   gate/up 的输出、silu 的输出、乘积。
# 这两个系数是估算，V100 上换 SDPA 后端会变，所以报告里必须带 estimate 标签。
ACT_COEF_HIDDEN = 8.0
ACT_COEF_INTER = 4.0


def dtype_bytes(name: str) -> int:
    key = str(name).lower()
    if key in ("float32", "fp32"):
        return BYTES_FP32
    if key in ("float16", "fp16", "bfloat16", "bf16"):
        return BYTES_FP16
    raise ValueError("不认识的 dtype " + repr(name))


# ---------------------------------------------------------------------------
# 手算
# ---------------------------------------------------------------------------


def hand_ledger(model_cfg: Dict[str, Any], world_size: int = 1,
                mode: str = "ddp", batch_per_rank: int = 1, seq_len: int = 512,
                n_units: Optional[int] = None,
                param_dtype: str = "float32",
                compute_dtype: str = "float16",
                optim_states: int = 2) -> Dict[str, Any]:
    """闭卷手算表。只吃配置，不建模型——所以它可以在没有 GPU 的地方先算好。"""
    cfg = M.config_from_dict(dict(model_cfg))
    breakdown = M.param_breakdown(cfg)
    P = int(breakdown["total"])
    W = max(int(world_size), 1)
    pb = dtype_bytes(param_dtype)
    cb = dtype_bytes(compute_dtype)
    units = int(n_units) if n_units is not None else (cfg.num_hidden_layers + 1)

    shard_param = {"ddp": 1, "no_shard": 1, "shard_grad_op": 1,
                   "full_shard": W}
    shard_grad = {"ddp": 1, "no_shard": 1, "shard_grad_op": W, "full_shard": W}
    if mode not in shard_param:
        raise ValueError("mode 只能是 " + " / ".join(sorted(shard_param)))

    param_bytes = P * pb / shard_param[mode]
    grad_bytes = P * pb / shard_grad[mode]
    # AdamW 除了 exp_avg / exp_avg_sq，还给**每个参数张量**存一个 fp32 的 step
    # 标量。张量数量级只有几十，字节数只有几十到几百——但少了这一项，
    # 手算列和实测列就永远差那么一点，看起来像「测量误差」，其实是漏了一项。
    n_tensors = int(breakdown["n_tensors"])
    optim_bytes = (P * BYTES_FP32 * int(optim_states) / shard_grad[mode]
                   + n_tensors * ADAM_STEP_BYTES)

    unit_params = P / max(units, 1)
    gather_buffer = (unit_params * cb) if mode == "full_shard" else 0.0

    B = int(batch_per_rank)
    T = int(seq_len)
    H = cfg.hidden_size
    I = int(cfg.intermediate_size)
    V = cfg.vocab_size
    L = cfg.num_hidden_layers
    act_layers = L * B * T * (ACT_COEF_HIDDEN * H + ACT_COEF_INTER * I) * cb
    # logits 与 cross_entropy 里的 float() 副本：小模型上这一项常常最大。
    act_logits = B * T * V * cb + B * T * V * BYTES_FP32
    act_embed = B * T * H * cb
    act_bytes = act_layers + act_logits + act_embed

    total = param_bytes + grad_bytes + optim_bytes + act_bytes
    return {
        "schema": "mm_v100.ledger.hand/1",
        "label": "estimate(activation) + exact(param/grad/optim)",
        "mode": mode,
        "world_size": W,
        "n_units": units,
        "params_total": P,
        "params_breakdown": breakdown,
        "n_param_tensors": n_tensors,
        "batch_per_rank": B,
        "seq_len": T,
        "param_bytes": int(param_bytes),
        "grad_bytes": int(grad_bytes),
        "optim_bytes": int(optim_bytes),
        "activation_bytes": int(act_bytes),
        "activation_parts": {
            "layers": int(act_layers),
            "logits": int(act_logits),
            "embed": int(act_embed),
        },
        "all_gather_buffer_bytes": int(gather_buffer),
        "total_bytes": int(total),
        "total_gib": total / (1024 ** 3),
        "formulas": {
            "param": "P * " + str(pb) + " / " + str(shard_param[mode]),
            "grad": "P * " + str(pb) + " / " + str(shard_grad[mode]),
            "optim": ("P * 4 * " + str(optim_states) + " / "
                      + str(shard_grad[mode]) + " + n_tensors * "
                      + str(ADAM_STEP_BYTES)),
            "activation": ("L*B*T*(" + str(ACT_COEF_HIDDEN) + "*H + "
                           + str(ACT_COEF_INTER) + "*I)*" + str(cb)
                           + " + B*T*V*(" + str(cb) + "+4) + B*T*H*" + str(cb)),
        },
    }


# ---------------------------------------------------------------------------
# 实测
# ---------------------------------------------------------------------------


def param_storage_ptrs(model: torch.nn.Module) -> Set[int]:
    """所有参数的 storage 指针。测 activation 时要把它们排除掉。"""
    ptrs: Set[int] = set()
    for p in model.parameters():
        ptrs.add(storage_key(p)[0])
    return ptrs


def storage_key(t: torch.Tensor) -> Tuple[int, int]:
    """(data_ptr, nbytes)。同一块 storage 的多个 view 只能算一次。"""
    try:
        st = t.untyped_storage()
        return (int(st.data_ptr()), int(st.nbytes()))
    except (AttributeError, RuntimeError):
        return (int(t.data_ptr()), int(t.numel()) * int(t.element_size()))


class SavedActivationMeter:
    """用 saved_tensors_hooks 统计「为反向保存了多少字节」。

    这是 CPU 上唯一靠得住的 activation 度量：torch.cuda.max_memory_allocated
    在没有 GPU 的机器上恒为 0，而 saved_tensors_hooks 与设备无关。
    到了 V100 上两个都测，可以互相印证。
    """

    def __init__(self, exclude_ptrs: Optional[Set[int]] = None) -> None:
        self.exclude = set(exclude_ptrs or ())
        self.seen: Dict[Tuple[int, int], int] = {}
        self.n_packed = 0
        self._ctx = None

    def _pack(self, t: torch.Tensor):
        self.n_packed += 1
        key = storage_key(t)
        if key[0] not in self.exclude:
            self.seen[key] = key[1]
        return t

    @staticmethod
    def _unpack(t: torch.Tensor) -> torch.Tensor:
        return t

    def total_bytes(self) -> int:
        return int(sum(self.seen.values()))

    def n_unique(self) -> int:
        return len(self.seen)

    def __enter__(self) -> "SavedActivationMeter":
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack,
                                                             self._unpack)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        if self._ctx is not None:
            self._ctx.__exit__(*exc)
            self._ctx = None
        return False


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """优化器状态的真实字节数。必须在至少一次 step 之后调用：
    AdamW 的 exp_avg / exp_avg_sq 是惰性创建的，step 之前是 0。"""
    total = 0
    seen: Set[Tuple[int, int]] = set()
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                key = storage_key(value)
                if key in seen:
                    continue
                seen.add(key)
                total += key[1]
    return int(total)


def param_and_grad_bytes(model: torch.nn.Module) -> Dict[str, int]:
    """去重后的参数与梯度字节。tied embedding 只能算一次。"""
    seen: Set[int] = set()
    p_bytes = 0
    g_bytes = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        p_bytes += p.numel() * p.element_size()
        if p.grad is not None:
            g_bytes += p.grad.numel() * p.grad.element_size()
    return {"param_bytes": int(p_bytes), "grad_bytes": int(g_bytes)}


def measure(model: torch.nn.Module, input_ids: torch.Tensor,
            labels: torch.Tensor, optimizer: torch.optim.Optimizer,
            dtype_name: str = "float32", device: str = "cpu",
            sdpa_backend: str = "auto") -> Dict[str, Any]:
    """跑一次完整的 forward+backward+step，实测四项字节。

    顺序有讲究：activation 只在 backward 之前存在，所以必须在
    saved_tensors_hooks 里做 forward；optim 只在 step 之后存在，所以要最后测。
    """
    torch_dtype = C.assert_dtype_supported(dtype_name, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    meter = SavedActivationMeter(param_storage_ptrs(model))
    C.reset_peak_mem(device)
    with meter:
        with C.sdpa_context(sdpa_backend, device):
            with C.autocast_context(torch_dtype, device):
                out = model(input_ids, labels=labels)
                loss = out.loss
        act_bytes = meter.total_bytes()
        n_unique = meter.n_unique()
        loss.backward()
    pg = param_and_grad_bytes(model)
    optimizer.step()
    opt_bytes = optimizer_state_bytes(optimizer)
    return {
        "schema": "mm_v100.ledger.measured/1",
        "param_bytes": pg["param_bytes"],
        "grad_bytes": pg["grad_bytes"],
        "optim_bytes": int(opt_bytes),
        "activation_bytes": int(act_bytes),
        "activation_tensors": int(n_unique),
        "peak_mem_mb": C.peak_mem_mb(device),
        "loss": float(loss.detach().float().item()),
        "dtype": dtype_name,
    }


# ---------------------------------------------------------------------------
# 对照表
# ---------------------------------------------------------------------------

ROWS = ("param_bytes", "grad_bytes", "optim_bytes", "activation_bytes")


def compare(hand: Dict[str, Any], measured: Dict[str, Any]) -> Dict[str, Any]:
    """手算 vs 实测的偏差表。param/grad/optim 要求相等，activation 只看比值。"""
    rows: List[Dict[str, Any]] = []
    for key in ROWS:
        h = float(hand.get(key, 0))
        m = float(measured.get(key, 0))
        ratio = (m / h) if h else (0.0 if m == 0 else float("inf"))
        exact_required = key != "activation_bytes"
        rows.append({
            "item": key,
            "hand": int(h),
            "measured": int(m),
            "delta": int(m - h),
            "ratio": ratio,
            "exact_required": exact_required,
            "ok": (m == h) if exact_required else (0.25 <= ratio <= 4.0),
        })
    return {
        "rows": rows,
        "exact_items_match": all(r["ok"] for r in rows if r["exact_required"]),
        "activation_ratio": next(r["ratio"] for r in rows
                                 if r["item"] == "activation_bytes"),
    }


def render_markdown(hand: Dict[str, Any], measured: Dict[str, Any]) -> str:
    """Day 3 的 byte_ledger.md：手算列 + 实测列 + 偏差列。"""
    cmp = compare(hand, measured)
    lines: List[str] = []
    lines.append("# 字节账（每卡）")
    lines.append("")
    lines.append("- 放置：`" + str(hand["mode"]) + "`，world_size="
                 + str(hand["world_size"]) + "，n_units=" + str(hand["n_units"]))
    lines.append("- 参数量 P = " + format(hand["params_total"], ",")
                 + "，batch/rank=" + str(hand["batch_per_rank"])
                 + "，seq_len=" + str(hand["seq_len"]))
    lines.append("")
    lines.append("| 项 | 手算 (B) | 实测 (B) | 偏差 (B) | 比值 | 判定 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in cmp["rows"]:
        verdict = "一致" if row["ok"] else "不一致"
        if not row["exact_required"]:
            verdict += "（估算项，只看量级）"
        lines.append("| `%s` | %s | %s | %s | %.3f | %s |"
                     % (row["item"], format(row["hand"], ","),
                        format(row["measured"], ","), format(row["delta"], ","),
                        row["ratio"], verdict))
    lines.append("")
    lines.append("公式：")
    for key, formula in hand["formulas"].items():
        lines.append("- `" + key + "` = " + formula)
    lines.append("")
    lines.append("all_gather buffer（仅 FULL_SHARD）= "
                 + format(hand["all_gather_buffer_bytes"], ",") + " B")
    lines.append("")
    lines.append("activation 分项（手算，估算）：")
    for key, value in hand["activation_parts"].items():
        lines.append("- " + key + " = " + format(value, ",") + " B")
    return "\n".join(lines)


def to_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
