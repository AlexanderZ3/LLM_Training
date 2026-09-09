"""Day 2 三档静默故障：注入、观测、判别。

来源
----
从 week01 lab/src/mm_probe/mask_fault.py（label mask 错位）与 week02
lab/src/mm_dist/faults.py（kill rank / NCCL 环境）复制并重组。原来两处的故障
都是「会报错」或「会 hang」的显性故障；本周要练的是**不报错也不 hang**、
loss 曲线看起来还挺正常、但学习信号已经断了的那三种。

三档
----
A label_shift   labels 整体挪一格。n_label_tokens 不变，loss 量级不变，
                曲线照样往下走（模型学会了「预测上一个 token」这个更简单的任务），
                但学到的东西是错的。
B scaler_stuck  fp16 的 GradScaler 每步都发现 inf，于是每步都跳过 optimizer.step。
                loss 会缓慢波动（因为数据在换），但参数一步没动。
C ddp_no_sync   DDP 的梯度从不同步。每张卡各自用自己的 micro-batch 更新，
                跑几百步后 8 张卡变成 8 个不同的模型。rank0 的 loss 曲线
                完全正常——它确实在下降，只是下降的是 1/8 的模型。

三个不变量（每档只被其中一个抓住）
----------------------------------
I1 label_align_violations   labels 非 -100 处必须等于 input_ids。A 违反，B/C 不违反。
I2 optimizer_step_ratio     applied_steps / total_steps 应接近 1；
                            同时 param_delta_norm > 0。B 违反，A/C 不违反。
I3 cross_rank_grad_delta    同步后各 rank 的梯度应逐元素相同（相对偏差为 0）。
                            C 违反，A/B 不违反。

判别表（Day 2 要手写的那三行）
------------------------------
              I1>0    I2低    I3>0
  A            是      否      否
  B            否      是      否
  C            否      否      是

每一档都有**单个充分**的不变量，所以三行表是可判定的。反过来，
「loss 在下降」对三档都成立，它是必要条件而非充分条件——这正是本周的主命题。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from .data_contract import IGNORE_INDEX, label_alignment_violations

FAULT_MODES = ("none", "label_shift", "scaler_stuck", "ddp_no_sync")

FAULT_DESCRIPTIONS = {
    "none": "正常对照组。",
    "label_shift": ("labels 整体右移一格：位置 i 的 target 变成了 input_ids[i-1]。"
                    "n_label_tokens 与 loss 量级都不变，曲线照样下降。"),
    "scaler_stuck": ("每步往一个参数的梯度里塞 inf，GradScaler 每步都跳过 step。"
                     "loss 有波动但参数不动。"),
    "ddp_no_sync": ("所有 micro-batch 都在 no_sync() 里做，梯度从不 allreduce。"
                    "单看 rank0 的日志一切正常。"),
}


# ---------------------------------------------------------------------------
# A: label 错位
# ---------------------------------------------------------------------------


def apply_label_fault(input_ids: torch.Tensor, labels: torch.Tensor,
                      mode: str) -> torch.Tensor:
    """返回注入故障后的 labels（新张量，不改原值）。

    label_shift 用 roll 而不是切片补 -100：切片会改变 n_label_tokens，
    那样第二个不变量就能抓到它，故障也就不「静默」了。roll 保持计数不变，
    只有 I1 能抓——这才是要练的那种。
    """
    if mode not in FAULT_MODES:
        raise ValueError("未知 mode=" + repr(mode) + "；可选 " + ", ".join(FAULT_MODES))
    out = labels.clone()
    if mode != "label_shift":
        return out
    # 有效位置的集合原样保留（所以 n_label_tokens 逐位不变），
    # 只把每个有效位置的 target 换成前一个 token。
    active = out != IGNORE_INDEX
    shifted_ids = torch.roll(input_ids, shifts=1, dims=-1)
    ignore = torch.full_like(out, IGNORE_INDEX)
    return torch.where(active, shifted_ids.to(out.dtype), ignore)


# ---------------------------------------------------------------------------
# B: scaler 持续跳步
# ---------------------------------------------------------------------------


class GradInfInjector:
    """在反向里往某一个参数的梯度上加 inf，让 GradScaler 每步都跳过。

    用 register_hook 而不是在 backward 之后直接改 .grad：后者在 DDP 下会被
    allreduce 之前的 bucket 复制绕过，注入不进去。挂在 tensor 上的 hook
    是梯度形成的路径上唯一一定会经过的地方。
    """

    def __init__(self, model: torch.nn.Module, enabled: bool = True,
                 param_index: int = 0) -> None:
        self.enabled = bool(enabled)
        self.handles: List[Any] = []
        self.n_fired = 0
        if not self.enabled:
            return
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("模型里没有可训练参数，注入不了 scaler_stuck 故障")
        target = params[min(int(param_index), len(params) - 1)]

        def hook(grad: torch.Tensor) -> torch.Tensor:
            self.n_fired += 1
            poisoned = grad.clone()
            poisoned.view(-1)[0] = float("inf")
            return poisoned

        self.handles.append(target.register_hook(hook))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> "GradInfInjector":
        return self

    def __exit__(self, *exc) -> bool:
        self.remove()
        return False


# ---------------------------------------------------------------------------
# C: DDP 梯度不同步
# ---------------------------------------------------------------------------


def should_sync_this_micro(mode: str, is_last_micro: bool) -> bool:
    """训练循环用它决定这个 micro-batch 要不要走 DDP 的同步路径。

    正常：只有最后一个 micro 同步（前面的用 no_sync 累加，省 accum-1 次 allreduce）。
    ddp_no_sync 故障：一次都不同步。
    """
    if mode == "ddp_no_sync":
        return False
    return bool(is_last_micro)


def cross_rank_grad_delta(model: torch.nn.Module, world_size: int,
                          n_params: int = 8,
                          n_elems: int = 1024) -> Optional[float]:
    """不变量 I3：同步之后各 rank 的梯度应当逐元素相同，返回相对最大偏差。

    为什么比的是**元素**而不是范数
    ------------------------------
    第一版比的是各 rank 的梯度全局范数之差。它在真实实验里几乎抓不到故障 C：
    不同 micro-batch 的梯度方向不同，但范数往往非常接近（小模型 + 随机数据上
    实测只差 3.6e-07，落在任何合理阈值以下）。范数是一个把方向信息全部丢掉的
    标量，而故障 C 恰恰只改变方向。所以这里改成 all_gather 一段**梯度指纹**
    （前 n_params 个参数张量各取前 n_elems 个元素），逐元素比。

    返回值是**相对**偏差 max|g_i - g_0| / mean|g_0|，与梯度量级无关，
    因此同一个阈值可以在 tiny_cpu 与 v100_768 上通用。
    正常 DDP 下它精确为 0.0（allreduce 之后各 rank 拿到同一份 buffer）。

    单进程时返回 None：这个不变量在 world_size=1 下没有内容，
    不能拿 0.0 冒充「通过」。
    """
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return None
    chunks: List[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            continue
        chunks.append(p.grad.detach().float().reshape(-1)[:n_elems])
        if len(chunks) >= int(n_params):
            break
    if not chunks:
        return None
    local = torch.cat(chunks).contiguous()
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    ref = gathered[0]
    scale = float(ref.abs().mean().item()) + 1e-12
    max_dev = 0.0
    for other in gathered[1:]:
        max_dev = max(max_dev, float((other - ref).abs().max().item()))
    return float(max_dev / scale)


# ---------------------------------------------------------------------------
# 观测与判别
# ---------------------------------------------------------------------------


def observe(input_ids: torch.Tensor, labels: torch.Tensor,
            applied_steps: int, total_steps: int, param_delta_norm: float,
            grad_delta: Optional[float]) -> Dict[str, Any]:
    """把三个不变量算成一行可写进 metrics.jsonl 的记录。"""
    return {
        "I1_label_align_violations": label_alignment_violations(input_ids, labels),
        "I2_optimizer_step_ratio": (float(applied_steps) / max(int(total_steps), 1)),
        "I2_param_delta_norm": float(param_delta_norm),
        "I3_cross_rank_grad_delta": grad_delta,
    }


def classify(obs: Dict[str, Any], step_ratio_floor: float = 0.5,
             grad_delta_tol: float = 1e-6,
             param_delta_floor: float = 0.0) -> Dict[str, Any]:
    """按三行判别表反推是哪一档。返回结论 + 触发它的那条不变量。

    多条同时触发时按 A -> B -> C 的顺序报告，并把全部命中列出来——
    真实排查里同时踩两个坑并不罕见，只报一个会误导。
    """
    hits: List[str] = []
    if int(obs.get("I1_label_align_violations", 0)) > 0:
        hits.append("label_shift")
    ratio = float(obs.get("I2_optimizer_step_ratio", 1.0))
    delta_norm = float(obs.get("I2_param_delta_norm", 1.0))
    if ratio < step_ratio_floor or delta_norm <= param_delta_floor:
        hits.append("scaler_stuck")
    gd = obs.get("I3_cross_rank_grad_delta")
    if gd is not None and float(gd) > grad_delta_tol:
        hits.append("ddp_no_sync")
    verdict = hits[0] if hits else "none"
    reason = {
        "label_shift": ("I1=" + str(obs.get("I1_label_align_violations"))
                        + " > 0：labels 与 input_ids 在非 -100 处对不上"),
        "scaler_stuck": ("I2 step_ratio=" + ("%.3f" % ratio)
                         + "，param_delta_norm=" + ("%.3e" % delta_norm)),
        "ddp_no_sync": ("I3 cross_rank_grad_delta=" + str(gd) + " > "
                        + str(grad_delta_tol)),
        "none": "三个不变量都在正常范围内",
    }[verdict]
    return {
        "verdict": verdict,
        "all_hits": hits,
        "reason": reason,
        "observations": dict(obs),
        "thresholds": {
            "step_ratio_floor": step_ratio_floor,
            "grad_delta_tol": grad_delta_tol,
            "param_delta_floor": param_delta_floor,
        },
    }


DECISION_TABLE_HEADER = ("mode", "I1>0", "I2 低", "I3>0")
DECISION_TABLE_ROWS = (
    ("none", "否", "否", "否"),
    ("label_shift", "是", "否", "否"),
    ("scaler_stuck", "否", "是", "否"),
    ("ddp_no_sync", "否", "否", "是"),
)


def render_decision_table() -> str:
    """打印那张三行判别表。Day 2 要求手写一份，这个只用来对答案。"""
    lines = ["%-14s %-6s %-6s %-6s" % DECISION_TABLE_HEADER,
             "-" * 36]
    for row in DECISION_TABLE_ROWS:
        lines.append("%-14s %-6s %-6s %-6s" % row)
    lines.append("")
    lines.append("必要不充分：loss 在下降 —— 四种模式下都成立。")
    lines.append("充分：任一不变量单独越界即可定位到唯一一档。")
    return "\n".join(lines)


def describe(mode: str) -> str:
    if mode not in FAULT_DESCRIPTIONS:
        raise ValueError("未知 mode=" + repr(mode))
    return FAULT_DESCRIPTIONS[mode]
