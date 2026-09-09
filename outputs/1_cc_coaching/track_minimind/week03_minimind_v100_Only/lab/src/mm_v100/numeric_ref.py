"""Day 1 误差账：fp32 参考曲线 vs fp16 AMP 曲线，逐步对照。

来源
----
从 week01 lab/src/mm_probe/bounded_train.py 的 dtype/scaler 部分复制并裁剪，
去掉了 MiniMind 依赖与 checkpoint 逻辑，只留「同一初始权重、同一批数据、
只改精度」这一件事。

这份对照要回答的问题
--------------------
fp16 相对 fp32 从哪一步开始偏、偏多少、偏的方向。四个输出字段：

max_dloss           两条曲线上 |loss_fp16 - loss_fp32| 的最大值。
first_diverge_step  第一次 |Δ| 超过 --tol 的 step；一直没超就是 None。
                    这个数比 max_dloss 更有诊断价值：偏差从第 1 步就出现，
                    多半是前向数值问题（RMSNorm / softmax 溢出）；
                    到第 5 步才出现，多半是优化器状态在低精度梯度下慢慢跑偏。
scale_trajectory    GradScaler 的 scale 序列。健康形态是「起点 65536，
                    前几步可能连掉几次，然后长期不变」。持续下降到个位数
                    = 梯度里一直有 inf，那是 Day 2 故障 B。
skip_steps          被 scaler 跳掉的 step 数。>0 不一定是病（开头几步很常见），
                    但 skip_steps/steps 高于 1/3 时这条 loss 曲线不能用来比较。

方法上的三个约束（缺一个结论就不成立）
--------------------------------------
1. 两条曲线的初始权重必须逐位相同 —— 用 deepcopy，不是同 seed 重建。
   同 seed 重建在 CUDA 上不保证逐位一致（cuDNN 算法选择、kernel 顺序）。
2. 两条曲线第 s 步消费的样本必须相同 —— 用 common.micro_batch_indices。
3. loss 都在 fp32 里累加 —— 否则比的是「求和的精度差」而不是「训练的精度差」。
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List

import torch

from . import common as C
from . import model as M


def run_curve(model: torch.nn.Module, dataset, dtype_name: str, device: str,
              steps: int, global_batch: int, accum: int, base_lr: float,
              grad_clip: float, sdpa_backend: str = "auto",
              force_pure_python_scaler: bool = False) -> Dict[str, Any]:
    """跑 steps 步，返回逐步 loss / grad_norm / scale / skipped。

    模型是原地训练的：调用方要保证传进来的是自己那一份 copy。
    """
    torch_dtype = C.assert_dtype_supported(dtype_name, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scaler = C.ScalerAdapter(torch_dtype, device,
                             force_pure_python=force_pure_python_scaler)
    model.train()
    losses: List[float] = []
    grad_norms: List[float] = []
    scales: List[float] = []
    skipped: List[bool] = []
    for step in range(1, int(steps) + 1):
        lr = C.cosine_lr(step, steps, base_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), dtype=torch.float32, device=device)
        for acc_idx in range(accum):
            idx = C.micro_batch_indices(step, global_batch, 1, accum, 0, acc_idx)
            input_ids, labels = C.collate_indices(dataset, idx, device)
            with C.sdpa_context(sdpa_backend, device):
                with C.autocast_context(torch_dtype, device):
                    out = model(input_ids, labels=labels)
                    loss = out.loss
            scaler.scale(loss / accum).backward()
            total = total + loss.detach().float()
        scaler.unscale_(optimizer)
        gnorm = C.grad_global_norm(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        applied = scaler.step_and_update(optimizer)
        losses.append(float(total.item() / accum))
        grad_norms.append(gnorm)
        scales.append(scaler.get_scale())
        skipped.append(not applied)
    return {
        "dtype": dtype_name,
        "losses": losses,
        "grad_norms": grad_norms,
        "scale_trajectory": scales,
        "skipped": skipped,
        "skip_steps": int(sum(1 for s in skipped if s)),
        "scaler": scaler.describe(),
    }


def compare_curves(ref: Dict[str, Any], low: Dict[str, Any],
                   tol: float) -> Dict[str, Any]:
    """把两条曲线折成 Day 1 的四个证据字段。"""
    n = min(len(ref["losses"]), len(low["losses"]))
    deltas = [abs(low["losses"][i] - ref["losses"][i]) for i in range(n)]
    signed = [low["losses"][i] - ref["losses"][i] for i in range(n)]
    rel = [deltas[i] / max(abs(ref["losses"][i]), 1e-12) for i in range(n)]
    first = None
    for i, d in enumerate(deltas):
        if d > tol:
            first = i + 1
            break
    finite_ref = all(math.isfinite(v) for v in ref["losses"][:n])
    finite_low = all(math.isfinite(v) for v in low["losses"][:n])
    return {
        "steps_compared": n,
        "tol": float(tol),
        "max_dloss": max(deltas) if deltas else 0.0,
        "max_rel_dloss": max(rel) if rel else 0.0,
        "mean_dloss": (sum(deltas) / n) if n else 0.0,
        "first_diverge_step": first,
        "final_delta_sign": (0 if not signed else
                             (1 if signed[-1] > 0 else (-1 if signed[-1] < 0 else 0))),
        "ref_all_finite": finite_ref,
        "low_all_finite": finite_low,
        "skip_steps": low["skip_steps"],
        "skip_ratio": (low["skip_steps"] / n) if n else 0.0,
        "scale_first": low["scale_trajectory"][0] if low["scale_trajectory"] else None,
        "scale_last": low["scale_trajectory"][-1] if low["scale_trajectory"] else None,
        "scale_monotone_down": bool(
            len(low["scale_trajectory"]) > 1
            and all(low["scale_trajectory"][i + 1] <= low["scale_trajectory"][i]
                    for i in range(len(low["scale_trajectory"]) - 1))
            and low["scale_trajectory"][-1] < low["scale_trajectory"][0]),
        "deltas": deltas,
    }


def verdict(cmp: Dict[str, Any], max_skip_ratio: float = 0.34) -> Dict[str, Any]:
    """给一个可以直接写进证据的 PASS/FAIL 判定，并说明失败时先查什么。"""
    problems: List[str] = []
    if not cmp["ref_all_finite"]:
        problems.append("fp32 参考曲线里出现了 nan/inf。首个检查：lr 是不是太大，"
                        "或者数据里有越界 token id。fp32 都不稳，fp16 没有讨论价值。")
    if not cmp["low_all_finite"]:
        problems.append("fp16 曲线里出现了 nan/inf。首个检查：RMSNorm 有没有在 "
                        "float32 里算平方和；再看 scale_trajectory 是不是一路下降。")
    if cmp["skip_ratio"] > max_skip_ratio:
        problems.append("skip_ratio=" + ("%.2f" % cmp["skip_ratio"])
                        + " 超过阈值，说明大部分 step 的梯度里有 inf，"
                          "这条曲线不能与 fp32 比较。首个检查：grad_norm 的量级。")
    if cmp["scale_monotone_down"]:
        problems.append("scale 单调下降到底，是 Day 2 故障 B 的形态。"
                        "首个检查：是不是某个算子在 autocast 里溢出了。")
    return {"pass": not problems, "problems": problems}


def run(config: Dict[str, Any], device: str = "cpu", steps: int = 6,
        tol: float = 1e-3, seed: int = 42, sdpa_backend: str = "auto",
        low_dtype: str = "float16",
        force_pure_python_scaler: bool = False) -> Dict[str, Any]:
    """完整的 Day 1 误差账。返回可直接 json.dump 的字典。"""
    C.set_seed(seed, per_rank=False)
    train_cfg = config["train"]
    global_batch = int(train_cfg["global_batch"])
    accum = int(train_cfg.get("accum", 1))
    seq_len = int(train_cfg["seq_len"])
    base_lr = float(train_cfg.get("lr", 5e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    model_ref, model_cfg = M.build_model(config["model"], device=device, seed=seed)
    # deepcopy 而不是同 seed 重建：CUDA 上重建不保证逐位一致。
    model_low = copy.deepcopy(model_ref)

    dataset = C.TinyDataset(
        num_samples=max(int(train_cfg.get("num_samples", 512)),
                        steps * global_batch),
        seq_len=seq_len, vocab_size=int(model_cfg.vocab_size),
        seed=int(train_cfg.get("data_seed", 1234)),
        label_ignore_prefix=int(train_cfg.get("label_ignore_prefix", 0)))

    ref = run_curve(model_ref, dataset, "float32", device, steps, global_batch,
                    accum, base_lr, grad_clip, sdpa_backend)
    low = run_curve(model_low, dataset, low_dtype, device, steps, global_batch,
                    accum, base_lr, grad_clip, sdpa_backend,
                    force_pure_python_scaler=force_pure_python_scaler)
    cmp = compare_curves(ref, low, tol)
    return {
        "schema": "mm_v100.numeric_ref/1",
        "config": config.get("name"),
        "device": device,
        "low_dtype": low_dtype,
        "steps": int(steps),
        "global_batch": global_batch,
        "accum": accum,
        "seq_len": seq_len,
        "params": M.count_params(model_ref),
        "fp32": ref,
        "low": low,
        "compare": cmp,
        "verdict": verdict(cmp),
        "env": C.env_summary(),
    }


def render_text(report: Dict[str, Any]) -> str:
    cmp = report["compare"]
    lines = ["== Day 1 误差账 fp32 vs " + report["low_dtype"] + " =="]
    lines.append("step |     fp32 |     %s |    |delta|" % report["low_dtype"][:6])
    lines.append("-----+----------+-----------+-----------")
    for i, d in enumerate(cmp["deltas"]):
        lines.append("%4d | %8.5f | %9.5f | %.3e"
                     % (i + 1, report["fp32"]["losses"][i],
                        report["low"]["losses"][i], d))
    lines.append("")
    lines.append("max_dloss=%.3e  max_rel=%.3e  first_diverge_step=%s  tol=%.1e"
                 % (cmp["max_dloss"], cmp["max_rel_dloss"],
                    cmp["first_diverge_step"], cmp["tol"]))
    lines.append("scale: first=%s last=%s  skip_steps=%d  skip_ratio=%.2f"
                 % (cmp["scale_first"], cmp["scale_last"],
                    cmp["skip_steps"], cmp["skip_ratio"]))
    lines.append("verdict=" + ("PASS" if report["verdict"]["pass"] else "FAIL"))
    for p in report["verdict"]["problems"]:
        lines.append("  - " + p)
    return "\n".join(lines)
