"""router 负载统计、热点专家注入、bias / aux_coef 扫描。

router 在 MiniMind 里的位置
---------------------------
``MOEFeedForward.gate = nn.Linear(hidden_size, num_experts, bias=False)``。
没有 ``MoEGate`` 类，也没有可学习的 router bias——所以"制造热点专家"只能从外部
给某个专家的 logit 加一个常数。本文件用 forward hook 做这件事：
``inject_router_bias(model, expert_id=2, bias=2.0)`` 会在每个 MoE 层的 gate 输出上
给第 2 列加 2.0，softmax 之后该专家被选中的概率显著上升。

统计什么
--------
* 每个专家收到的 token 数 ``expert_counts[e]``（对所有 MoE 层求和，也分层记录）；
* 负载比 ``f_e = expert_counts[e] / sum(expert_counts)``；
* ``max_load_ratio = max(f_e) * E``（均衡时 ≈1，全部坍缩到一个专家时 = E）；
* ``mean_load_ratio``（恒为 1，用来自检统计口径没写错）；
* router 熵 ``H = -sum(p log p)``，均衡时趋近 ``log E``；
* 每 rank 的计算时间与等待时间：``compute_ms`` 是本 rank 的 forward+backward，
  ``wait_ms`` 是紧随其后的一次 ``dist.barrier()`` 的耗时——木桶效应下，
  最慢的 rank 的 ``wait_ms`` 接近 0，其余 rank 的 ``wait_ms`` 就是它们被拖住的时间。

aux loss
--------
与 MiniMind L169–171 完全一致：``(load * scores.mean(0)).sum() * E * coef``，
其中 ``load = one_hot(topk_idx, E).float().mean(0)``。实现复用
:func:`mm_dist.ep_moe.aux_loss_value`，避免两处公式漂移。

子命令
------
``stats``  跑一段短训练，把每步的 router 统计写成 JSONL。
``sweep``  按 ``--bias-list`` × ``--aux-coef-list`` 逐组合跑短训练，输出一张 CSV。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_dist import common as C  # noqa: E402
from mm_dist.ep_moe import aux_loss_value  # noqa: E402

CSV_COLUMNS = [
    "bias", "aux_coef", "expert_id", "steps", "num_experts", "top_k",
    "final_loss", "mean_logits_loss", "mean_aux_loss",
    "max_load_ratio", "min_load_ratio", "mean_load_ratio", "router_entropy",
    "expert_fractions", "mean_step_time_s", "mean_compute_ms", "mean_wait_ms",
    "peak_mem_mb", "world_size", "dtype",
]


# ---------------------------------------------------------------------------
# 找到所有 MoE 层
# ---------------------------------------------------------------------------


def find_moe_layers(model, moe_cls) -> List[Tuple[str, Any]]:
    """返回 ``[(名字, MOEFeedForward 模块), ...]``；dense 模型返回空列表。"""
    return [(name, mod) for name, mod in model.named_modules()
            if isinstance(mod, moe_cls)]


# ---------------------------------------------------------------------------
# 热点专家注入
# ---------------------------------------------------------------------------


class RouterBiasHandle:
    """``inject_router_bias`` 的返回值；``remove()`` 撤销全部 hook。"""

    def __init__(self, handles: List[Any], expert_id: int, bias: float,
                 layers: List[str]) -> None:
        self.handles = handles
        self.expert_id = int(expert_id)
        self.bias = float(bias)
        self.layers = list(layers)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self) -> "RouterBiasHandle":
        return self

    def __exit__(self, *exc) -> bool:
        self.remove()
        return False


def _register_prepend(module, hook):
    """尽量把 hook 排到最前面（torch>=1.13 的 prepend 参数）。"""
    try:
        return module.register_forward_hook(hook, prepend=True)
    except TypeError:
        return module.register_forward_hook(hook)


def inject_router_bias(model, expert_id: int, bias: float,
                       moe_cls=None) -> RouterBiasHandle:
    """给每个 MoE 层的 router logits 的第 ``expert_id`` 列加常数 ``bias``。

    ``bias=0`` 时仍然注册 hook（加 0），这样"有偏置"和"无偏置"两条路径的
    数值噪声来源一致，扫描结果可比。

    为什么加在 logit 而不是 softmax 之后：softmax 后再加会破坏归一化，
    ``topk_weight`` 的和不再是 1，与 MiniMind 的 ``norm_topk_prob`` 语义冲突。
    """
    if moe_cls is None:
        mm = C.import_minimind(None)
        moe_cls = mm.MOEFeedForward
    layers = find_moe_layers(model, moe_cls)
    if not layers:
        raise ValueError(
            "模型里没有 MOEFeedForward 层：config 的 model.use_moe 必须为 true。")
    handles: List[Any] = []
    names: List[str] = []

    def make_hook(num_experts: int):
        def hook(module, inputs, output):
            if int(expert_id) < 0 or int(expert_id) >= num_experts:
                raise ValueError(
                    "expert_id=" + str(expert_id) + " 超出 [0, "
                    + str(num_experts) + ")")
            add = torch.zeros(num_experts, device=output.device,
                              dtype=output.dtype)
            add[int(expert_id)] = float(bias)
            return output + add
        return hook

    for name, mod in layers:
        handles.append(_register_prepend(mod.gate,
                                         make_hook(int(mod.config.num_experts))))
        names.append(name)
    return RouterBiasHandle(handles, expert_id, bias, names)


# ---------------------------------------------------------------------------
# 统计收集
# ---------------------------------------------------------------------------


class RouterStatsCollector:
    """在每个 MoE 层的 gate 上挂 hook，累计 per-expert token 计数与 router 熵。"""

    def __init__(self, model, moe_cls=None) -> None:
        if moe_cls is None:
            mm = C.import_minimind(None)
            moe_cls = mm.MOEFeedForward
        self.layers = find_moe_layers(model, moe_cls)
        if not self.layers:
            raise ValueError("模型里没有 MOEFeedForward 层（model.use_moe 必须为 true）")
        cfg = self.layers[0][1].config
        self.num_experts = int(cfg.num_experts)
        self.top_k = int(cfg.num_experts_per_tok)
        self.aux_coef = float(getattr(cfg, "router_aux_loss_coef", 0.0))
        self.counts = torch.zeros(self.num_experts, dtype=torch.float64)
        self.per_layer: Dict[str, torch.Tensor] = {
            n: torch.zeros(self.num_experts, dtype=torch.float64)
            for n, _ in self.layers}
        self.entropy_sum = 0.0
        self.entropy_batches = 0
        self.aux_sum = 0.0
        self._handles: List[Any] = []

    def _hook(self, name: str):
        def hook(module, inputs, output):
            with torch.no_grad():
                scores = F.softmax(output.float(), dim=-1)
                _, idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
                flat = idx.reshape(-1)
                cnt = torch.bincount(flat, minlength=self.num_experts
                                     ).to(torch.float64).cpu()
                self.counts += cnt
                self.per_layer[name] += cnt
                probs = scores.mean(dim=0).clamp_min(1e-12)
                self.entropy_sum += float(-(probs * probs.log()).sum())
                self.entropy_batches += 1
                self.aux_sum += float(aux_loss_value(scores, idx,
                                                     self.num_experts, 1.0))
            return None
        return hook

    def attach(self) -> "RouterStatsCollector":
        for name, mod in self.layers:
            self._handles.append(mod.gate.register_forward_hook(self._hook(name)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self) -> None:
        self.counts.zero_()
        for v in self.per_layer.values():
            v.zero_()
        self.entropy_sum = 0.0
        self.entropy_batches = 0
        self.aux_sum = 0.0

    def summary(self) -> Dict[str, Any]:
        total = float(self.counts.sum())
        if total <= 0:
            fractions = [0.0] * self.num_experts
        else:
            fractions = [float(v) / total for v in self.counts.tolist()]
        e = self.num_experts
        batches = max(self.entropy_batches, 1)
        return {
            "num_experts": e,
            "top_k": self.top_k,
            "expert_counts": [int(v) for v in self.counts.tolist()],
            "expert_fractions": [round(f, 6) for f in fractions],
            "max_load_ratio": round(max(fractions) * e, 6) if fractions else 0.0,
            "min_load_ratio": round(min(fractions) * e, 6) if fractions else 0.0,
            "mean_load_ratio": round(sum(fractions) / e * e, 6) if fractions else 0.0,
            "router_entropy": round(self.entropy_sum / batches, 6),
            "router_entropy_max": round(math.log(e), 6),
            "aux_loss_unit_coef": round(self.aux_sum / batches, 6),
            "per_layer_fractions": {
                n: [round(float(v) / max(float(t.sum()), 1e-12), 6)
                    for v in t.tolist()]
                for n, t in self.per_layer.items()},
        }

    def __enter__(self) -> "RouterStatsCollector":
        return self.attach()

    def __exit__(self, *exc) -> bool:
        self.detach()
        return False


# ---------------------------------------------------------------------------
# 短训练
# ---------------------------------------------------------------------------


def run_short_training(args: argparse.Namespace, bias: float, aux_coef: float,
                       info: C.DistInfo, device: str,
                       jsonl: Optional[C.JsonlLogger] = None) -> Dict[str, Any]:
    """跑 ``--max-steps`` 步，返回 router 统计 + 时间 + loss 汇总。

    每次调用都重新建模型：扫描的自变量是 (bias, aux_coef)，模型初始权重必须相同，
    否则比出来的是初始化差异。``C.set_seed(args.seed, per_rank=False)`` 保证这一点。
    """
    C.set_seed(args.seed, per_rank=False)
    model, lm_config, cfg, mm = C.build_model(
        args.config, minimind_root=args.minimind_root, device=device,
        override={"router_aux_loss_coef": float(aux_coef)})
    if not bool(lm_config.use_moe):
        raise SystemExit("router_stats 需要 MoE 配置（model.use_moe=true）")

    train_cfg = cfg["train"]
    seq_len = int(args.seq_len or train_cfg["seq_len"])
    global_batch = int(args.global_batch or train_cfg["global_batch"])
    accum = int(args.accum or train_cfg.get("accum", 1))
    base_lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 5e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    denom = info.world_size * accum
    if global_batch % denom != 0:
        raise SystemExit("global_batch 必须能被 world_size*accum 整除")

    dataset = C.build_dataset(
        "tiny", seq_len=seq_len, vocab_size=int(lm_config.vocab_size),
        minimind_root=args.minimind_root, num_samples=int(args.num_samples),
        seed=int(train_cfg.get("data_seed", 1234)))

    torch_dtype = C.assert_dtype_supported(args.dtype, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scaler = C.make_grad_scaler(torch_dtype, device)
    C.reset_peak_mem(device)

    bias_handle = inject_router_bias(model, int(args.expert_id), float(bias),
                                     moe_cls=mm.MOEFeedForward)
    collector = RouterStatsCollector(model, moe_cls=mm.MOEFeedForward)
    collector.attach()
    model.train()

    steps = max(int(args.max_steps), 1)
    logits_losses: List[float] = []
    aux_losses: List[float] = []
    step_times: List[float] = []
    compute_ms: List[float] = []
    wait_ms: List[float] = []
    final_loss = 0.0
    for step in range(1, steps + 1):
        t_step = time.perf_counter()
        lr = C.cosine_lr(step, steps, base_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        sums = torch.zeros(2, device=device, dtype=torch.float32)
        t_compute = time.perf_counter()
        for acc_idx in range(accum):
            idx = C.micro_batch_indices(step, global_batch, info.world_size,
                                        accum, info.rank, acc_idx)
            input_ids, labels = C.collate_indices(dataset, idx, device)
            with C.autocast_context(torch_dtype, device):
                res = model(input_ids, labels=labels)
                aux = res.aux_loss if res.aux_loss is not None else res.loss.new_zeros(())
                loss = res.loss + aux
            scaler.scale(loss / accum).backward()
            sums[0] += res.loss.detach().float()
            sums[1] += aux.detach().float()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        compute_ms.append((time.perf_counter() - t_compute) * 1000.0)

        t_wait = time.perf_counter()
        if info.world_size > 1:
            dist.barrier()
        wait_ms.append((time.perf_counter() - t_wait) * 1000.0)

        if info.world_size > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        sums = sums / float(denom)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        logits_losses.append(float(sums[0]))
        aux_losses.append(float(sums[1]))
        final_loss = float(sums[0]) + float(sums[1])
        step_times.append(time.perf_counter() - t_step)
        if jsonl is not None:
            rec = {"step": step, "bias": bias, "aux_coef": aux_coef,
                   "loss": final_loss, "logits_loss": float(sums[0]),
                   "aux_loss": float(sums[1]), "lr": lr,
                   "step_time_s": step_times[-1],
                   "compute_ms": compute_ms[-1], "wait_ms": wait_ms[-1],
                   "peak_mem_mb": C.peak_mem_mb(device)}
            rec.update(collector.summary())
            jsonl.write(rec)

    summary = collector.summary()
    collector.detach()
    bias_handle.remove()
    summary.update({
        "bias": bias,
        "aux_coef": aux_coef,
        "expert_id": int(args.expert_id),
        "steps": steps,
        "final_loss": final_loss,
        "mean_logits_loss": sum(logits_losses) / len(logits_losses),
        "mean_aux_loss": sum(aux_losses) / len(aux_losses),
        "mean_step_time_s": sum(step_times) / len(step_times),
        "mean_compute_ms": sum(compute_ms) / len(compute_ms),
        "mean_wait_ms": sum(wait_ms) / len(wait_ms),
        "peak_mem_mb": C.peak_mem_mb(device),
        "world_size": info.world_size,
        "dtype": args.dtype,
    })
    return summary


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def cmd_stats(args: argparse.Namespace) -> int:
    info = C.init_dist(backend=args.backend, timeout_s=args.nccl_timeout_s,
                       force_cpu=args.force_cpu)
    device = "cpu" if args.force_cpu else info.device
    logger = C.JsonlLogger(args.out_dir, args.log_name or "router_stats",
                           rank=info.rank)
    logger.write({"step": 0, "event": "header", "runner": "router_stats",
                  "bias": args.bias, "expert_id": args.expert_id,
                  "aux_coef": args.aux_coef, "world_size": info.world_size,
                  "dtype": args.dtype, "env": C.env_summary()})
    summary = run_short_training(args, float(args.bias), float(args.aux_coef),
                                 info, device, jsonl=logger)
    logger.close()
    if info.is_main:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
              flush=True)
        print("[router] max_load_ratio=%.3f entropy=%.4f/%.4f mean_wait_ms=%.2f"
              % (summary["max_load_ratio"], summary["router_entropy"],
                 summary["router_entropy_max"], summary["mean_wait_ms"]), flush=True)
    C.cleanup_dist(info)
    return 0


def _parse_float_list(text: str) -> List[float]:
    values = [t.strip() for t in str(text).split(",") if t.strip()]
    if not values:
        raise ValueError("列表不能为空，例如 --bias-list 0,0.5,1,2,4")
    return [float(v) for v in values]


def cmd_sweep(args: argparse.Namespace) -> int:
    info = C.init_dist(backend=args.backend, timeout_s=args.nccl_timeout_s,
                       force_cpu=args.force_cpu)
    device = "cpu" if args.force_cpu else info.device
    biases = _parse_float_list(args.bias_list)
    coefs = _parse_float_list(args.aux_coef_list)
    combos = list(itertools.product(biases, coefs))
    C.log_main("[sweep] %d 组合 x %d 步（bias=%s, aux_coef=%s）"
               % (len(combos), int(args.max_steps), biases, coefs))

    rows: List[Dict[str, Any]] = []
    logger = C.JsonlLogger(args.out_dir, args.log_name or "moe_sweep", rank=info.rank)
    for bias, coef in combos:
        t0 = time.perf_counter()
        summary = run_short_training(args, bias, coef, info, device, jsonl=None)
        summary["wall_s"] = time.perf_counter() - t0
        rows.append(summary)
        logger.write(dict(summary, step=len(rows), event="combo"))
        C.log_main("[sweep] bias=%.3g aux_coef=%.3g -> max_load_ratio=%.3f "
                   "logits_loss=%.4f aux_loss=%.6f entropy=%.4f wait_ms=%.2f"
                   % (bias, coef, summary["max_load_ratio"],
                      summary["mean_logits_loss"], summary["mean_aux_loss"],
                      summary["router_entropy"], summary["mean_wait_ms"]))
    logger.close()

    csv_path = args.csv or os.path.join(args.out_dir, "moe_sweep.csv")
    if info.is_main:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["expert_fractions"] = "|".join(
                    "%.4f" % v for v in row["expert_fractions"])
                writer.writerow(out)
        print("[sweep] CSV -> " + os.path.abspath(csv_path), flush=True)
    C.cleanup_dist(info)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MoE router 负载统计 / 热点专家注入 / bias × aux_coef 扫描",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    C.add_common_args(p)
    p.add_argument("command", choices=["stats", "sweep"],
                   help="stats=单组合短训练并写 JSONL；sweep=多组合并写 CSV")
    p.add_argument("--expert-id", type=int, default=0,
                   help="被加 bias 的专家 id（制造热点）")
    p.add_argument("--bias", type=float, default=0.0, help="stats 用的 router bias")
    p.add_argument("--aux-coef", type=float, default=0.0005,
                   help="stats 用的 router_aux_loss_coef")
    p.add_argument("--bias-list", type=str, default="0,0.5,1,2,4",
                   help="sweep 的 bias 取值，逗号分隔")
    p.add_argument("--aux-coef-list", type=str, default="0,0.001,0.01,0.1",
                   help="sweep 的 aux 系数取值，逗号分隔")
    p.add_argument("--csv", type=str, default=None,
                   help="sweep 的 CSV 输出路径，默认 <out-dir>/moe_sweep.csv")
    p.add_argument("--global-batch", type=int, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-samples", type=int, default=8192)
    p.add_argument("--log-name", type=str, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config:
        raise SystemExit("必须给 --config configs/moe_tiny_cpu.json（或 v100_moe.json）")
    return cmd_stats(args) if args.command == "stats" else cmd_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
