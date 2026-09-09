#!/usr/bin/env python
"""跑量化实验 Q1–Q6，结果写成 JSON + JSONL。

设计原则（CLAUDE.md 3.2）：**全部逻辑在 Python 里**。
run_quant_suite.sh 只做一件事——用对解释器调一次这个脚本。
重试、循环、编码、目录创建、磁盘检查都在这里，不在 shell 里。

实验对照表
----------
  q1  逐层 x 三种粒度的相对误差与存储账（含逐通道最坏值）
  q2  截断阈值 alpha 扫描 + MSE 的 clipping/rounding 分解（8 bit 与 4 bit 各一遍）
  q3  block-wise 8-bit Adam：三种映射对照 + 收敛对照 + 字节比
  q4  W8A16 weight-only PTQ：字节比 + 逐 token top-1 一致率
  q5  QAT：自写 STE vs torch.ao 两条路径的前向/反向一致性 + 几步 AMP 训练
  q6  激活 outlier 统计、跨 batch 重合率、幅度迁移 alpha 扫描

权重从哪来
----------
默认用**合成权重**（构造成有明显的逐通道尺度差异和行内 outlier），
这样在没有 GPU、没有 checkpoint 的机器上也能把整条链跑通。
真结论要用真权重：--state-dict <相对 MM_WEIGHTS_ROOT 的路径>，
脚本只读取张量的形状与数值统计，不打印任何权重内容。

用法
----
  python lab/scripts/run_quant_suite.py --help
  python lab/scripts/run_quant_suite.py --config lab/configs/quant_cpu_smoke.json --dry-run
  python lab/scripts/run_quant_suite.py --config lab/configs/quant_cpu_smoke.json
  python lab/scripts/run_quant_suite.py --config lab/configs/quant_v100.json --device cuda \
      --experiments q3,q5 --state-dict minimind/full_sft_768.pth
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mm_v100 import paths  # noqa: E402
from mm_v100.console import use_utf8  # noqa: E402

from mm_quant import error_budget, opt8bit, qat, quantizers, smooth, weight_only  # noqa: E402

EXPERIMENTS: Tuple[str, ...] = ("q1", "q2", "q3", "q4", "q5", "q6")
DEFAULT_CONFIG = "lab/configs/quant_cpu_smoke.json"


# --------------------------------------------------------------------------- #
# 数据来源
# --------------------------------------------------------------------------- #
def synthetic_weights(cfg: Dict[str, Any], device: str, seed: int) -> List[Tuple[str, torch.Tensor]]:
    """造一组「像真权重」的张量：逐输出通道尺度不同 + 行内稀疏 outlier。

    两种结构分别对应 per-channel 和 per-group 各自能救的那类误差，
    所以 Q1 的三条曲线才会明显分开。用纯 randn 是看不出差别的。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    out: List[Tuple[str, torch.Tensor]] = []
    for spec in cfg["layers"]:
        name = spec["name"]
        rows, cols = int(spec["out_features"]), int(spec["in_features"])
        decades = float(spec.get("channel_scale_decades", 2.0))
        n_out = int(spec.get("outliers_per_row", 4))
        mag = float(spec.get("outlier_magnitude", 20.0))
        w = torch.randn((rows, cols), generator=g)
        w = w * torch.logspace(0.0, decades, rows).reshape(-1, 1)
        if n_out > 0 and cols > n_out:
            idx = torch.randint(0, cols, (rows, n_out), generator=g)
            w.scatter_(1, idx, torch.randn((rows, n_out), generator=g) * mag)
        out.append((name, w.to(device)))
    return out


def load_state_dict_weights(
    rel_path: str, device: str, min_numel: int, max_layers: int
) -> List[Tuple[str, torch.Tensor]]:
    """从 MM_WEIGHTS_ROOT 下读一个 state_dict，取二维权重张量。

    只读形状与数值，不打印内容。torch.load 用 map_location='cpu' 以免
    一个 checkpoint 直接把显存塞满。
    """
    root = paths.weights_root()
    p = (root / rel_path).resolve()
    if not p.is_file():
        raise paths.PathContractError(
            "找不到权重文件 {}\n"
            "下一步：确认 MM_WEIGHTS_ROOT（当前 {}）下有这个相对路径；"
            "权重要在外网机器下好再拷进来，见 weights/README.md。".format(p, root)
        )
    sd = torch.load(str(p), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    picked: List[Tuple[str, torch.Tensor]] = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.dim() != 2 or v.numel() < min_numel:
            continue
        picked.append((k, v.to(device=device, dtype=torch.float32)))
        if len(picked) >= max_layers:
            break
    if not picked:
        raise ValueError(
            "{} 里没有满足 dim==2 且 numel>={} 的张量。"
            "换一个 --min-numel，或确认这是模型权重不是优化器状态。".format(p, min_numel)
        )
    return picked


def make_mlp(cfg: Dict[str, Any], device: str, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    d_in = int(cfg["in_features"])
    d_hidden = int(cfg["hidden_features"])
    d_out = int(cfg["out_features"])
    model = nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.SiLU(),
        nn.Linear(d_hidden, d_hidden),
        nn.SiLU(),
        nn.Linear(d_hidden, d_out),
    )
    return model.to(device)


# --------------------------------------------------------------------------- #
# Q1 / Q2
# --------------------------------------------------------------------------- #
def run_q1(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    weights = ctx["weights"]
    rows = error_budget.error_table(
        weights,
        granularities=tuple(cfg.get("granularities", error_budget.DEFAULT_GRANULARITIES)),
        scheme=cfg.get("scheme", "symmetric"),
        axis=int(cfg.get("axis", 0)),
        group_size=int(cfg.get("group_size", 128)),
        num_bits=int(cfg.get("num_bits", 8)),
        reference_dtype=torch.float16,
    )
    monotone = True
    for r in rows:
        g = r["granularity"]
        if all(k in g for k in ("per_tensor", "per_channel", "per_group")):
            monotone = monotone and (
                g["per_tensor"]["rel_err"] >= g["per_channel"]["rel_err"] >= g["per_group"]["rel_err"]
            )
    return {
        "table": rows,
        "granularity_monotone": monotone,
        "text": error_budget.format_error_table(rows),
    }


def run_q2(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    name, w = ctx["weights"][0]
    alphas = tuple(cfg.get("alphas", error_budget.DEFAULT_ALPHAS))
    out: Dict[str, Any] = {"layer": name, "by_bits": {}}
    for nb in cfg.get("num_bits_list", [8, 4]):
        sweep = error_budget.clipping_sweep(
            w,
            alphas=alphas,
            granularity=cfg.get("granularity", "per_channel"),
            axis=int(cfg.get("axis", 0)),
            group_size=int(cfg.get("group_size", 128)),
            num_bits=int(nb),
        )
        out["by_bits"][str(nb)] = {
            "sweep": sweep,
            "best": error_budget.best_alpha(sweep),
            "text": error_budget.format_alpha_sweep(sweep, "{} bit".format(nb)),
        }
    return out


# --------------------------------------------------------------------------- #
# Q3
# --------------------------------------------------------------------------- #
def _train_lstsq(
    optimizer_factory: Callable[[List[torch.nn.Parameter]], torch.optim.Optimizer],
    steps: int,
    device: str,
    seed: int,
    d: int,
) -> Tuple[List[float], torch.optim.Optimizer]:
    torch.manual_seed(seed)
    lin = nn.Linear(d, d).to(device)
    target = torch.randn(4 * d, d, device=device)
    inputs = torch.randn(4 * d, d, device=device)
    opt = optimizer_factory(list(lin.parameters()))
    losses: List[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((lin(inputs) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses, opt


def run_q3(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    device = ctx["device"]
    seed = ctx["seed"]
    block = int(cfg.get("block_size", opt8bit.DEFAULT_BLOCK_SIZE))
    power = float(cfg.get("sq_power", opt8bit.DEFAULT_SQ_POWER))
    steps = int(cfg.get("steps", 40))
    d = int(cfg.get("dim", 64))

    # (a) 三种映射在一个长尾 exp_avg_sq 上的对照
    g = torch.Generator(device="cpu").manual_seed(seed)
    decades = float(cfg.get("state_decades", 9.0))
    nblocks = int(cfg.get("state_blocks", 4))
    parts = [
        torch.rand(block, generator=g) * (10.0 ** (-decades * i / max(nblocks - 1, 1)))
        for i in range(nblocks)
    ]
    v = torch.cat(parts).to(device)
    mapping_cmp = opt8bit.compare_state_quantization(v, block_size=block, power=power)

    # (b) 收敛对照：AdamW / 未量化 / 幂律 / 线性
    losses: Dict[str, List[float]] = {}
    losses["adamw_fp32"], _ = _train_lstsq(
        lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.01), steps, device, seed, d
    )
    losses["adam8bit_unquantized"], opt_plain = _train_lstsq(
        lambda ps: opt8bit.Adam8bit(ps, lr=1e-2, weight_decay=0.01, quantize_state=False),
        steps, device, seed, d,
    )
    losses["adam8bit_power"], opt_power = _train_lstsq(
        lambda ps: opt8bit.Adam8bit(
            ps, lr=1e-2, weight_decay=0.01, block_size=block,
            min_quantize_numel=1, sq_mapping="power", sq_power=power,
        ),
        steps, device, seed, d,
    )
    losses["adam8bit_linear"], _ = _train_lstsq(
        lambda ps: opt8bit.Adam8bit(
            ps, lr=1e-2, weight_decay=0.01, block_size=block,
            min_quantize_numel=1, sq_mapping="linear",
        ),
        steps, device, seed, d,
    )

    # (c) 字节账：用一个够大的参数，避免 padding 与 scale 开销失真
    big = torch.nn.Parameter(torch.randn(int(cfg.get("byte_probe_rows", 1024)),
                                         int(cfg.get("byte_probe_cols", 256)), device=device))
    opt_big = opt8bit.Adam8bit([big], lr=1e-3, block_size=block, min_quantize_numel=1)
    big.grad = torch.randn_like(big)
    opt_big.step()

    bnb_mod, bnb_note = opt8bit.try_import_bnb()
    return {
        "mapping_comparison": mapping_cmp,
        "final_loss": {k: v[-1] for k, v in losses.items()},
        "first_loss": {k: v[0] for k, v in losses.items()},
        "loss_curves": losses,
        "state_bytes_unquantized": opt8bit.state_bytes_report(opt_plain),
        "state_bytes_quantized": opt8bit.state_bytes_report(opt_power),
        "state_bytes_big_param": opt8bit.state_bytes_report(opt_big),
        "bitsandbytes": {"available": bnb_mod is not None, "note": bnb_note},
    }


# --------------------------------------------------------------------------- #
# Q4
# --------------------------------------------------------------------------- #
def run_q4(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    device = ctx["device"]
    seed = ctx["seed"]
    model = make_mlp(cfg["model"], device, seed).eval()
    torch.manual_seed(seed + 1)
    x = torch.randn(int(cfg.get("batch", 8)), int(cfg.get("seq", 32)),
                    int(cfg["model"]["in_features"]), device=device)
    with torch.no_grad():
        ref_logits = model(x)

    compute_dtype = torch.float16 if cfg.get("compute_dtype", "float32") == "float16" else torch.float32
    out: Dict[str, Any] = {"compute_dtype": str(compute_dtype).replace("torch.", ""), "variants": {}}
    for variant in cfg.get("variants", [{"granularity": "per_channel"},
                                        {"granularity": "per_group", "group_size": 128}]):
        qmodel = copy.deepcopy(model)
        qmodel, replaced = weight_only.quantize_linear_modules(
            qmodel,
            granularity=variant["granularity"],
            group_size=int(variant.get("group_size", 128)),
            compute_dtype=compute_dtype,
        )
        with torch.no_grad():
            q_logits = qmodel(x)
        key = variant["granularity"]
        out["variants"][key] = {
            "n_replaced": len(replaced),
            "bytes": weight_only.module_weight_bytes(qmodel, reference_dtype=torch.float16),
            "output_rel_err": quantizers.relative_error(ref_logits, q_logits),
            "top1": weight_only.top1_agreement(ref_logits, q_logits),
            "logit_divergence": weight_only.logit_divergence(ref_logits, q_logits),
        }
    return out


# --------------------------------------------------------------------------- #
# Q5
# --------------------------------------------------------------------------- #
def run_q5(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    device = ctx["device"]
    seed = ctx["seed"]
    name, w = ctx["weights"][0]
    backends = qat.compare_backends(w.to(device), axis=int(cfg.get("axis", 0)))

    torch.manual_seed(seed)
    d_in = int(cfg["model"]["in_features"])
    base = make_mlp(cfg["model"], device, seed)
    wrapped = nn.Sequential(
        qat.FakeQuantLinear(base[0], backend=cfg.get("backend", "ste")),
        nn.SiLU(),
        qat.FakeQuantLinear(base[2], backend=cfg.get("backend", "ste")),
        nn.SiLU(),
        qat.FakeQuantLinear(base[4], backend=cfg.get("backend", "ste")),
    ).to(device)

    torch.manual_seed(seed + 2)
    n_batches = int(cfg.get("n_batches", 4))
    batch = int(cfg.get("batch", 16))
    batches = [
        (torch.randn(batch, d_in, device=device),
         torch.randn(batch, int(cfg["model"]["out_features"]), device=device))
        for _ in range(n_batches)
    ]

    def loss_fn(model: nn.Module, b: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        xb, yb = b
        return ((model(xb) - yb) ** 2).mean()

    amp = bool(cfg.get("amp", False)) and device == "cuda"
    scaler = None
    if amp:
        from mm_rl.scaler_rules import make_grad_scaler

        scaler, _api = make_grad_scaler("cuda", enabled=True)
    train = qat.qat_train_steps(
        wrapped,
        batches,
        loss_fn,
        lr=float(cfg.get("lr", 1e-3)),
        steps=int(cfg.get("steps", 5)),
        device=device,
        amp=amp,
        scaler=scaler,
    )
    return {
        "reference_layer": name,
        "backend_comparison": backends,
        "training": train,
        "amp_used": amp,
        "amp_note": (
            "AMP 只在 --device cuda 时打开：CPU 的 fp16 是软件模拟，"
            "跑通不代表 V100 上数值稳定。"
        ),
    }


# --------------------------------------------------------------------------- #
# Q6
# --------------------------------------------------------------------------- #
def run_q6(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    device = ctx["device"]
    seed = ctx["seed"]
    n_tokens = int(cfg.get("n_tokens", 128))
    channels = int(cfg.get("channels", 128))
    out_features = int(cfg.get("out_features", 96))
    n_outlier = int(cfg.get("n_outlier_channels", 4))
    mag = float(cfg.get("outlier_magnitude", 30.0))
    n_batches = int(cfg.get("n_batches", 3))

    g = torch.Generator(device="cpu").manual_seed(seed)
    outlier_idx = torch.randperm(channels, generator=g)[:n_outlier]
    batches = []
    for _ in range(n_batches):
        x = torch.randn((n_tokens, channels), generator=g)
        x[:, outlier_idx] *= mag
        batches.append(x.to(device))
    w = (torch.randn((out_features, channels), generator=g)).to(device)

    stats = [smooth.activation_outlier_stats(b, float(cfg.get("sigma_mult", 6.0))) for b in batches]
    overlaps = [
        smooth.overlap_rate(stats[i]["outlier_channels"], stats[i + 1]["outlier_channels"])
        for i in range(len(stats) - 1)
    ]
    sweep = smooth.alpha_sweep(
        batches[0], w, alphas=tuple(cfg.get("alphas", smooth.DEFAULT_ALPHAS))
    )
    best = min(sweep, key=lambda r: r["output_rel_err"])
    baseline = next(r for r in sweep if r["alpha"] == 0.0)
    return {
        "planted_outlier_channels": sorted(int(i) for i in outlier_idx.tolist()),
        "per_batch_stats": [
            {k: v for k, v in s.items() if k != "outlier_channels"} for s in stats
        ],
        "detected_channels_batch0": sorted(int(i) for i in stats[0]["outlier_channels"].tolist()),
        "cross_batch_overlap": overlaps,
        "alpha_sweep": sweep,
        "best_alpha": best["alpha"],
        "output_rel_err_at_best": best["output_rel_err"],
        "output_rel_err_at_alpha0": baseline["output_rel_err"],
        "improvement_factor": (
            baseline["output_rel_err"] / best["output_rel_err"]
            if best["output_rel_err"] > 0
            else float("inf")
        ),
        "text": smooth.format_alpha_sweep(sweep, "SmoothQuant alpha 扫描"),
    }


RUNNERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    "q1": run_q1,
    "q2": run_q2,
    "q3": run_q3,
    "q4": run_q4,
    "q5": run_q5,
    "q6": run_q6,
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="跑量化实验 Q1–Q6（零新依赖，纯 PyTorch）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="配置 JSON，相对周包根目录或绝对路径")
    p.add_argument("--experiments", default="", help="逗号分隔，默认用配置里的 experiments 字段")
    p.add_argument("--device", default="", help="cpu 或 cuda；留空用配置里的值")
    p.add_argument("--seed", type=int, default=-1, help="随机种子；-1 表示用配置里的值")
    p.add_argument("--tag", default="", help="run 目录标签；留空用配置里的 tag")
    p.add_argument(
        "--state-dict",
        default="",
        help="可选：相对 MM_WEIGHTS_ROOT 的 state_dict 路径，用真权重替代合成权重",
    )
    p.add_argument("--min-numel", type=int, default=4096, help="--state-dict 时挑张量的最小元素数")
    p.add_argument("--max-layers", type=int, default=8, help="--state-dict 时最多取几层")
    p.add_argument("--max-steps", type=int, default=-1, help="覆盖 q3/q5 的训练步数；-1 不覆盖")
    p.add_argument("--dry-run", action="store_true", help="只打印计划与输入形状，不做计算")
    p.add_argument("--quiet", action="store_true", help="不打印文本表格，只写文件")
    return p


def resolve_config(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.is_file():
        return p
    candidate = (paths.package_root() / raw).resolve()
    if candidate.is_file():
        return candidate
    if p.is_file():
        return p.resolve()
    raise paths.PathContractError(
        "找不到配置文件 {}\n"
        "下一步：用周包内自带的一份，例如\n"
        "  python lab/scripts/run_quant_suite.py --config {}".format(raw, DEFAULT_CONFIG)
    )


def main(argv: List[str]) -> int:
    use_utf8()
    args = build_parser().parse_args(argv)

    try:
        cfg_path = resolve_config(args.config)
    except paths.PathContractError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    device = args.device or cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("请求 --device cuda，但这台机器没有 CUDA。改用 --device cpu 或换机器。",
              file=sys.stderr)
        return 2
    seed = args.seed if args.seed >= 0 else int(cfg.get("seed", 0))
    tag = args.tag or cfg.get("tag", "quant_suite")
    selected = [s.strip() for s in (args.experiments or ",".join(cfg.get("experiments", EXPERIMENTS))).split(",") if s.strip()]
    for name in selected:
        if name not in EXPERIMENTS:
            print("未知实验 {!r}；可选：{}".format(name, ", ".join(EXPERIMENTS)), file=sys.stderr)
            return 2

    torch.manual_seed(seed)
    if args.state_dict:
        try:
            weights = load_state_dict_weights(args.state_dict, device, args.min_numel, args.max_layers)
            weight_source = "state_dict:{}".format(args.state_dict)
        except (paths.PathContractError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        weights = synthetic_weights(cfg["synthetic_weights"], device, seed)
        weight_source = "synthetic"

    if args.max_steps >= 0:
        for key in ("q3", "q5"):
            cfg.setdefault(key, {})["steps"] = args.max_steps

    plan = {
        "config": str(cfg_path),
        "device": device,
        "seed": seed,
        "tag": tag,
        "experiments": selected,
        "weight_source": weight_source,
        "weight_shapes": [[n, list(w.shape)] for n, w in weights],
        "torch": torch.__version__,
    }
    print("计划：")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\n--dry-run：只打印计划，不做计算。去掉这个开关就会真跑。")
        return 0

    ctx = {"device": device, "seed": seed, "weights": weights}
    run_root = paths.run_dir(tag)
    log_dir = run_root / "logs"
    jsonl_path = log_dir / "quant_suite.jsonl"
    summary_path = log_dir / "quant_suite_summary.json"

    results: Dict[str, Any] = {}
    with open(jsonl_path, "w", encoding="utf-8") as jl:
        for name in selected:
            t0 = time.time()
            try:
                res = RUNNERS[name](cfg.get(name, {}), ctx)
                status = "ok"
                err = ""
            except Exception as exc:  # 一个实验失败不该带走其余实验
                res = {}
                status = "error"
                err = "{}: {}".format(type(exc).__name__, exc)
            elapsed = time.time() - t0
            record = {
                "experiment": name,
                "status": status,
                "error": err,
                "seconds": round(elapsed, 3),
                "result": res,
            }
            results[name] = record
            jl.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            if not args.quiet:
                print("\n=== {} ({}，{:.2f}s) ===".format(name.upper(), status, elapsed))
                if status == "error":
                    print("  " + err)
                elif isinstance(res, dict) and "text" in res:
                    print(res["text"])
                elif name == "q2":
                    for bits, entry in res["by_bits"].items():
                        print(entry["text"])
                        print("  最优 alpha = {alpha}，MSE 相对 alpha=1 降低 {mse_reduction_vs_alpha1:.3f}x".format(**entry["best"]))
                elif name == "q3":
                    print("  最终 loss：" + json.dumps(res["final_loss"], ensure_ascii=False))
                    print("  状态字节比（量化 vs fp32）：{:.4f}".format(
                        res["state_bytes_big_param"]["ratio_vs_fp32"]))
                elif name == "q4":
                    for g, e in res["variants"].items():
                        print("  {:<12} 字节比 {:.4f}  top1 一致率 {:.4f}  输出相对误差 {:.5f}".format(
                            g, e["bytes"]["weight_bytes_ratio"], e["top1"]["top1_agreement"],
                            e["output_rel_err"]))
                elif name == "q5":
                    bc = res["backend_comparison"]
                    print("  STE vs aten fake-quant：前向逐位相等 {}，反向逐位相等 {}".format(
                        bc["forward_exact_equal_vs_aten"], bc["backward_exact_equal_vs_aten"]))
                    print("  loss {} -> {}".format(
                        res["training"]["loss_first"], res["training"]["loss_last"]))

    payload = {
        "schema": "mm_quant.suite.v1",
        "plan": plan,
        "results": results,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("\n逐条记录：{}".format(jsonl_path))
    print("汇总：    {}".format(summary_path))
    n_err = sum(1 for r in results.values() if r["status"] == "error")
    if n_err:
        print("\n{} 个实验失败，见上面的 error 行。".format(n_err), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
