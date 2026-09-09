#!/usr/bin/env python
"""跑低精度 RL 实验 R1–R6，结果写成 JSON + JSONL。

设计原则（CLAUDE.md 3.2）：**全部逻辑在 Python 里**。
run_rl_suite.sh 只做一件事——用对解释器调一次这个脚本。

实验对照表
----------
  r1  同一批 token 的 logprob 在 fp32 / fp16-compute / fp16-stored 三条路径下的差异，
      并把差异归到 IDENTICAL / MASK_MISMATCH / PRECISION / PATH_MISMATCH 四类之一。
      顺带注入一次 mask 错位和一次路径错位，证明归因器**能把它们分开**。
  r2  三个 KL 估计量 k1/k2/k3 在 fp64/fp32/fp16 下的均值、标准差、相对偏差、负值比例。
  r3  GradScaler 三条规则的正确用法 + 五种违规各自被抓到。
      **本机 scaler 是 enabled=False，验的是调用顺序，不是溢出行为。**
  r4  组内 advantage 归一化的灾难性抵消：扫 base 与 spread。
  r5  冻结 ref 模型 weight-only INT8：KL 相对偏差、log_ratio 符号一致率、policy 梯度余弦。
  r6  LoRA policy 的四项字节账 + step-0 不变量（KL 必须**精确**为 0）。

模型从哪来
----------
默认用 mm_rl.toy.TinyCausalLM（几十 KB，CPU 上一秒跑完）。
它**不是 MiniMind**，只保证子模块命名一致、前向确定。
所有关于「真实模型上偏差多大」的结论都要在 V100 上换真权重重跑。

用法
----
  python lab/scripts/run_rl_suite.py --help
  python lab/scripts/run_rl_suite.py --config lab/configs/rl_cpu_smoke.json --dry-run
  python lab/scripts/run_rl_suite.py --config lab/configs/rl_cpu_smoke.json
  python lab/scripts/run_rl_suite.py --config lab/configs/rl_v100.json --device cuda --experiments r3,r5
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

from mm_rl import advantage, kl, logprob, lora_policy, ref_model, scaler_rules  # noqa: E402
from mm_rl.toy import TinyCausalLM, ToyConfig, make_batch  # noqa: E402

EXPERIMENTS: Tuple[str, ...] = ("r1", "r2", "r3", "r4", "r5", "r6")
DEFAULT_CONFIG = "lab/configs/rl_cpu_smoke.json"


def build_model(cfg: Dict[str, Any], device: str, seed: int) -> TinyCausalLM:
    tc = ToyConfig(
        vocab_size=int(cfg.get("vocab_size", 256)),
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        num_heads=int(cfg.get("num_heads", 4)),
        num_kv_heads=int(cfg.get("num_kv_heads", 2)),
        intermediate_size=int(cfg.get("intermediate_size", 128)),
        max_seq_len=int(cfg.get("max_seq_len", 64)),
        tie_embeddings=bool(cfg.get("tie_embeddings", True)),
    )
    return TinyCausalLM(tc, seed=seed).to(device)


def _batch(cfg: Dict[str, Any], model_cfg: Dict[str, Any], device: str, seed: int):
    return make_batch(
        batch_size=int(cfg.get("batch", 4)),
        seq_len=int(cfg.get("seq", 16)),
        vocab_size=int(model_cfg.get("vocab_size", 256)),
        seed=seed,
        device=device,
        prompt_len=int(cfg.get("prompt_len", 4)),
    )


# --------------------------------------------------------------------------- #
def run_r1(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    model = ctx["model"].eval()
    b = _batch(cfg, ctx["model_cfg"], ctx["device"], ctx["seed"] + 1)
    with torch.no_grad():
        logits = model(b["input_ids"]).to(torch.float32)
    paths_lp = logprob.dual_precision_logprobs(logits, b["labels"])

    out: Dict[str, Any] = {"attribution": {}, "ratio": {}}
    for name in ("fp16_compute", "fp16_stored"):
        out["attribution"][name] = logprob.attribute_divergence(
            paths_lp["fp32"], paths_lp[name], b["mask"], b["mask"], dtype=torch.float16
        )
    # 注入 1：mask 错位（把 prompt 也算进 loss）——归因器必须先抓 mask，不谈精度
    out["attribution"]["injected_mask_shift"] = logprob.attribute_divergence(
        paths_lp["fp32"], paths_lp["fp32"], b["mask"], ~b["mask"], dtype=torch.float16
    )
    # 注入 2：路径错位（等价于 old_logp 来自另一个模型版本 / 不同 temperature）
    shifted = paths_lp["fp32"] * float(cfg.get("path_mismatch_scale", 1.05))
    out["attribution"]["injected_path_mismatch"] = logprob.attribute_divergence(
        paths_lp["fp32"], shifted, b["mask"], b["mask"], dtype=torch.float16
    )

    for compute in (torch.float32, torch.float16):
        key = str(compute).replace("torch.", "")
        out["ratio"][key] = logprob.ratio_stats(
            paths_lp["fp32"],
            paths_lp["fp16_stored"],
            clip_eps=float(cfg.get("clip_eps", 0.2)),
            mask=b["mask"],
            compute_dtype=compute,
        )
    out["determinism"] = logprob.determinism_check(lambda: model(b["input_ids"]))
    out["n_valid_tokens"] = int(b["mask"].sum())
    return out


def run_r2(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    model = ctx["model"].eval()
    b = _batch(cfg, ctx["model_cfg"], ctx["device"], ctx["seed"] + 2)
    with torch.no_grad():
        logits = model(b["input_ids"]).to(torch.float32)
    lp_new = logprob.token_logprobs(logits, b["labels"], compute_dtype=torch.float32)
    out: Dict[str, Any] = {"by_divergence": {}}
    for sigma in cfg.get("logratio_sigmas", [0.002, 0.02, 0.2]):
        g = torch.Generator(device="cpu").manual_seed(ctx["seed"])
        noise = torch.randn(lp_new.shape, generator=g).to(lp_new.device) * float(sigma)
        lp_ref = lp_new + noise
        rep = kl.precision_report(lp_new, lp_ref, mask=b["mask"])
        out["by_divergence"][str(sigma)] = rep
        out.setdefault("text", []).append(
            "log_ratio 尺度 sigma={}\n".format(sigma) + kl.format_precision_report(rep)
        )
    out["text"] = "\n\n".join(out["text"])
    return out


def run_r3(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    device = ctx["device"]
    torch.manual_seed(ctx["seed"])
    d = int(cfg.get("dim", 16))
    policy_net = nn.Linear(d, d).to(device)
    value_net = nn.Linear(d, 1).to(device)
    opt_p = torch.optim.SGD(policy_net.parameters(), lr=0.01)
    opt_v = torch.optim.SGD(value_net.parameters(), lr=0.01)

    amp_enabled = bool(cfg.get("amp", False)) and device == "cuda"
    scaler, api = scaler_rules.make_grad_scaler(
        device if device == "cuda" else "cpu", enabled=amp_enabled
    )
    dscaler = scaler_rules.DisciplinedScaler(scaler, ["policy", "value"])
    tasks = [
        scaler_rules.AmpTask("policy", opt_p, lambda b: (policy_net(b) ** 2).mean(),
                             list(policy_net.parameters()), float(cfg.get("max_grad_norm", 1.0))),
        scaler_rules.AmpTask("value", opt_v, lambda b: (value_net(b) ** 2).mean(),
                             list(value_net.parameters()), float(cfg.get("max_grad_norm", 1.0))),
    ]
    micro = [torch.randn(int(cfg.get("batch", 8)), d, device=device)
             for _ in range(int(cfg.get("accum_steps", 4)))]
    correct = scaler_rules.rl_amp_iteration(
        dscaler, tasks, micro, device_type=device, amp_enabled=amp_enabled
    )

    # 五种违规，每种都要被抓到，并且抓到的规则号必须对
    violations: Dict[str, Any] = {}

    def _expect(rule: str, fn: Callable[[], None]) -> Dict[str, Any]:
        try:
            fn()
            return {"raised": False, "rule": None, "as_expected": False}
        except scaler_rules.ScalerRuleViolation as exc:
            return {"raised": True, "rule": exc.rule, "as_expected": exc.rule == rule,
                    "message": str(exc).splitlines()[0]}

    def v_double_update() -> None:
        d1 = scaler_rules.ScalerDiscipline(["policy"])
        d1.begin_iteration(65536.0)
        d1.note_backward("policy", 65536.0)
        d1.note_step("policy", 65536.0)
        d1.note_update(65536.0)
        d1.note_update(32768.0)

    def v_scale_changed() -> None:
        d2 = scaler_rules.ScalerDiscipline(["policy"])
        d2.begin_iteration(65536.0)
        d2.note_backward("policy", 65536.0)
        d2.note_micro_boundary()
        d2.note_backward("policy", 32768.0)

    def v_missing_step() -> None:
        d3 = scaler_rules.ScalerDiscipline(["policy", "value"])
        d3.begin_iteration(1.0)
        d3.note_backward("policy", 1.0)
        d3.note_step("policy", 1.0)
        d3.note_update(1.0)

    def v_double_unscale() -> None:
        d4 = scaler_rules.ScalerDiscipline(["policy"])
        d4.begin_iteration(1.0)
        d4.note_backward("policy", 1.0)
        d4.note_unscale("policy", 1.0)
        d4.note_unscale("policy", 1.0)

    def v_step_after_update() -> None:
        d5 = scaler_rules.ScalerDiscipline(["policy"])
        d5.begin_iteration(1.0)
        d5.note_backward("policy", 1.0)
        d5.note_step("policy", 1.0)
        d5.note_update(1.0)
        d5.note_step("policy", 1.0)

    violations["R3-1_double_update"] = _expect("R3-1", v_double_update)
    violations["R3-3_scale_changed_mid_accum"] = _expect("R3-3", v_scale_changed)
    violations["R3-4_missing_optimizer_step"] = _expect("R3-4", v_missing_step)
    violations["R3-5_double_unscale"] = _expect("R3-5", v_double_unscale)
    violations["R3-2_step_after_update"] = _expect("R3-2", v_step_after_update)

    # inf/nan 首次出现定位
    probe = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)).to(device)
    tracker = scaler_rules.first_nonfinite_hooks(probe)
    with torch.no_grad():
        probe[2].weight.fill_(float("inf"))
    x = torch.randn(4, d, device=device, requires_grad=True)
    probe(x).sum().backward()
    tracker.remove()

    return {
        "scaler_api": api,
        "scaler_enabled": bool(getattr(scaler, "is_enabled", lambda: False)()),
        "amp_enabled": amp_enabled,
        "correct_iteration": correct,
        "violations": violations,
        "all_violations_caught": all(v["as_expected"] for v in violations.values()),
        "nonfinite_tracker": tracker.report(),
        "cpu_caveat": (
            "本机 scaler enabled=False（torch 2.1 的 GradScaler 只有 CUDA 实现），"
            "scale 恒为 1、永不跳步。这里验的是**调用顺序**，"
            "溢出与 backoff 的真实行为待 V100。"
        ),
    }


def run_r4(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"cases": {}, "text": []}
    for case in cfg.get("cases", [{"base": 1.0, "spread": 1e-3}]):
        rewards = advantage.cancellation_case(
            n_groups=int(cfg.get("n_groups", 64)),
            group_size=int(cfg.get("group_size", 8)),
            base=float(case["base"]),
            spread=float(case["spread"]),
            seed=ctx["seed"],
            device=ctx["device"],
        )
        rep = advantage.advantage_report(rewards, eps=float(cfg.get("eps", 1e-4)))
        key = "base={}_spread={}".format(case["base"], case["spread"])
        out["cases"][key] = rep
        out["text"].append(key + "\n" + advantage.format_advantage_report(rep))
    out["text"] = "\n\n".join(out["text"])
    return out


def run_r5(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    device = ctx["device"]
    base = ctx["model"]
    ref_fp = copy.deepcopy(base).eval()
    policy = copy.deepcopy(base)
    # policy 必须先偏离 ref，否则 log_ratio 恒为 0，符号一致率和相对偏差都没有定义
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(ctx["seed"] + 7)
        for p in policy.parameters():
            p.add_(torch.randn(p.shape, generator=g).to(p.device) * float(cfg.get("policy_drift", 0.01)))

    compute_dtype = torch.float16 if cfg.get("compute_dtype", "float32") == "float16" else torch.float32
    ref_q, info = ref_model.quantize_reference_model(
        ref_fp,
        granularity=cfg.get("granularity", "per_channel"),
        group_size=int(cfg.get("group_size", 128)),
        compute_dtype=compute_dtype,
        skip_name_contains=tuple(cfg.get("skip_name_contains", ["lm_head"])),
    )
    b = _batch(cfg, ctx["model_cfg"], device, ctx["seed"] + 5)
    with torch.no_grad():
        lp = policy(b["input_ids"]).to(torch.float32)
        lrf = ref_fp(b["input_ids"]).to(torch.float32)
        lrq = ref_q(b["input_ids"]).to(torch.float32)
    kl_rep = ref_model.ref_kl_report(lp, lrf, lrq, b["labels"], b["mask"])

    g2 = torch.Generator(device="cpu").manual_seed(ctx["seed"] + 9)
    adv = torch.randn(b["labels"].shape, generator=g2).to(device)
    grads = ref_model.policy_grad_cosine(
        policy, ref_fp, ref_q, b["input_ids"], b["labels"], adv, b["mask"],
        beta=float(cfg.get("beta", 0.04)),
    )
    return {
        "quantization": info,
        "kl_report": kl_rep,
        "grad": grads,
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "text": ref_model.format_ref_report(kl_rep, info)
        + "\n  policy 梯度余弦相似度 {:.6f}（{} 个梯度元素）".format(
            grads["grad_cosine"], grads["n_grad_elements"]
        ),
    }


def run_r6(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    base = ctx["model"]
    ref = copy.deepcopy(base).eval()
    policy = copy.deepcopy(base)
    policy, targets = lora_policy.apply_lora(
        policy,
        target_suffixes=tuple(cfg.get("target_suffixes", ["q_proj", "v_proj"])),
        r=int(cfg.get("r", 8)),
        alpha=int(cfg.get("alpha", 16)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    b = _batch(cfg, ctx["model_cfg"], ctx["device"], ctx["seed"] + 6)
    invariant = lora_policy.step0_identity_check(
        policy, ref, b["input_ids"], b["labels"], b["mask"]
    )
    full_ledger = lora_policy.byte_ledger(copy.deepcopy(base), b["input_ids"])
    lora_ledger = lora_policy.byte_ledger(policy, b["input_ids"])

    broken = copy.deepcopy(policy)
    for mod in broken.modules():
        if isinstance(mod, lora_policy.LoRALinear):
            mod.break_zero_init(float(cfg.get("break_std", 0.02)), seed=ctx["seed"])
    invariant_broken = lora_policy.step0_identity_check(
        broken, ref, b["input_ids"], b["labels"], b["mask"]
    )
    return {
        "lora_targets": targets,
        "step0_invariant": invariant,
        "step0_invariant_after_breaking_zero_init": invariant_broken,
        "byte_ledger_full": full_ledger,
        "byte_ledger_lora": lora_ledger,
        "text": lora_policy.format_byte_ledger(full_ledger, lora_ledger)
        + "\nstep-0 KL 精确为 0：{}（破坏零初始化后：{}）".format(
            invariant["exact_zero_kl"], invariant_broken["exact_zero_kl"]
        ),
    }


RUNNERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    "r1": run_r1,
    "r2": run_r2,
    "r3": run_r3,
    "r4": run_r4,
    "r5": run_r5,
    "r6": run_r6,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="跑低精度 RL 实验 R1–R6（零新依赖，不装 vLLM/verl/OpenRLHF/trl）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="配置 JSON，相对周包根目录或绝对路径")
    p.add_argument("--experiments", default="", help="逗号分隔，默认用配置里的 experiments 字段")
    p.add_argument("--device", default="", help="cpu 或 cuda；留空用配置里的值")
    p.add_argument("--seed", type=int, default=-1, help="随机种子；-1 表示用配置里的值")
    p.add_argument("--tag", default="", help="run 目录标签；留空用配置里的 tag")
    p.add_argument("--dry-run", action="store_true", help="只打印计划与模型规模，不做计算")
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
        "  python lab/scripts/run_rl_suite.py --config {}".format(raw, DEFAULT_CONFIG)
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
    tag = args.tag or cfg.get("tag", "rl_suite")
    selected = [
        s.strip()
        for s in (args.experiments or ",".join(cfg.get("experiments", EXPERIMENTS))).split(",")
        if s.strip()
    ]
    for name in selected:
        if name not in EXPERIMENTS:
            print("未知实验 {!r}；可选：{}".format(name, ", ".join(EXPERIMENTS)), file=sys.stderr)
            return 2

    torch.manual_seed(seed)
    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg, device, seed)

    plan = {
        "config": str(cfg_path),
        "device": device,
        "seed": seed,
        "tag": tag,
        "experiments": selected,
        "model": model.cfg.to_dict(),
        "model_params": model.num_parameters(),
        "torch": torch.__version__,
    }
    print("计划：")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\n--dry-run：只打印计划，不做计算。去掉这个开关就会真跑。")
        return 0

    ctx = {"device": device, "seed": seed, "model": model, "model_cfg": model_cfg}
    run_root = paths.run_dir(tag)
    log_dir = run_root / "logs"
    jsonl_path = log_dir / "rl_suite.jsonl"
    summary_path = log_dir / "rl_suite_summary.json"

    results: Dict[str, Any] = {}
    with open(jsonl_path, "w", encoding="utf-8") as jl:
        for name in selected:
            t0 = time.time()
            try:
                res = RUNNERS[name](cfg.get(name, {}), ctx)
                status, err = "ok", ""
            except Exception as exc:
                res, status, err = {}, "error", "{}: {}".format(type(exc).__name__, exc)
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
                elif name == "r1":
                    for k, v in res["attribution"].items():
                        print("  {:<26} {:<14} {}".format(k, v["verdict"], v["reason"]))
                    for k, v in res["ratio"].items():
                        print("  ratio(compute={}) max|r-1| = {:.3e}，clip 比例 {:.4f}".format(
                            k, v["max_abs_ratio_minus_1"], v["frac_clipped"]))
                elif name == "r3":
                    print("  scaler: {}（enabled={}）".format(res["scaler_api"], res["scaler_enabled"]))
                    print("  正确 iteration：{} 个 micro-batch，梯度范数 {}".format(
                        res["correct_iteration"]["n_micro_batches"],
                        {k: round(v, 4) for k, v in res["correct_iteration"]["grad_norm"].items()}))
                    for k, v in res["violations"].items():
                        print("    {:<34} 抓到={} 规则={} 符合预期={}".format(
                            k, v["raised"], v["rule"], v["as_expected"]))
                    first = res["nonfinite_tracker"]["first"]
                    print("  第一个非有限值：{}".format(
                        "{}@{}".format(first["module"], first["where"]) if first else "无"))

    payload = {"schema": "mm_rl.suite.v1", "plan": plan, "results": results}
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
