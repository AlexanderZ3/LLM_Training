"""从 run 目录里的产物生成 evidence.json —— 唯一设计成可以带出公司的文件。

用法::

    python lab/scripts/make_evidence.py --day 2 --week 3 \\
        --run-tag day2_sft --skill fault_triage --ai-level A2 \\
        --status PASS --time-spent 110

    # 先看看会带出什么，不写文件
    python lab/scripts/make_evidence.py --day 0 --run-tag day0_probe --dry-run

它做什么 / 不做什么
------------------
做：读 run 目录下的 metrics_*.jsonl 与 *.json，把它们**聚合**成比值、布尔、
    计数、第几步，塞进 observations。
不做：不复制任何原始数字序列（losses 数组、逐步 grad_norm、逐步显存）、
    不写任何路径、不写主机名、不写拓扑、不写 GPU 型号与显存容量。

字段对齐 .claude/schemas/evidence.schema.json 的必填项：
date / week / day / card / skill / env / ai_level / primary_artifact /
status / failure_class / time_spent_min。

最后一道关是你自己
------------------
脚本只做机械过滤。生成后自己打开看一眼再带走——这一眼是你的责任，不是脚本的。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mm_v100 import cli  # noqa: E402
from mm_v100 import common as C  # noqa: E402

# 允许进入 observations 的键。白名单而不是黑名单：新增字段默认不带出。
SCALAR_WHITELIST = (
    "world_size", "micro_batch", "global_batch", "accum", "seq_len",
    "params_total", "n_fsdp_units", "steps", "skipped_steps", "applied_steps",
    "max_dloss", "max_rel_dloss", "first_diverge_step", "skip_ratio",
    "scale_first", "scale_last", "steps_compared", "max_abs_delta",
    "mean_abs_delta", "tol", "pass", "all_detected", "exact_items_match",
    "activation_ratio", "zero_std_group_ratio", "adv_std", "mean_reward",
    "init_loss", "abs_delta_to_ln2", "D1_init_loss_is_ln2",
    "D2_ref_grads_all_none", "G2_ratio_is_one", "ratio_max_abs_dev_from_1",
    "eos_rate", "role_leak_rate", "mean_new_tokens", "mean_echo_ratio",
    "n_default_bfloat16", "n_autocast_no_scaler", "n_torch_compile",
    "n_newer_api_hits", "label_align_violations_total", "label_ratio_mean",
    "saved_world_size", "load_world_size", "converted_step",
    "slices_sum_equals_total", "all_checks_pass", "bf16_supported",
    "device_count", "looks_exclusive", "total_calls", "total_bytes",
)

# 绝对不带出的键，即使它们是标量。
DENY_SUBSTRINGS = ("path", "dir", "root", "host", "node", "uuid", "serial",
                   "topology", "name", "file")


def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """把嵌套 dict 摊平成 a.b.c -> 标量。列表整个丢掉（原始序列不带出）。"""
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, prefix + str(key) + "."))
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        out[prefix.rstrip(".")] = obj
    return out


def allowed(key: str) -> bool:
    leaf = key.split(".")[-1]
    if any(bad in leaf.lower() for bad in DENY_SUBSTRINGS):
        return False
    return leaf in SCALAR_WHITELIST


def collect_observations(run_dir: str) -> Dict[str, Any]:
    """扫 run 目录下的 *.json 与 metrics_*.jsonl，只留白名单里的标量。"""
    obs: Dict[str, Any] = {}
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            "run 目录不存在：" + run_dir + "\n"
            "下一步：先跑对应那天的脚本，或者用 --run-dir 指到正确的目录。")
    for name in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, name)
        if name.endswith(".json") and os.path.isfile(full):
            with open(full, "r", encoding="utf-8") as fh:
                try:
                    payload = json.load(fh)
                except json.JSONDecodeError:
                    continue
            stem = os.path.splitext(name)[0]
            for key, value in flatten(payload).items():
                if allowed(key):
                    obs[stem + ":" + key.split(".")[-1]] = value
    metrics = [n for n in sorted(os.listdir(run_dir))
               if n.startswith("metrics_") and n.endswith("_rank0.jsonl")]
    for name in metrics:
        rows = [r for r in C.read_jsonl(os.path.join(run_dir, name))
                if r.get("event") != "header" and "loss" in r]
        if not rows:
            continue
        stem = os.path.splitext(name)[0]
        first, last = rows[0], rows[-1]
        obs[stem + ":n_steps"] = len(rows)
        obs[stem + ":loss_ratio_last_over_first"] = (
            float(last["loss"]) / float(first["loss"])
            if float(first["loss"]) else None)
        obs[stem + ":loss_monotone_down"] = bool(
            all(float(rows[i + 1]["loss"]) <= float(rows[i]["loss"])
                for i in range(len(rows) - 1)))
        obs[stem + ":final_grad_norm_finite"] = bool(
            float(last.get("grad_norm", 0.0)) == float(last.get("grad_norm", 0.0)))
        obs[stem + ":skipped_steps"] = int(last.get("skipped_steps", 0))
        obs[stem + ":I1_align_viol"] = int(
            last.get("I1_label_align_violations", 0))
        obs[stem + ":I2_step_ratio"] = float(
            last.get("I2_optimizer_step_ratio", 1.0))
        i3 = last.get("I3_cross_rank_grad_delta")
        obs[stem + ":I3_grad_delta_is_zero"] = (
            None if i3 is None else bool(float(i3) == 0.0))
        if "peak_mem_mb" in last and float(last["peak_mem_mb"]) > 0:
            # 显存绝对值属于公司机器信息；只带出「相对第一步的倍数」。
            base = float(first.get("peak_mem_mb", 0.0)) or float(last["peak_mem_mb"])
            obs[stem + ":peak_mem_ratio_last_over_first"] = (
                float(last["peak_mem_mb"]) / base)
    return obs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="从 run 目录聚合出 evidence.json（只含抽象值）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--week", type=int, default=3, help="周次")
    p.add_argument("--day", type=int, required=True, help="第几天（0-5）")
    p.add_argument("--card", type=str, default=None,
                   help="卡名；默认 day<N>")
    p.add_argument("--skill", type=str, required=True,
                   help="技能标签，如 byte_ledger / fault_triage / amp_scaler")
    p.add_argument("--env", type=str, default="v100",
                   choices=["cpu", "5070ti", "v100", "h100", "other"])
    p.add_argument("--ai-level", type=str, default="A2",
                   choices=["A0", "A1", "A2", "A3", "A4"])
    p.add_argument("--status", type=str, default="PASS",
                   choices=["PASS", "FAIL-MODEL", "FAIL-SYSTEM",
                            "INCONCLUSIVE", "BLOCKED"])
    p.add_argument("--failure-class", type=str, default="none",
                   choices=["none", "measurement", "data_contract", "numeric",
                            "system", "model", "insufficient", "env", "license"])
    p.add_argument("--time-spent", type=int, default=0, help="分钟")
    p.add_argument("--primary-artifact", type=str, default=None,
                   help="今天的唯一主要产物（一句话描述，不要写路径）")
    p.add_argument("--config-id", type=str, default=None,
                   help="配置/数据/代码版本的非敏感标识")
    p.add_argument("--notes", type=str, default=None)
    p.add_argument("--date", type=str, default=None,
                   help="YYYY-MM-DD；默认今天")
    p.add_argument("--run-tag", type=str, default=None,
                   help="run 目录名；与 --run-dir 二选一")
    p.add_argument("--run-dir", type=str, default=None,
                   help="直接给 run 目录路径")
    p.add_argument("--dry-run", action="store_true",
                   help="只把 evidence 打到 stdout，不写文件")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    if not args.run_tag and not args.run_dir:
        raise SystemExit("必须给 --run-tag 或 --run-dir")
    run_dir = args.run_dir or cli.resolve_out_dir(None, args.run_tag)
    obs = collect_observations(run_dir)
    date = args.date or datetime.date.today().isoformat()
    evidence: Dict[str, Any] = {
        "date": date,
        "week": int(args.week),
        "day": int(args.day),
        "card": args.card or ("day" + str(args.day)),
        "skill": args.skill,
        "env": args.env,
        "ai_level": args.ai_level,
        "primary_artifact": (args.primary_artifact
                             or ("day" + str(args.day) + " 的主要产物（待填写）")),
        "status": args.status,
        "failure_class": args.failure_class,
        "time_spent_min": int(args.time_spent),
        "observations": obs,
    }
    if args.config_id:
        evidence["config_id"] = args.config_id
    if args.notes:
        evidence["notes"] = args.notes
    text = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    print("")
    print("[evidence] observations 共 %d 个字段，全部来自白名单聚合。" % len(obs))
    print("[evidence] 带走之前自己再看一眼——脚本只做机械过滤。")
    if args.dry_run:
        return 0
    path = cli.write_json(evidence, os.path.join(run_dir, "evidence.json"))
    print("[evidence] -> " + path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
