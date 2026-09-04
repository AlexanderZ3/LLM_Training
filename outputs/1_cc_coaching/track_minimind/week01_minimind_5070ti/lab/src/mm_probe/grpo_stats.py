"""GRPO 组内统计：advantage、std=0 组占比、平均生成长度。

与 MiniMind `train_grpo.py` 完全相同的 advantage 定义：
    mean_r = group.mean();  std_r = group.std(unbiased=False)
    advantage = (r - mean_r) / (std_r + 1e-4)
reward 全等的组 std=0 → advantage 全 0 → 该组没有任何学习信号（只剩 KL 项）。

输入来源两种：
1. JSONL：每行 {"step": int, "rewards": [[g0r0, g0r1, ...], [g1r0, ...]], "lengths": [[...], ...]（可选）}
   或 {"step": int, "rewards": [flat...], "num_generations": G}
2. MiniMind `--debug_mode --debug_interval 1` 的 stdout（`[DEBUG] gen[j] reward=...` 行），由
   scripts/run_grpo_smoke.* tee 到日志文件；长度用 RESPONSE_BEGIN/END 之间的字符数（`resp_chars`），
   token 级平均长度取同步的 `Avg Response Len` 字段。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

Number = Union[int, float]

_DEBUG_STEP_RE = re.compile(r"\[DEBUG\] step=(\d+), sample\[(\d+)\]")
_DEBUG_REWARD_RE = re.compile(r"\[DEBUG\] gen\[(\d+)\] reward=([-+0-9.eE]+)")
_RESP_BEGIN_RE = re.compile(r"\[DEBUG\] gen\[(\d+)\] RESPONSE_BEGIN")
_RESP_END_RE = re.compile(r"\[DEBUG\] gen\[(\d+)\] RESPONSE_END")
_SUMMARY_RE = re.compile(r"Epoch:\[\d+/\d+\]\((\d+)/\d+\),.*?Avg Response Len: ([-+0-9.eE]+)")


def _mean(xs: Sequence[Number]) -> float:
    return sum(float(x) for x in xs) / len(xs) if xs else float("nan")


def _pstd(xs: Sequence[Number]) -> float:
    if not xs:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((float(x) - m) ** 2 for x in xs) / len(xs))


def _to_groups(rewards, num_generations: Optional[int]) -> List[List[float]]:
    if hasattr(rewards, "tolist"):
        rewards = rewards.tolist()
    rewards = list(rewards)
    if rewards and isinstance(rewards[0], (list, tuple)):
        return [[float(x) for x in g] for g in rewards]
    if not num_generations:
        raise ValueError("扁平 rewards 需要 num_generations")
    if len(rewards) % num_generations != 0:
        raise ValueError(f"len(rewards)={len(rewards)} 不能被 num_generations={num_generations} 整除")
    return [[float(x) for x in rewards[i:i + num_generations]] for i in range(0, len(rewards), num_generations)]


def group_advantage(rewards, eps: float = 1e-4, num_generations: Optional[int] = None):
    """纯函数：组内 (r - mean) / (pop_std + eps)。

    rewards 为 list[list[float]]（每 prompt 一组）时返回同结构；
    为扁平 list 且给 num_generations 时返回扁平 list（与 MiniMind 的 [B*G] 排布一致）。
    """
    flat_in = not (rewards and isinstance(list(rewards)[0], (list, tuple)))
    groups = _to_groups(rewards, num_generations)
    adv = []
    for g in groups:
        m, s = _mean(g), _pstd(g)
        adv.append([(x - m) / (s + eps) for x in g])
    if flat_in:
        return [a for g in adv for a in g]
    return adv


def group_stats(groups: List[List[float]], eps: float = 1e-4) -> List[Dict[str, Any]]:
    out = []
    for gi, g in enumerate(groups):
        s = _pstd(g)
        out.append({"group": gi, "n": len(g), "mean": _mean(g), "std": s, "zero_std": s == 0.0,
                    "advantages": group_advantage([g], eps)[0], "max_minus_min": (max(g) - min(g)) if g else float("nan")})
    return out


def summarize(records: List[Dict[str, Any]], eps: float = 1e-4) -> Dict[str, Any]:
    """records: [{"step", "rewards": groups, "lengths": groups(optional), "avg_len"(optional)}]"""
    per_step = []
    n_groups = 0
    n_zero = 0
    all_std: List[float] = []
    all_rewards: List[float] = []
    all_lens: List[float] = []
    for rec in records:
        groups = _to_groups(rec["rewards"], rec.get("num_generations"))
        gs = group_stats(groups, eps)
        n_groups += len(gs)
        n_zero += sum(1 for g in gs if g["zero_std"])
        all_std.extend(g["std"] for g in gs)
        all_rewards.extend(x for g in groups for x in g)
        lens = rec.get("lengths")
        step_len = None
        if lens:
            flat = [float(x) for g in _to_groups(lens, rec.get("num_generations")) for x in g]
            all_lens.extend(flat)
            step_len = _mean(flat)
        elif rec.get("avg_len") is not None:
            step_len = float(rec["avg_len"])
            all_lens.append(step_len)
        adv_flat = [a for g in gs for a in g["advantages"]]
        per_step.append({
            "step": rec.get("step"),
            "n_groups": len(gs),
            "zero_std_groups": sum(1 for g in gs if g["zero_std"]),
            "mean_reward": _mean([x for g in groups for x in g]),
            "mean_group_std": _mean([g["std"] for g in gs]),
            "adv_mean": _mean(adv_flat),
            "adv_std": _pstd(adv_flat),
            "avg_len": step_len,
        })
    return {
        "n_steps": len(records),
        "n_groups": n_groups,
        "zero_std_group_ratio": (n_zero / n_groups) if n_groups else float("nan"),
        "mean_group_std": _mean(all_std) if all_std else float("nan"),
        "mean_reward": _mean(all_rewards) if all_rewards else float("nan"),
        "mean_len": _mean(all_lens) if all_lens else None,
        "per_step": per_step,
    }


def parse_jsonl(path: str) -> List[Dict[str, Any]]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def parse_debug_log(path: str) -> List[Dict[str, Any]]:
    """解析 MiniMind train_grpo.py --debug_mode 的 stdout，按 step 汇成 groups。"""
    steps: Dict[int, Dict[str, Any]] = {}
    cur_step: Optional[int] = None
    cur_sample: Optional[int] = None
    in_resp: Optional[int] = None
    resp_chars = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _DEBUG_STEP_RE.search(line)
            if m:
                cur_step, cur_sample = int(m.group(1)), int(m.group(2))
                st = steps.setdefault(cur_step, {"step": cur_step, "rewards": {}, "lengths": {}, "avg_len": None})
                st["rewards"].setdefault(cur_sample, [])
                st["lengths"].setdefault(cur_sample, [])
                continue
            if in_resp is not None:
                if _RESP_END_RE.search(line):
                    steps[cur_step]["lengths"][cur_sample].append(resp_chars)
                    in_resp = None
                else:
                    resp_chars += len(line) + 1
                continue
            m = _RESP_BEGIN_RE.search(line)
            if m and cur_step is not None:
                in_resp = int(m.group(1))
                resp_chars = 0
                continue
            m = _DEBUG_REWARD_RE.search(line)
            if m and cur_step is not None:
                steps[cur_step]["rewards"][cur_sample].append(float(m.group(2)))
                continue
            m = _SUMMARY_RE.search(line)
            if m:
                s = int(m.group(1))
                if s in steps:
                    steps[s]["avg_len"] = float(m.group(2))
    records = []
    for s in sorted(steps):
        st = steps[s]
        groups = [st["rewards"][k] for k in sorted(st["rewards"]) if st["rewards"][k]]
        lens = [st["lengths"][k] for k in sorted(st["lengths"]) if st["lengths"][k]]
        rec = {"step": s, "rewards": groups, "avg_len": st["avg_len"]}
        if lens and all(len(l) == len(g) for l, g in zip(lens, groups)):
            rec["resp_chars"] = lens
        records.append(rec)
    return records


def write_csv(summary: Dict[str, Any], path: str) -> None:
    rows = summary["per_step"]
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="GRPO 组内 reward/advantage 统计")
    p.add_argument("--input", required=True, help="rewards JSONL 或 train_grpo --debug_mode 的日志")
    p.add_argument("--format", choices=["auto", "jsonl", "debug-log"], default="auto")
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--csv", default=None)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    fmt = args.format
    if fmt == "auto":
        with open(args.input, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(2048)
        fmt = "jsonl" if head.lstrip().startswith("{") else "debug-log"
    records = parse_jsonl(args.input) if fmt == "jsonl" else parse_debug_log(args.input)
    if fmt == "debug-log":
        for r in records:
            if "resp_chars" in r:
                r["lengths"] = r.pop("resp_chars")
    summary = summarize(records, args.eps)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_step"}, ensure_ascii=False, indent=2))
    print(f"{'step':>6} {'groups':>6} {'zero_std':>8} {'mean_r':>9} {'grp_std':>9} {'adv_std':>9} {'avg_len':>9}")
    for r in summary["per_step"]:
        al = "" if r["avg_len"] is None else f"{r['avg_len']:.1f}"
        print(f"{str(r['step']):>6} {r['n_groups']:>6} {r['zero_std_groups']:>8} {r['mean_reward']:>9.4f} {r['mean_group_std']:>9.4f} {r['adv_std']:>9.4f} {al:>9}")
    if args.csv:
        write_csv(summary, args.csv)
        print(f"written {args.csv}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"written {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
