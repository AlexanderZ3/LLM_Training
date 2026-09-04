"""解析训练日志 → CSV（+ PNG 或 ASCII 摘要）。

支持两种输入：
(a) bounded_train.py 的 JSONL（每行一个 dict，含 step/loss/lr/grad_norm/...）；
(b) MiniMind 原脚本 stdout（commit 7a6fddd，格式从源码抄录）：
    train_pretrain.py / train_full_sft.py:
      Epoch:[1/2](100/38783), loss: 5.1234, logits_loss: 5.1234, aux_loss: 0.0000, lr: 0.00049000, epoch_time: 12.0min
    train_dpo.py:
      Epoch:[1/1](100/2000), loss: 0.6931, dpo_loss: 0.6931, aux_loss: 0.0000, learning_rate: 0.00000004, epoch_time: 3.000min
    train_grpo.py:
      Epoch:[1/1](3/500), Reward: 1.2345, KL_ref: 0.0012, Adv Std: 0.9000, Adv Mean: 0.0000, Actor Loss: -0.0123, Avg Response Len: 120.50, Learning Rate: 0.00000030
字段名统一为小写下划线（`Adv Std` → `adv_std`，`Learning Rate` → `learning_rate`）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import Any, Dict, List, Optional
import re

_HEAD_RE = re.compile(r"^\s*Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\),\s*(.*)$")
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z_ ]*?):\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)\s*(min)?")
_SPARK = " .:-=+*#%@"


def _norm_key(k: str) -> str:
    return re.sub(r"\s+", "_", k.strip()).lower()


def parse_minimind_line(line: str) -> Optional[Dict[str, Any]]:
    m = _HEAD_RE.match(line)
    if not m:
        return None
    row: Dict[str, Any] = {"epoch": int(m.group(1)), "epochs": int(m.group(2)), "step": int(m.group(3)), "iters": int(m.group(4))}
    for k, v, _unit in _KV_RE.findall(m.group(5)):
        row[_norm_key(k)] = float(v)
    return row


def parse_minimind_stdout(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            r = parse_minimind_line(line)
            if r is not None:
                rows.append(r)
    return rows


def parse_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "step" in obj:
                rows.append(obj)
    return rows


def detect_format(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if s.startswith("{"):
                return "jsonl"
            if _HEAD_RE.match(s):
                return "minimind"
    return "minimind"


def load_rows(path: str, fmt: str = "auto") -> List[Dict[str, Any]]:
    if fmt == "auto":
        fmt = detect_format(path)
    return parse_jsonl(path) if fmt == "jsonl" else parse_minimind_stdout(path)


def numeric_fields(rows: List[Dict[str, Any]]) -> List[str]:
    keys: List[str] = []
    for r in rows:
        for k, v in r.items():
            if k not in keys and isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.append(k)
    return keys


def write_csv(rows: List[Dict[str, Any]], path: str) -> List[str]:
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    return keys


def sparkline(values: List[float], width: int = 40) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return ""
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return _SPARK[len(_SPARK) // 2] * len(vals)
    return "".join(_SPARK[min(len(_SPARK) - 1, int((v - lo) / (hi - lo) * (len(_SPARK) - 1)))] for v in vals)


def ascii_summary(rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> str:
    if not rows:
        return "(no rows)"
    fields = fields or [k for k in numeric_fields(rows) if k not in ("epoch", "epochs", "iters", "step")]
    lines = [f"rows={len(rows)} step {rows[0].get('step')}→{rows[-1].get('step')}",
             f"{'field':<18}{'first':>12}{'last':>12}{'min':>12}{'max':>12}  trend"]
    for k in fields:
        series = [r.get(k) for r in rows]
        nums = [float(v) for v in series if isinstance(v, (int, float)) and v is not None]
        if not nums:
            continue
        lines.append(f"{k:<18}{nums[0]:>12.5g}{nums[-1]:>12.5g}{min(nums):>12.5g}{max(nums):>12.5g}  {sparkline(nums)}")
    return "\n".join(lines)


def plot_png(rows: List[Dict[str, Any]], fields: List[str], path: str, title: str = "") -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fields = [f for f in fields if any(isinstance(r.get(f), (int, float)) for r in rows)]
    if not fields:
        return False
    n = len(fields)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, f in zip(axes, fields):
        xs = [r["step"] for r in rows if isinstance(r.get(f), (int, float))]
        ys = [r[f] for r in rows if isinstance(r.get(f), (int, float))]
        ax.plot(xs, ys, marker="." if len(xs) < 200 else None, linewidth=1)
        ax.set_ylabel(f)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="解析 bounded_train JSONL 或 MiniMind stdout 日志 → CSV/PNG/ASCII")
    p.add_argument("--input", required=True)
    p.add_argument("--format", choices=["auto", "jsonl", "minimind"], default="auto")
    p.add_argument("--csv", default=None, help="输出 CSV 路径")
    p.add_argument("--png", default=None, help="输出 PNG 路径（需 matplotlib；否则打印 ASCII）")
    p.add_argument("--fields", default=None, help="逗号分隔，缺省自动选 loss/lr/grad_norm/tokens_per_s/peak_mem_mb/reward 等")
    p.add_argument("--title", default="")
    args = p.parse_args(argv)
    rows = load_rows(args.input, args.format)
    if not rows:
        print(f"未在 {args.input} 解析到任何记录（格式={args.format}）")
        return 1
    preferred = ["loss", "logits_loss", "dpo_loss", "actor_loss", "reward", "kl_ref", "adv_std", "lr", "learning_rate",
                 "grad_norm", "scaler_scale", "tokens_per_s", "peak_mem_mb", "avg_response_len", "reward_margin"]
    avail = numeric_fields(rows)
    fields = [f.strip() for f in args.fields.split(",")] if args.fields else [f for f in preferred if f in avail]
    if args.csv:
        keys = write_csv(rows, args.csv)
        print(f"written {args.csv} ({len(rows)} rows, {len(keys)} cols)")
    print(ascii_summary(rows, fields))
    if args.png:
        ok = plot_png(rows, fields, args.png, args.title)
        print(f"written {args.png}" if ok else "matplotlib 不可用或无可绘字段，已用 ASCII 摘要代替")
    return 0


if __name__ == "__main__":
    sys.exit(main())
