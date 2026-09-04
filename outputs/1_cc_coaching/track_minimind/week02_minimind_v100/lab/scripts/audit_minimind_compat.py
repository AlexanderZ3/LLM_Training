"""静态审计：MiniMind 的 trainer/*.py 与 model/*.py 在 torch 2.1.0 + V100 上会踩什么。

只读，不改动 MiniMind 一个字节
------------------------------
本脚本用正则扫源码，不 import、不执行、不写回。输出一份 Markdown 到 ``--out``，
供 Day 0 的兼容审计使用。

四类发现
--------
1. **默认 bfloat16 的脚本**：MiniMind 各 trainer 的 ``--dtype`` 默认值。
   V100 (sm70) 上 ``autocast(dtype=bfloat16)`` 直接抛
   ``RuntimeError: Current CUDA Device does not support bfloat16``，
   所以每个这样的脚本都必须显式加 ``--dtype float16``。
2. **GradScaler 有无**：fp16 训练没有 GradScaler，梯度会在 fp16 下溢到 0，
   loss 看着在动其实没学到东西。逐文件报告 ``GradScaler`` / ``scaler.scale`` /
   ``scaler.step`` / ``scaler.update`` 的出现次数。
3. **torch.compile 使用点**：Triton 2.1 声称支持 CC>=7.0，但公司机器未实测；
   本周默认关闭，先把调用点列出来。
4. **torch 2.1 里不存在的 API**：内置符号清单（见 ``NEWER_APIS``）正则匹配。
   命中说明该文件是按更新的 torch 写的，在公司的 2.1.0 上会 ``AttributeError`` /
   ``ImportError``。

用法::

    python lab/scripts/audit_minimind_compat.py --minimind-root /path/to/minimind \
        --out ./compat_audit.md
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mm_dist import common as C  # noqa: E402

# (符号, 正则, 最早出现的 torch 版本, 说明)
NEWER_APIS: List[Tuple[str, str, str, str]] = [
    ("torch.distributed.tensor",
     r"torch\.distributed\.tensor|from\s+torch\.distributed\.tensor",
     "2.5", "DTensor 的公开入口；2.1 只有私有的 torch.distributed._tensor"),
    ("fully_shard", r"\bfully_shard\b", "2.4",
     "FSDP2 的 per-parameter 分片入口；2.1 只有 FullyShardedDataParallel"),
    ("DTensor", r"\bDTensor\b", "2.5",
     "分布式张量类型；2.1 无公开 DTensor"),
    ("device_mesh", r"\bdevice_mesh\b|\binit_device_mesh\b|\bDeviceMesh\b", "2.2",
     "init_device_mesh 在 2.2 才公开；2.1 用 dist.new_group 手工建组"),
    ("torch.accelerator", r"torch\.accelerator\b", "2.6",
     "设备无关加速器 API；2.1 用 torch.cuda.*"),
    ("torch.get_default_device", r"torch\.get_default_device\b", "2.3",
     "2.1 只有 set_default_device，没有 get"),
    ("torch.distributed.pipelining", r"torch\.distributed\.pipelining", "2.4",
     "流水并行库；2.1 无"),
    ("torch.nn.attention", r"torch\.nn\.attention|from\s+torch\.nn\.attention",
     "2.3", "sdpa_kernel/SDPBackend 的新家；2.1 用 torch.backends.cuda.sdp_kernel"),
    ("sdpa_kernel", r"\bsdpa_kernel\b", "2.3",
     "2.1 的等价物是 torch.backends.cuda.sdp_kernel"),
    ("enable_gqa", r"enable_gqa\s*=", "2.5",
     "scaled_dot_product_attention 的 GQA 参数；2.1 需要手工 repeat_kv"),
    ("torch.distributed.checkpoint.state_dict",
     r"torch\.distributed\.checkpoint\.state_dict|from\s+torch\.distributed\.checkpoint\.state_dict",
     "2.2", "get_state_dict/set_state_dict 辅助层；2.1 用 FSDP.state_dict_type"),
    ("torch.library.custom_op", r"torch\.library\.custom_op|register_fake", "2.4",
     "自定义算子新 API；2.1 用 torch.library.Library"),
    ("torch.utils.swap_tensors", r"torch\.utils\.swap_tensors", "2.3", "2.1 无"),
    ("torch.amp.GradScaler", r"torch\.amp\.GradScaler", "2.3",
     "2.1 的写法是 torch.cuda.amp.GradScaler"),
    ("torch.compiler", r"torch\.compiler\.", "2.2",
     "torch.compiler 命名空间；2.1 用 torch._dynamo"),
]

DTYPE_DEFAULT_RE = re.compile(
    r"add_argument\(\s*[\"']--dtype[\"'][^)]*?default\s*=\s*[\"']([A-Za-z0-9_]+)[\"']",
    re.S)
SCALER_PATTERNS = {
    "GradScaler": re.compile(r"\bGradScaler\b"),
    "scaler.scale": re.compile(r"scaler\.scale\b"),
    "scaler.step": re.compile(r"scaler\.step\b"),
    "scaler.update": re.compile(r"scaler\.update\b"),
    "scaler.unscale_": re.compile(r"scaler\.unscale_\b"),
}
COMPILE_RE = re.compile(r"torch\.compile\s*\(")
AUTOCAST_RE = re.compile(r"autocast\s*\(")
BF16_RE = re.compile(r"bfloat16")


def iter_target_files(root: str) -> List[str]:
    """trainer/*.py 与 model/*.py，按路径排序。"""
    files: List[str] = []
    for sub in ("trainer", "model"):
        directory = os.path.join(root, sub)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.endswith(".py"):
                files.append(os.path.join(directory, name))
    return files


def line_numbers(text: str, pattern: re.Pattern) -> List[int]:
    return [text.count("\n", 0, m.start()) + 1 for m in pattern.finditer(text)]


def scan_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    dtype_defaults = DTYPE_DEFAULT_RE.findall(text)
    scaler_hits = {name: len(pat.findall(text))
                   for name, pat in SCALER_PATTERNS.items()}
    newer: List[Dict[str, Any]] = []
    for symbol, pattern, since, note in NEWER_APIS:
        lines = line_numbers(text, re.compile(pattern))
        if lines:
            newer.append({"symbol": symbol, "since": since, "note": note,
                          "lines": lines})
    return {
        "path": path,
        # 报告里统一用 / 分隔，Windows 与 Linux 上的输出才可以直接比对
        "name": (os.path.basename(os.path.dirname(path)) + "/"
                 + os.path.basename(path)),
        "loc": text.count("\n") + 1,
        "dtype_defaults": dtype_defaults,
        "defaults_bfloat16": any(d == "bfloat16" for d in dtype_defaults),
        "mentions_bfloat16": bool(BF16_RE.search(text)),
        "scaler": scaler_hits,
        "has_gradscaler": scaler_hits["GradScaler"] > 0,
        "uses_autocast": len(AUTOCAST_RE.findall(text)) > 0,
        "compile_lines": line_numbers(text, COMPILE_RE),
        "newer_apis": newer,
    }


def render_markdown(root: str, results: List[Dict[str, Any]],
                    redact_root: bool) -> str:
    shown_root = os.path.basename(root) if redact_root else root
    commit = C.minimind_commit(root)
    bf16_scripts = [r for r in results if r["defaults_bfloat16"]]
    amp_no_scaler = [r for r in results
                     if r["uses_autocast"] and not r["has_gradscaler"]]
    compile_hits = [r for r in results if r["compile_lines"]]
    api_hits = [r for r in results if r["newer_apis"]]

    lines: List[str] = []
    lines.append("# MiniMind × torch 2.1.0 / V100 兼容审计（静态扫描）")
    lines.append("")
    lines.append("- MiniMind 目录：`" + shown_root + "`")
    lines.append("- commit：`" + commit + "`（期望 `" + C.MINIMIND_COMMIT + "`，"
                 + ("一致" if commit == C.MINIMIND_COMMIT else "**不一致**") + "）")
    lines.append("- 目标 torch：`" + C.TARGET_TORCH_VERSION + "`")
    lines.append("- 扫描文件数：" + str(len(results))
                 + "（trainer/*.py 与 model/*.py，只读）")
    lines.append("")
    lines.append("## 1. 默认 `--dtype bfloat16` 的脚本")
    lines.append("")
    if bf16_scripts:
        lines.append("V100 是 sm70，`torch.cuda.is_bf16_supported()` 为 False，"
                     "`autocast(dtype=bfloat16)` 会抛 "
                     "`RuntimeError: Current CUDA Device does not support bfloat16`。"
                     "下列脚本每次运行都必须显式加 `--dtype float16`：")
        lines.append("")
        lines.append("| 文件 | --dtype 默认值 |")
        lines.append("| --- | --- |")
        for r in bf16_scripts:
            lines.append("| `" + r["name"] + "` | `"
                         + "`, `".join(r["dtype_defaults"]) + "` |")
    else:
        lines.append("没有扫到默认 `bfloat16` 的 `--dtype` 参数。")
    lines.append("")
    lines.append("## 2. GradScaler 有无")
    lines.append("")
    lines.append("| 文件 | autocast | GradScaler | scale | unscale_ | step | update |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        s = r["scaler"]
        lines.append("| `%s` | %s | %s | %d | %d | %d | %d |" % (
            r["name"], "有" if r["uses_autocast"] else "无",
            "有" if r["has_gradscaler"] else "**无**",
            s["scaler.scale"], s["scaler.unscale_"], s["scaler.step"],
            s["scaler.update"]))
    lines.append("")
    if amp_no_scaler:
        lines.append("**用了 autocast 却没有 GradScaler 的文件**："
                     + "、".join("`" + r["name"] + "`" for r in amp_no_scaler)
                     + "。在 V100 上只能跑 fp16，没有 scaler 时小梯度会下溢成 0，"
                       "loss 曲线看着在动但学不到东西；本周在 lab 侧补 scaler。")
    else:
        lines.append("所有用 autocast 的文件都带 GradScaler。")
    lines.append("")
    lines.append("## 3. `torch.compile` 使用点")
    lines.append("")
    if compile_hits:
        lines.append("| 文件 | 行号 |")
        lines.append("| --- | --- |")
        for r in compile_hits:
            lines.append("| `" + r["name"] + "` | "
                         + ", ".join(str(n) for n in r["compile_lines"]) + " |")
        lines.append("")
        lines.append("Triton 2.1 声称支持 CC>=7.0，V100 满足但公司机器未实测；"
                     "本周一律 `use_compile=0`，先把这些点记下来。")
    else:
        lines.append("没有扫到 `torch.compile(` 调用。")
    lines.append("")
    lines.append("## 4. torch 2.1 中不存在的 API")
    lines.append("")
    if api_hits:
        lines.append("| 文件 | 符号 | 最早版本 | 行号 | 说明 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in api_hits:
            for hit in r["newer_apis"]:
                lines.append("| `%s` | `%s` | %s | %s | %s |" % (
                    r["name"], hit["symbol"], hit["since"],
                    ", ".join(str(n) for n in hit["lines"]), hit["note"]))
    else:
        lines.append("在内置符号清单范围内，没有扫到 torch 2.1 之后才有的 API。"
                     "清单见脚本里的 `NEWER_APIS`（共 "
                     + str(len(NEWER_APIS)) + " 条），它不是穷举，"
                     "只覆盖本周会碰到的分布式/AMP/SDPA 面。")
    lines.append("")
    lines.append("## 5. 逐文件明细")
    lines.append("")
    lines.append("| 文件 | 行数 | 提到 bfloat16 | --dtype 默认 | 新 API 命中 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in results:
        lines.append("| `%s` | %d | %s | %s | %d |" % (
            r["name"], r["loc"], "是" if r["mentions_bfloat16"] else "否",
            ("`" + "`, `".join(r["dtype_defaults"]) + "`")
            if r["dtype_defaults"] else "—",
            len(r["newer_apis"])))
    lines.append("")
    lines.append("## 6. 结论与本周动作")
    lines.append("")
    lines.append("1. 所有 MiniMind 官方脚本在公司机器上运行时显式加 `--dtype float16`；"
                 "lab 侧的入口由 `mm_dist.common.assert_dtype_supported` 在 "
                 "autocast 之前就拒绝 bfloat16。")
    lines.append("2. 缺 GradScaler 的脚本不直接用于 fp16 训练；本周的对照训练走 "
                 "`mm_dist.train_ddp` / `mm_dist.train_fsdp`，两者都带 scaler 与跳步计数。")
    lines.append("3. `use_compile=0`（本周不开 torch.compile）。")
    lines.append("4. 本审计是静态扫描，**不是**运行验证；"
                 "运行侧证据由 `lab/scripts/probe_company.py` 的 probe.json 提供。")
    lines.append("5. 本文件与 probe.json 都只写到 `--out` 指定的路径；"
                 "公司环境请写公司内部路径，不要外传。")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="静态扫描 MiniMind 的 torch 2.1 / V100 兼容性（只读）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--minimind-root", type=str, default=None,
                   help="MiniMind 根目录，默认读环境变量 MINIMIND_ROOT")
    p.add_argument("--out", type=str, default="./compat_audit.md",
                   help="Markdown 输出路径")
    p.add_argument("--redact-root", dest="redact_root", action="store_true",
                   default=True, help="报告里只写目录名不写绝对路径（默认开启）")
    p.add_argument("--no-redact-root", dest="redact_root", action="store_false")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = C.resolve_minimind_root(args.minimind_root)
    files = iter_target_files(root)
    if not files:
        raise SystemExit("在 " + root + " 下没有找到 trainer/*.py 或 model/*.py")
    results = [scan_file(path) for path in files]
    md = render_markdown(root, results, bool(args.redact_root))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    bf16 = sum(1 for r in results if r["defaults_bfloat16"])
    no_scaler = sum(1 for r in results
                    if r["uses_autocast"] and not r["has_gradscaler"])
    api = sum(len(r["newer_apis"]) for r in results)
    print("[audit] 文件=%d  默认bfloat16=%d  autocast无scaler=%d  新API命中=%d"
          % (len(results), bf16, no_scaler, api), flush=True)
    print("[audit] -> " + os.path.abspath(args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
