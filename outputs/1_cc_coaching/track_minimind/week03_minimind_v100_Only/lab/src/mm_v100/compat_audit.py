"""Day 0 静态审计：MiniMind 的源码在 torch 2.1.0 + V100 上会踩什么。

来源
----
从 week02 lab/scripts/audit_minimind_compat.py 复制并裁剪：保留四类发现与
NEWER_APIS 清单，去掉 MoE / EP 相关条目，输出改成「四个计数 + 明细」的结构化
字典，让 make_evidence.py 可以直接取那四个数。

只读，不改动 MiniMind 一个字节
------------------------------
用正则扫源码，不 import、不执行、不写回。原因是 import 会触发 MiniMind 的
依赖链（transformers / datasets），而公司内网里这条链能不能装上本身就是未知数；
静态扫描在任何环境都能跑。

四个计数
--------
n_default_bfloat16   --dtype 默认值是 bfloat16 的脚本数。V100 上这些脚本每次
                     运行都必须显式 --dtype float16，否则 autocast 直接抛
                     RuntimeError: Current CUDA Device does not support bfloat16。
n_autocast_no_scaler 用了 autocast 但文件里没有 GradScaler 的处数。fp16 训练
                     缺 GradScaler 时梯度会下溢到 0，loss 看着在动其实没学到
                     东西——这是 Day 2 故障 B 的现实版本。
n_torch_compile      torch.compile 调用点数。本周一律关闭（Triton 2.1 在 Volta
                     上未实测，属纯风险项）。
n_newer_api_hits     torch 2.1 里不存在的 API 命中次数。命中即说明该文件是按更新
                     的 torch 写的，在 2.1.0 上会 AttributeError / ImportError。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

# (符号, 正则, 最早出现的 torch 版本, 说明)
NEWER_APIS: List[Tuple[str, str, str, str]] = [
    ("torch.distributed.tensor",
     r"torch\.distributed\.tensor|from\s+torch\.distributed\.tensor",
     "2.5", "DTensor 的公开入口；2.1 只有私有的 torch.distributed._tensor"),
    ("fully_shard", r"\bfully_shard\b", "2.4",
     "FSDP2 的 per-parameter 分片入口；2.1 只有 FullyShardedDataParallel"),
    ("DTensor", r"\bDTensor\b", "2.5", "分布式张量类型；2.1 无公开 DTensor"),
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
     r"torch\.distributed\.checkpoint\.state_dict"
     r"|from\s+torch\.distributed\.checkpoint\.state_dict",
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

DEFAULT_SUBDIRS = ("trainer", "model", "dataset")


def iter_target_files(root: str,
                      subdirs: Tuple[str, ...] = DEFAULT_SUBDIRS) -> List[str]:
    """要扫的 .py 文件，按路径排序，保证两台机器上的输出可以逐行 diff。"""
    files: List[str] = []
    for sub in subdirs:
        directory = os.path.join(root, sub)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.endswith(".py"):
                files.append(os.path.join(directory, name))
    return files


def line_numbers(text: str, pattern: re.Pattern) -> List[int]:
    return [text.count("\n", 0, m.start()) + 1 for m in pattern.finditer(text)]


def scan_text(text: str, name: str) -> Dict[str, Any]:
    """扫一份源码文本。单独抽出来是为了让单测可以喂合成样本，不依赖真仓库。"""
    dtype_defaults = DTYPE_DEFAULT_RE.findall(text)
    scaler_hits = {key: len(pat.findall(text))
                   for key, pat in SCALER_PATTERNS.items()}
    newer: List[Dict[str, Any]] = []
    for symbol, pattern, since, note in NEWER_APIS:
        lines = line_numbers(text, re.compile(pattern))
        if lines:
            newer.append({"symbol": symbol, "since": since, "note": note,
                          "lines": lines})
    autocast_lines = line_numbers(text, AUTOCAST_RE)
    return {
        "name": name,
        "loc": text.count("\n") + 1,
        "dtype_defaults": dtype_defaults,
        "defaults_bfloat16": any(d == "bfloat16" for d in dtype_defaults),
        "mentions_bfloat16": bool(BF16_RE.search(text)),
        "scaler": scaler_hits,
        "has_gradscaler": scaler_hits["GradScaler"] > 0,
        "autocast_lines": autocast_lines,
        "uses_autocast": bool(autocast_lines),
        "compile_lines": line_numbers(text, COMPILE_RE),
        "newer_apis": newer,
    }


def scan_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    name = (os.path.basename(os.path.dirname(path)) + "/"
            + os.path.basename(path))
    result = scan_text(text, name)
    result["path"] = path
    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """四个计数 + 触发它们的文件名。证据字段直接取这四个数。"""
    bf16_files = [r["name"] for r in results if r["defaults_bfloat16"]]
    no_scaler = [r for r in results
                 if r["uses_autocast"] and not r["has_gradscaler"]]
    compile_files = [r for r in results if r["compile_lines"]]
    api_hits = [r for r in results if r["newer_apis"]]
    return {
        "n_files_scanned": len(results),
        "n_default_bfloat16": len(bf16_files),
        "files_default_bfloat16": bf16_files,
        "n_autocast_no_scaler": sum(len(r["autocast_lines"]) for r in no_scaler),
        "files_autocast_no_scaler": [r["name"] for r in no_scaler],
        "n_torch_compile": sum(len(r["compile_lines"]) for r in compile_files),
        "files_torch_compile": [r["name"] for r in compile_files],
        "n_newer_api_hits": sum(len(a["lines"]) for r in api_hits
                                for a in r["newer_apis"]),
        "files_newer_api": [r["name"] for r in api_hits],
        "newer_api_symbols": sorted({a["symbol"] for r in api_hits
                                     for a in r["newer_apis"]}),
    }


def audit(root: str, subdirs: Tuple[str, ...] = DEFAULT_SUBDIRS) -> Dict[str, Any]:
    files = iter_target_files(root, subdirs)
    if not files:
        raise FileNotFoundError(
            "在 " + root + " 下的 " + "/".join(subdirs) + " 里一个 .py 都没找到。\n"
            "下一步：确认 MINIMIND_ROOT 指的是 git clone 出来的仓库根目录本身，"
            "而不是它的父目录。"
        )
    results = [scan_file(p) for p in files]
    summary = summarize(results)
    return {"schema": "mm_v100.compat_audit/1", "summary": summary,
            "files": results}


def render_markdown(report: Dict[str, Any], root_label: str) -> str:
    """输出给 Day 0 证据用的 Markdown。只含文件名与行号，不含源码片段。"""
    s = report["summary"]
    results = report["files"]
    lines: List[str] = []
    lines.append("# MiniMind x torch 2.1.0 / V100 兼容审计（静态扫描）")
    lines.append("")
    lines.append("- 扫描目录：`" + root_label + "`")
    lines.append("- 扫描文件数：" + str(s["n_files_scanned"]) + "（只读）")
    lines.append("")
    lines.append("| 计数 | 值 |")
    lines.append("| --- | --- |")
    lines.append("| `n_default_bfloat16` | " + str(s["n_default_bfloat16"]) + " |")
    lines.append("| `n_autocast_no_scaler` | " + str(s["n_autocast_no_scaler"]) + " |")
    lines.append("| `n_torch_compile` | " + str(s["n_torch_compile"]) + " |")
    lines.append("| `n_newer_api_hits` | " + str(s["n_newer_api_hits"]) + " |")
    lines.append("")
    lines.append("## 1. 默认 --dtype bfloat16 的脚本")
    lines.append("")
    if s["files_default_bfloat16"]:
        lines.append("这些脚本每次运行都必须显式加 `--dtype float16`：")
        lines.append("")
        for name in s["files_default_bfloat16"]:
            lines.append("- `" + name + "`")
    else:
        lines.append("没有扫到默认 bfloat16 的 --dtype 参数。")
    lines.append("")
    lines.append("## 2. 用了 autocast 但没有 GradScaler")
    lines.append("")
    if s["files_autocast_no_scaler"]:
        for name in s["files_autocast_no_scaler"]:
            hit = next(r for r in results if r["name"] == name)
            lines.append("- `" + name + "`：autocast 出现在第 "
                         + ", ".join(str(n) for n in hit["autocast_lines"]) + " 行")
    else:
        lines.append("没有这类文件。")
    lines.append("")
    lines.append("## 3. torch.compile 调用点")
    lines.append("")
    if s["files_torch_compile"]:
        for name in s["files_torch_compile"]:
            hit = next(r for r in results if r["name"] == name)
            lines.append("- `" + name + "`：第 "
                         + ", ".join(str(n) for n in hit["compile_lines"]) + " 行")
    else:
        lines.append("没有 torch.compile 调用点。")
    lines.append("")
    lines.append("## 4. torch 2.1 里不存在的 API")
    lines.append("")
    if s["files_newer_api"]:
        lines.append("| 文件 | 符号 | 最早版本 | 行号 |")
        lines.append("| --- | --- | --- | --- |")
        for r in results:
            for api in r["newer_apis"]:
                lines.append("| `" + r["name"] + "` | `" + api["symbol"]
                             + "` | " + api["since"] + " | "
                             + ", ".join(str(n) for n in api["lines"]) + " |")
    else:
        lines.append("没有命中。")
    lines.append("")
    return "\n".join(lines)


def to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
