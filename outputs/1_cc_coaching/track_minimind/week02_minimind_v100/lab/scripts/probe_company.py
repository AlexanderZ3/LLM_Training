"""公司机器探针：把"这台机器到底能做什么"变成一个 probe.json。

设计原则：只输出可以带出公司的抽象事实
--------------------------------------
* **输出**：版本号、卡数、每卡型号名与 compute capability、bf16 是否支持、
  SDPA 三个后端能不能真的跑起来、NCCL 版本、MiniMind commit。
* **不输出**：主机名、用户名、绝对路径、GPU UUID/序列号、`nvidia-smi topo -m`
  的原始矩阵、任何数据/模型路径。拓扑只以"连接类型计数直方图 + 是否全对称"
  的形式出现——它足以回答"EP 分组会不会踩到慢链路"，又不泄露机器结构。
* ``--redact`` 默认开启。要看原始拓扑矩阵只能在公司机器上直接跑 ``nvidia-smi topo -m``，
  本脚本任何时候都不会把矩阵写进 JSON。

SDPA 探测为什么要真跑一遍
-------------------------
``torch.backends.cuda.flash_sdp_enabled()`` 只说"这个后端有没有被禁用"，
不说"这张卡能不能用"。V100 是 sm70，torch 2.1 的 flash 后端只覆盖 sm75–sm90，
所以 enabled 会返回 True 而真正调用时静默回退或报错。本脚本在
``sdp_kernel(enable_flash=True, ...)`` 里真的算一次小的 attention，用成功/异常
来判定——这才是 `已确认` 级别的证据。

用法::

    python lab/scripts/probe_company.py --out <公司内路径>/probe.json
    python lab/scripts/probe_company.py --out ./probe.json --minimind-root /path/to/minimind
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mm_dist import common as C  # noqa: E402

# nvidia-smi topo -m 里出现的连接类型
TOPO_TOKENS = ("X", "NV1", "NV2", "NV3", "NV4", "NV6", "NV8", "NV12",
               "PIX", "PXB", "PHB", "NODE", "SYS")


# ---------------------------------------------------------------------------
# 版本
# ---------------------------------------------------------------------------


def version_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "platform_release": platform.release(),
        "torch": torch.__version__,
        "target_torch": C.TARGET_TORCH_VERSION,
        "torch_matches_target": torch.__version__.split("+")[0]
                                == C.TARGET_TORCH_VERSION,
        "torch_cuda": torch.version.cuda,
        "torch_git": getattr(torch.version, "git_version", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn": (torch.backends.cudnn.version()
                  if torch.backends.cudnn.is_available() else None),
    }
    try:
        info["nccl"] = ".".join(str(v) for v in torch.cuda.nccl.version())
    except (AttributeError, RuntimeError, OSError):
        info["nccl"] = None
    for pkg in ("transformers", "datasets", "accelerate", "numpy"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = None
    return info


def arch_info() -> Dict[str, Any]:
    """编译期支持的架构列表；sm_70 不在里面就说明这份 wheel 根本不支持 V100。"""
    try:
        arch_list = list(torch.cuda.get_arch_list())
    except (AttributeError, RuntimeError):
        arch_list = []
    return {
        "arch_list": arch_list,
        "has_sm_70": any("70" in a for a in arch_list),
        "has_sm_75": any("75" in a for a in arch_list),
        "has_sm_80": any("80" in a for a in arch_list),
    }


def device_info() -> Dict[str, Any]:
    """每卡型号名 + capability + 显存总量（GB，取整），不含 UUID/序列号/PCI 地址。"""
    if not torch.cuda.is_available():
        return {"device_count": 0, "devices": [], "bf16_supported": False,
                "note": "本机没有可用的 CUDA 设备；GPU 相关结论全部为未知。"}
    devices: List[Dict[str, Any]] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        devices.append({
            "index": i,
            "name": props.name,
            "capability": str(cap[0]) + "." + str(cap[1]),
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 1),
            "multi_processor_count": props.multi_processor_count,
        })
    try:
        bf16 = bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        bf16 = False
    names = {d["name"] for d in devices}
    caps = {d["capability"] for d in devices}
    return {
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "homogeneous": len(names) == 1 and len(caps) == 1,
        "bf16_supported": bf16,
        "bf16_note": ("V100 是 sm70，torch 的 is_bf16_supported() 要求 major>=8；"
                      "为 False 时本 lab 所有命令必须用 --dtype float16 + GradScaler。"),
    }


def sdpa_info() -> Dict[str, Any]:
    """真跑一次小 attention，判定三个 SDPA 后端在本机能不能用。"""
    result: Dict[str, Any] = {
        "flags": {
            "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_enabled": bool(
                torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        },
        "functional": {},
        "note": "flags 只说后端有没有被禁用；functional 是真跑一次的结果。",
    }
    if not torch.cuda.is_available():
        result["functional"] = {"flash": None, "mem_efficient": None, "math": None}
        result["functional_note"] = "无 CUDA，三个后端都无法判定（未知，不是不支持）。"
        return result
    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        result["functional"] = {"flash": None, "mem_efficient": None, "math": None}
        result["functional_note"] = (
            "本机 torch " + torch.__version__ + " 没有 torch.backends.cuda.sdp_kernel；"
            "目标机 torch 2.1.0 有该 API，请在目标机上重跑本探针。")
        return result
    combos = {
        "flash": dict(enable_flash=True, enable_mem_efficient=False,
                      enable_math=False),
        "mem_efficient": dict(enable_flash=False, enable_mem_efficient=True,
                              enable_math=False),
        "math": dict(enable_flash=False, enable_mem_efficient=False,
                     enable_math=True),
    }
    q = torch.randn(1, 8, 128, 96, device="cuda", dtype=torch.float16)
    for name, flags in combos.items():
        try:
            with torch.backends.cuda.sdp_kernel(**flags):
                out = F.scaled_dot_product_attention(q, q, q, is_causal=True)
            result["functional"][name] = bool(torch.isfinite(out).all())
        except (RuntimeError, ValueError) as exc:
            result["functional"][name] = False
            result.setdefault("errors", {})[name] = type(exc).__name__ + ": " \
                + str(exc).splitlines()[0][:200]
    return result


# ---------------------------------------------------------------------------
# 拓扑（抽象化）
# ---------------------------------------------------------------------------


def parse_topo_matrix(text: str, device_count: int) -> Dict[str, Any]:
    """把 ``nvidia-smi topo -m`` 解析成"连接类型计数直方图 + 是否全对称"。

    只保留 GPU×GPU 上三角（不含对角线）的连接类型计数。不返回矩阵、不返回行列标签。
    """
    hist: Dict[str, int] = {}
    rows: List[List[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not re.match(r"^GPU\d+\s", stripped):
            continue
        cells = stripped.split()[1:]
        entries = [c for c in cells if c in TOPO_TOKENS]
        if entries:
            rows.append(entries)
    n = min(len(rows), device_count if device_count > 0 else len(rows))
    for i in range(n):
        for j in range(i + 1, min(len(rows[i]), n)):
            token = rows[i][j]
            hist[token] = hist.get(token, 0) + 1
    distinct = [k for k in hist if k != "X"]
    return {
        "parsed_gpu_rows": n,
        "link_type_histogram": dict(sorted(hist.items())),
        "distinct_link_types": sorted(distinct),
        "fully_symmetric": len(distinct) <= 1,
        "note": ("只输出连接类型计数与是否全对称；矩阵本身不出公司。"
                 "distinct_link_types 多于 1 种时，EP 分组要避免跨慢链路。"),
    }


def topology_info(redact: bool = True) -> Dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False,
                "reason": "找不到 nvidia-smi；本机无 NVIDIA 驱动，拓扑未知。"}
    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                             text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": type(exc).__name__ + ": " + str(exc)}
    if out.returncode != 0:
        return {"available": False,
                "reason": "nvidia-smi topo -m 退出码 " + str(out.returncode)}
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    parsed = parse_topo_matrix(out.stdout, count)
    parsed["available"] = True
    parsed["redacted"] = bool(redact)
    if not redact:
        parsed["raw_withheld"] = (
            "--no-redact 只影响本地打印；原始矩阵在任何情况下都不会写入 probe.json。"
            "需要看矩阵请在公司机器上直接运行 nvidia-smi topo -m。")
    return parsed


# ---------------------------------------------------------------------------
# MiniMind
# ---------------------------------------------------------------------------


def minimind_info(root_arg: Optional[str], redact: bool) -> Dict[str, Any]:
    try:
        root = C.resolve_minimind_root(root_arg)
    except RuntimeError as exc:
        return {"found": False, "reason": str(exc).splitlines()[0],
                "expected_commit": C.MINIMIND_COMMIT}
    commit = C.minimind_commit(root)
    info: Dict[str, Any] = {
        "found": True,
        "commit": commit,
        "expected_commit": C.MINIMIND_COMMIT,
        "commit_matches": commit == C.MINIMIND_COMMIT,
        "has_trainer_dir": os.path.isdir(os.path.join(root, "trainer")),
        "has_model_file": os.path.isfile(
            os.path.join(root, "model", "model_minimind.py")),
    }
    info["root"] = os.path.basename(root) if redact else root
    return info


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def collect(args: argparse.Namespace) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema": "mm_dist.probe_company/1",
        "redacted": bool(args.redact),
        "versions": version_info(),
        "arch": arch_info(),
        "devices": device_info(),
        "sdpa": sdpa_info(),
        "topology": topology_info(redact=bool(args.redact)),
        "minimind": minimind_info(args.minimind_root, bool(args.redact)),
    }
    if not args.redact:
        report["host"] = {"node": platform.node()}
    checks: List[Dict[str, Any]] = [
        {"id": "torch_version",
         "ok": report["versions"]["torch_matches_target"],
         "detail": "torch " + report["versions"]["torch"] + " vs 目标 "
                   + C.TARGET_TORCH_VERSION},
        {"id": "sm70_in_arch_list", "ok": report["arch"]["has_sm_70"],
         "detail": "get_arch_list()=" + json.dumps(report["arch"]["arch_list"])},
        {"id": "bf16_unsupported_as_expected",
         "ok": not report["devices"].get("bf16_supported", False),
         "detail": "is_bf16_supported()="
                   + str(report["devices"].get("bf16_supported"))
                   + "；V100 上应为 False，本 lab 一律 fp16"},
        {"id": "eight_gpus",
         "ok": report["devices"].get("device_count", 0) == 8,
         "detail": "device_count=" + str(report["devices"].get("device_count"))},
        {"id": "homogeneous_gpus",
         "ok": bool(report["devices"].get("homogeneous", False)),
         "detail": "所有卡型号与 capability 是否一致"},
        {"id": "minimind_commit",
         "ok": bool(report["minimind"].get("commit_matches", False)),
         "detail": "期望 " + C.MINIMIND_COMMIT},
    ]
    report["checks"] = checks
    report["all_checks_pass"] = all(c["ok"] for c in checks)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="公司 8xV100 环境探针（输出抽象化的 probe.json）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", type=str, default="./probe.json",
                   help="probe.json 输出路径（公司环境请写公司内部路径）")
    p.add_argument("--minimind-root", type=str, default=None,
                   help="MiniMind 根目录，默认读环境变量 MINIMIND_ROOT")
    p.add_argument("--redact", dest="redact", action="store_true", default=True,
                   help="抹掉主机名与绝对路径（默认开启）")
    p.add_argument("--no-redact", dest="redact", action="store_false",
                   help="保留主机名与绝对路径；拓扑矩阵仍然不会写入")
    p.add_argument("--print", dest="do_print", action="store_true", default=True,
                   help="同时把 JSON 打到 stdout")
    p.add_argument("--quiet", dest="do_print", action="store_false")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect(args)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
    if args.do_print:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
              flush=True)
    print("", flush=True)
    for check in report["checks"]:
        print("[%s] %-28s %s" % ("PASS" if check["ok"] else "FAIL",
                                 check["id"], check["detail"]), flush=True)
    print("[probe] -> " + os.path.abspath(args.out), flush=True)
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
