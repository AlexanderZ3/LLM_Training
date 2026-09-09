"""Day 0 探针：把"这台 V100 到底能做什么"变成一份可带出公司的 JSON。

来源
----
从 week02 lab/scripts/probe_company.py 复制并裁剪：去掉了 nvidia-smi 拓扑解析
（本周不做 EP 分组，拓扑对结论没有影响，少一个泄密面），新增 int8 能力探测、
GradScaler 默认值读取、torch.distributed.checkpoint 的 API 名探测、
GPU 独占性判断和磁盘余量档位。

只输出可以带出公司的抽象事实
----------------------------
输出：版本号、卡数、型号名与 compute capability、bf16 是否支持、SDPA 三后端
能不能真跑、NCCL 版本、GradScaler 默认值、DCP API 名、GPU 是否被别人占着、
可写目录剩余空间的档位。

不输出：主机名、用户名、绝对路径、GPU UUID / 序列号、拓扑矩阵、任何数据或
模型路径。剩余空间只给档位（<10 / 10-50 / 50-200 / >200 GB），不给精确数值。

为什么 SDPA 要真跑一遍
----------------------
torch.backends.cuda.flash_sdp_enabled() 只说"这个后端有没有被禁用"，不说
"这张卡能不能用"。V100 是 sm70，torch 2.1 的 flash 后端只覆盖 sm75-sm90，
所以 enabled 返回 True 而真正调用时会报错或静默回退。这里在 sdp_kernel 里
真的算一次小 attention，用成功/异常判定——这才是"已确认"级别的证据。

为什么 int8 探测必须 try/except 且不阻塞
---------------------------------------
V100 的 Tensor Core 只吃 FP16，INT8 的 IMMA 指令要 Turing(sm75) 才有。
torch._int_mm 在 sm70 上大概率抛 RuntimeError，那是**预期结果**，不是故障。
这一条只是把"在这台机器上 8-bit 不会更快"从推断变成已确认，所以它必须
记录异常类型而不是让整个探针失败。
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from . import common as C

SCHEMA = "mm_v100.probe/1"

# 剩余空间只报档位，不报精确值——精确值属于公司机器的容量信息。
SPACE_TIERS = ((10.0, "<10GB"), (50.0, "10-50GB"), (200.0, "50-200GB"))


def space_tier(free_gb: float) -> str:
    for limit, label in SPACE_TIERS:
        if free_gb < limit:
            return label
    return ">200GB"


def version_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "torch": torch.__version__,
        "target_torch": C.TARGET_TORCH_VERSION,
        "torch_matches_target": (torch.__version__.split("+")[0]
                                 == C.TARGET_TORCH_VERSION),
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn": (torch.backends.cudnn.version()
                  if torch.backends.cudnn.is_available() else None),
    }
    try:
        info["nccl"] = ".".join(str(v) for v in torch.cuda.nccl.version())
    except (AttributeError, RuntimeError, OSError):
        info["nccl"] = None
    for pkg in ("numpy", "transformers", "datasets"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = None
    try:
        arch = list(torch.cuda.get_arch_list())
    except (AttributeError, RuntimeError):
        arch = []
    info["arch_list"] = arch
    info["has_sm_70"] = any("70" in a for a in arch)
    return info


def device_info() -> Dict[str, Any]:
    """每卡型号 + capability + 显存总量（GB 取整）。不含 UUID / PCI 地址。"""
    if not torch.cuda.is_available():
        return {
            "device_count": 0,
            "devices": [],
            "bf16_supported": False,
            "bf16_known": False,
            "note": "本机没有可用的 CUDA 设备；GPU 相关结论全部为未知，不是 False。",
        }
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
        "bf16_known": True,
        "bf16_note": ("V100 是 sm70，is_bf16_supported() 要求 major>=8；"
                      "为 False 时本 lab 所有命令必须 --dtype float16 + GradScaler。"),
    }


def gpu_exclusive_info() -> Dict[str, Any]:
    """判断这几张卡是不是只有我在用。非独占时所有时间类结论都不可信。

    做法：读 nvidia-smi 的 used memory 与 compute app 数量。取不到就报未知，
    绝不猜。输出里没有进程名、没有 PID、没有用户名。
    """
    if shutil.which("nvidia-smi") is None:
        return {"known": False,
                "reason": "找不到 nvidia-smi；独占性未知，不能当成独占。"}
    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False)
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"known": False, "reason": type(exc).__name__ + ": " + str(exc)[:120]}
    if used.returncode != 0:
        return {"known": False,
                "reason": "nvidia-smi 退出码 " + str(used.returncode)}
    rows = [r.strip() for r in used.stdout.splitlines() if r.strip()]
    mem_used: List[int] = []
    util: List[int] = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            mem_used.append(int(parts[0]))
            util.append(int(parts[1]))
    n_apps = len([r for r in apps.stdout.splitlines() if r.strip()])
    busy = [i for i, m in enumerate(mem_used) if m > 512]
    return {
        "known": True,
        "n_gpus_reporting": len(mem_used),
        "n_compute_apps": n_apps,
        "n_gpus_with_memory_in_use": len(busy),
        "max_util_percent": max(util) if util else None,
        "looks_exclusive": bool(n_apps == 0 and not busy),
        "note": ("looks_exclusive=False 时，Day 3 的 step_time_ratio 一律标为 "
                 "measurement 类不确定，只保留 mem_ratio 与 collectives_total 作结论。"),
    }


def sdpa_info() -> Dict[str, Any]:
    """真跑一次小 attention，判定三个 SDPA 后端在本机能不能用。"""
    result: Dict[str, Any] = {
        "flags": {},
        "functional": {"flash": None, "mem_efficient": None, "math": None},
        "note": "flags 只说后端有没有被禁用；functional 是真跑一次的结果。",
    }
    for name, getter in (("flash", "flash_sdp_enabled"),
                         ("mem_efficient", "mem_efficient_sdp_enabled"),
                         ("math", "math_sdp_enabled")):
        fn = getattr(torch.backends.cuda, getter, None)
        result["flags"][name] = bool(fn()) if callable(fn) else None
    if not torch.cuda.is_available():
        result["functional_note"] = "无 CUDA，三个后端都无法判定（未知，不是不支持）。"
        return result
    if not hasattr(torch.backends.cuda, "sdp_kernel"):
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
            result.setdefault("errors", {})[name] = (
                type(exc).__name__ + ": " + str(exc).splitlines()[0][:200])
    return result


def int8_info() -> Dict[str, Any]:
    """torch._int_mm 能不能在这张卡上跑。失败是预期结果，只记录不阻塞。"""
    info: Dict[str, Any] = {
        "has_torch_int_mm": hasattr(torch, "_int_mm"),
        "callable_on_device": None,
        "note": ("V100 的 Tensor Core 只接受 FP16 输入，INT8 的 IMMA 指令从 "
                 "Turing(sm75) 才有。这里失败是预期结果，作用是把「8-bit 在这台"
                 "机器上不会更快」从推断升级成已确认。"),
    }
    if not hasattr(torch, "_int_mm"):
        info["callable_on_device"] = False
        info["reason"] = "本 torch 没有 torch._int_mm"
        return info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info["probe_device"] = device
    a = torch.randint(-8, 8, (32, 64), dtype=torch.int8, device=device)
    b = torch.randint(-8, 8, (64, 32), dtype=torch.int8, device=device)
    try:
        out = torch._int_mm(a, b)
        info["callable_on_device"] = True
        info["out_dtype"] = str(out.dtype)
    except (RuntimeError, TypeError) as exc:
        # 只写 RuntimeError 就够：torch 在不支持的设备上抛的
        # "not implemented" 类异常是 RuntimeError 的子类，会被这里接住。
        info["callable_on_device"] = False
        info["reason"] = type(exc).__name__ + ": " + str(exc).splitlines()[0][:200]
    return info


def grad_scaler_defaults() -> Dict[str, Any]:
    """读官方 GradScaler 的默认值，和 common.py 里写死的四个常数核对。"""
    info: Dict[str, Any] = {
        "expected": {
            "init_scale": C.DEFAULT_INIT_SCALE,
            "growth_factor": C.DEFAULT_GROWTH_FACTOR,
            "backoff_factor": C.DEFAULT_BACKOFF_FACTOR,
            "growth_interval": C.DEFAULT_GROWTH_INTERVAL,
        },
        "actual": {},
        "source": None,
    }
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = C.new_torch_scaler(device_type, torch.cuda.is_available())
    if scaler is None:
        info["source"] = "unavailable"
        info["note"] = ("这台机器上构造不出官方 GradScaler（torch 2.1 的 "
                        "GradScaler 只服务 CUDA）；lab 用 PurePythonScaler 顶上。")
        return info
    info["source"] = type(scaler).__module__ + "." + type(scaler).__name__
    for attr, key in (("_init_scale", "init_scale"),
                      ("_growth_factor", "growth_factor"),
                      ("_backoff_factor", "backoff_factor"),
                      ("_growth_interval", "growth_interval")):
        value = getattr(scaler, attr, None)
        info["actual"][key] = float(value) if value is not None else None
    info["matches_expected"] = all(
        info["actual"].get(k) == v for k, v in info["expected"].items()
        if info["actual"].get(k) is not None)
    return info


def dcp_api_info() -> Dict[str, Any]:
    """torch.distributed.checkpoint 在这台机器上叫什么名字。

    torch 2.1 的公开入口是 save_state_dict / load_state_dict；2.2 之后改推
    save / load，旧名标 deprecated。ckpt_reshard.py 两种都试，这里先把事实记下来，
    免得 Day 3 才发现 API 名对不上。
    """
    try:
        import torch.distributed.checkpoint as dcp
    except ImportError as exc:
        return {"importable": False, "reason": str(exc)[:200]}
    return {
        "importable": True,
        "has_save_state_dict": hasattr(dcp, "save_state_dict"),
        "has_load_state_dict": hasattr(dcp, "load_state_dict"),
        "has_save": hasattr(dcp, "save"),
        "has_load": hasattr(dcp, "load"),
        "has_FileSystemWriter": hasattr(dcp, "FileSystemWriter"),
        "has_FileSystemReader": hasattr(dcp, "FileSystemReader"),
    }


def fsdp_api_info() -> Dict[str, Any]:
    """本周要用到的 FSDP1 符号是否齐全。缺任何一个，Day 3 都跑不起来。"""
    names = ["FullyShardedDataParallel", "ShardingStrategy", "MixedPrecision",
             "StateDictType", "FullStateDictConfig", "ShardedStateDictConfig",
             "FullOptimStateDictConfig", "ShardedOptimStateDictConfig"]
    try:
        import torch.distributed.fsdp as fsdp
    except ImportError as exc:
        return {"importable": False, "reason": str(exc)[:200]}
    out: Dict[str, Any] = {"importable": True}
    for n in names:
        out[n] = hasattr(fsdp, n)
    out["all_present"] = all(out.get(n, False) for n in names)
    return out


def disk_info(runs_dir: Optional[str] = None) -> Dict[str, Any]:
    """可写目录的剩余空间档位。只报档位，不报绝对路径与精确数字。"""
    target = runs_dir or os.getcwd()
    probe = target
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    try:
        free_gb = shutil.disk_usage(probe).free / (1024 ** 3)
    except OSError as exc:
        return {"known": False, "reason": type(exc).__name__ + ": " + str(exc)[:120]}
    return {
        "known": True,
        "tier": space_tier(free_gb),
        "enough_for_day3": free_gb >= 20.0,
        "note": "只报档位；精确容量属于公司机器信息，不带出。",
    }


def collect(runs_dir: Optional[str] = None,
            minimind_root: Optional[str] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "versions": version_info(),
        "devices": device_info(),
        "gpu_exclusive": gpu_exclusive_info(),
        "sdpa": sdpa_info(),
        "int8": int8_info(),
        "grad_scaler": grad_scaler_defaults(),
        "dcp_api": dcp_api_info(),
        "fsdp_api": fsdp_api_info(),
        "disk": disk_info(runs_dir),
        "minimind_root_set": bool(minimind_root),
    }
    devices = report["devices"]
    versions = report["versions"]
    checks: List[Dict[str, Any]] = [
        {"id": "torch_version_matches_2.1.0",
         "ok": bool(versions["torch_matches_target"]),
         "detail": "torch " + versions["torch"] + " vs 目标 "
                   + C.TARGET_TORCH_VERSION},
        {"id": "sm70_in_arch_list", "ok": bool(versions["has_sm_70"]),
         "detail": "get_arch_list()=" + json.dumps(versions["arch_list"])},
        {"id": "eight_gpus", "ok": devices.get("device_count", 0) == 8,
         "detail": "device_count=" + str(devices.get("device_count"))},
        {"id": "bf16_absent_as_expected",
         "ok": bool(devices.get("bf16_known") and not devices.get("bf16_supported")),
         "detail": "is_bf16_supported()=" + str(devices.get("bf16_supported"))
                   + "（V100 上应为 False；无 CUDA 时这一项判 FAIL=未知）"},
        {"id": "fsdp_api_complete", "ok": bool(report["fsdp_api"].get("all_present")),
         "detail": "torch.distributed.fsdp 的 8 个符号是否齐全"},
        {"id": "dcp_importable", "ok": bool(report["dcp_api"].get("importable")),
         "detail": "torch.distributed.checkpoint 能否 import"},
        {"id": "disk_enough_for_day3",
         "ok": bool(report["disk"].get("enough_for_day3")),
         "detail": "剩余空间档位=" + str(report["disk"].get("tier"))},
    ]
    report["checks"] = checks
    report["all_checks_pass"] = all(c["ok"] for c in checks)
    report["blocking_failures"] = [c["id"] for c in checks if not c["ok"]]
    return report


def render_text(report: Dict[str, Any]) -> str:
    """人读的一屏摘要。可以直接贴出来——里面没有路径、主机名和拓扑。"""
    lines: List[str] = []
    v = report["versions"]
    d = report["devices"]
    lines.append("== Day 0 探针 ==")
    lines.append("torch=%s (cuda=%s) python=%s" % (v["torch"], v["torch_cuda"],
                                                   v["python"]))
    lines.append("nccl=%s  arch_list=%s" % (v["nccl"], v["arch_list"]))
    lines.append("device_count=%s  bf16_supported=%s  homogeneous=%s"
                 % (d.get("device_count"), d.get("bf16_supported"),
                    d.get("homogeneous")))
    for dev in d.get("devices", []):
        lines.append("  gpu%d %s cap=%s mem=%sGB sm=%s"
                     % (dev["index"], dev["name"], dev["capability"],
                        dev["total_memory_gb"], dev["multi_processor_count"]))
    lines.append("sdpa functional=%s" % json.dumps(report["sdpa"]["functional"]))
    lines.append("int8 _int_mm callable=%s" % report["int8"]["callable_on_device"])
    lines.append("gpu_exclusive=%s" % report["gpu_exclusive"].get("looks_exclusive"))
    lines.append("disk tier=%s" % report["disk"].get("tier"))
    lines.append("")
    for check in report["checks"]:
        lines.append("[%s] %-32s %s" % ("PASS" if check["ok"] else "FAIL",
                                        check["id"], check["detail"]))
    return "\n".join(lines)
