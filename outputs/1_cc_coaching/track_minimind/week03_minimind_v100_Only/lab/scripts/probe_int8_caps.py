#!/usr/bin/env python
"""把 V100 上「INT8 / fake-quant / bitsandbytes 到底能干什么」一次探完，输出 JSON。

这是**信息收集器，不是门**。任何一项失败都不会让脚本非零退出——
失败本身就是要记录的信息（例如 convert_fx 的产物 .cuda() 失败，是**预期结果**）。
只有用法错误（参数不对、输出目录写不进去）才返回非零。

为什么每一项都要现场探
----------------------
cc 这边的所有 GPU 结论都是**开环**的：代码在一台没有 GPU 的 Windows 机器上写，
第一次运行在公司 8xV100 上。下面这些事实在 2026-09-08 做过联网审计，
但审计的是「PyTorch 源码和文档说什么」，不是「你那台机器实际怎么样」。
两者不一致时以这个脚本的输出为准。

审计结论（2026-09-08，来源见 06_QUANT_LOWBIT.md 第 1 节）：
  - V100（sm70）**没有 INT8 Tensor Core**，IMMA 从 sm75（Turing）才有。
  - torch.ao.quantization 的真 INT8 后端（fbgemm/qnnpack/x86/onednn）**只有 CPU**；
    convert_fx(...) 的产物 .cuda() 预期失败。
  - fake-quant 在 CUDA 上有完整原生 kernel（FakeQuantizeCore.cu /
    FusedObsFakeQuant.cu），前向反向都有 => **QAT 在 V100 上是真跑的**。
  - torch._int_mm 在 torch 2.1 存在（要求 CUDA >= 11.7）但**没有算力检查**，
    V100 上行为未知；且有已知正确性缺陷（pytorch#107671）。
    所以本脚本不仅调用它，还**与 fp32 参考逐元素比对**——能跑不等于算对。
  - torch._weight_int8pack_mm 在 torch 2.1 **不存在**（2.3 才有）。
  - torchao 最低要求 torch 2.5，在目标机上**不可用**。
  - bitsandbytes：LLM.int8() 需要算力 7.5（V100 用不了）；
    8-bit 优化器需要 6.0（V100 可用）；NF4/FP4 需要 6.0（V100 可用）。
    与 torch 2.1 相容的最高版本是 0.45.5。

用法
----
  python lab/scripts/probe_int8_caps.py
  python lab/scripts/probe_int8_caps.py --tag day6_quant_probe
  python lab/scripts/probe_int8_caps.py --out -            # 只打到 stdout
  python lab/scripts/probe_int8_caps.py --skip bnb,convert_fx

输出只含版本号、布尔、形状和误差数值，不含任何权重或数据内容，
可以直接贴出来。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mm_v100 import paths  # noqa: E402
from mm_v100.console import use_utf8  # noqa: E402

CHECKS: List[str] = [
    "environment",
    "arch",
    "int_mm",
    "weight_int8pack_mm",
    "fake_quant",
    "fused_obs_fake_quant",
    "convert_fx",
    "torchao",
    "bnb",
]

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"


def _guard(fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """跑一个探针，任何异常都变成一条记录，绝不向上抛。"""
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001 - 探针必须吞掉一切，包括 SystemExit
        return {
            "status": STATUS_ERROR,
            "exception": type(exc).__name__,
            "message": str(exc)[:400],
            "traceback_tail": traceback.format_exc().strip().splitlines()[-1][:300],
        }


def probe_environment() -> Dict[str, Any]:
    import torch

    d: Dict[str, Any] = {
        "status": STATUS_OK,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    try:
        import numpy

        d["numpy"] = numpy.__version__
    except Exception as exc:
        d["numpy"] = "{}: {}".format(type(exc).__name__, exc)
    if d["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        d["device_name"] = props.name
        d["capability"] = "{}.{}".format(props.major, props.minor)
        d["total_memory_gib"] = round(props.total_memory / (1024 ** 3), 2)
        d["has_int8_tensor_core"] = (props.major, props.minor) >= (7, 5)
        d["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    else:
        d["note"] = "本机无 CUDA：所有 GPU 项会记成 unavailable，这是预期的。"
    return d


def probe_arch() -> Dict[str, Any]:
    import torch

    try:
        arch = list(torch.cuda.get_arch_list())
    except Exception as exc:
        return {"status": STATUS_ERROR, "message": "{}: {}".format(type(exc).__name__, exc)}
    return {
        "status": STATUS_OK if arch else STATUS_UNAVAILABLE,
        "arch_list": arch,
        "has_sm_70": any("70" in a for a in arch),
        "has_sm_75": any("75" in a for a in arch),
        "why_it_matters": (
            "wheel 里没有 sm_70 的 cubin/PTX 时，V100 上会在第一个 kernel 就报 "
            "no kernel image is available for execution on the device。"
        ),
    }


def probe_int_mm() -> Dict[str, Any]:
    """torch._int_mm：能不能调，以及**算得对不对**。

    只「能调用」不算通过。pytorch#107671 报告过 CUDA 12.1 update1 起
    torch._int_mm 可能给出错误结果，所以这里把 int8 输入同时用 fp32 算一遍，
    整数结果必须**逐元素完全相等**。
    """
    import torch

    if not hasattr(torch, "_int_mm"):
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": "torch._int_mm 不存在（torch {}）".format(torch.__version__),
        }
    out: Dict[str, Any] = {"exists": True}
    # 形状要满足 int_mm 的约束：M/K/N 都要是 8 的倍数，且够大。
    m, k, n = 32, 64, 48
    g = torch.Generator(device="cpu").manual_seed(0)
    a_cpu = torch.randint(-100, 100, (m, k), generator=g, dtype=torch.int8)
    b_cpu = torch.randint(-100, 100, (k, n), generator=g, dtype=torch.int8)
    ref = (a_cpu.to(torch.float64) @ b_cpu.to(torch.float64)).to(torch.int64)

    for dev in ("cpu", "cuda"):
        key = "on_" + dev
        if dev == "cuda" and not torch.cuda.is_available():
            out[key] = {"status": STATUS_UNAVAILABLE, "reason": "没有 CUDA 设备"}
            continue
        try:
            a = a_cpu.to(dev)
            b = b_cpu.to(dev)
            res = torch._int_mm(a, b)
            res64 = res.to("cpu").to(torch.int64)
            exact = bool(torch.equal(res64, ref))
            out[key] = {
                "status": STATUS_OK,
                "callable": True,
                "out_dtype": str(res.dtype).replace("torch.", ""),
                "matches_fp64_reference_exactly": exact,
                "max_abs_diff": int((res64 - ref).abs().max()),
                "verdict": (
                    "可用且结果正确" if exact else "能调用但结果错误：不要用（见 pytorch#107671）"
                ),
            }
        except Exception as exc:
            out[key] = {
                "status": STATUS_UNAVAILABLE,
                "exception": type(exc).__name__,
                "message": str(exc)[:300],
            }
    out["status"] = STATUS_OK
    out["note"] = (
        "即使 _int_mm 可用且正确，V100 上它也**不会更快**：sm70 没有 IMMA 指令，"
        "整数矩阵乘只能落到普通 CUDA core。"
    )
    return out


def probe_weight_int8pack_mm() -> Dict[str, Any]:
    import torch

    exists = hasattr(torch, "_weight_int8pack_mm")
    return {
        "status": STATUS_OK if exists else STATUS_UNAVAILABLE,
        "exists": exists,
        "expected_on_torch_2_1": False,
        "note": (
            "torch._weight_int8pack_mm 在 2.3 才加入，且实现在 CPU/MPS 侧。"
            "目标机 torch 2.1 上预期不存在；存在的话说明 torch 版本与假设不符，"
            "回头去核对 06_QUANT_LOWBIT.md 的版本表。"
        ),
    }


def probe_fake_quant() -> Dict[str, Any]:
    """fake-quant 的 per-tensor / per-channel 前向与**反向**，CPU 与 CUDA 各测一遍。

    这是量化模块里唯一一条在 V100 上完全成立的路径，所以要测得比别的项更细：
    前向要有输出，反向要有梯度，而且梯度必须是 STE 的形状（被 clip 的位置为 0）。
    """
    import torch

    out: Dict[str, Any] = {"status": STATUS_OK}
    for dev in ("cpu", "cuda"):
        key = "on_" + dev
        if dev == "cuda" and not torch.cuda.is_available():
            out[key] = {"status": STATUS_UNAVAILABLE, "reason": "没有 CUDA 设备"}
            continue
        entry: Dict[str, Any] = {}
        try:
            x = torch.randn(16, 64, device=dev, requires_grad=True)
            # per-tensor
            y = torch.fake_quantize_per_tensor_affine(x, 0.01, 0, -127, 127)
            y.sum().backward()
            entry["per_tensor_forward"] = True
            entry["per_tensor_backward"] = x.grad is not None
            entry["per_tensor_grad_has_zeros"] = bool((x.grad == 0).any())
            # per-channel
            x2 = torch.randn(16, 64, device=dev, requires_grad=True)
            scale = (x2.detach().abs().amax(dim=1) / 127.0).to(torch.float32)
            zp = torch.zeros(16, dtype=torch.int32, device=dev)
            y2 = torch.fake_quantize_per_channel_affine(x2, scale, zp, 0, -127, 127)
            g = torch.randn_like(y2)
            y2.backward(g)
            entry["per_channel_forward"] = True
            entry["per_channel_backward"] = x2.grad is not None
            entry["per_channel_grad_equals_passthrough"] = bool(torch.equal(x2.grad, g))
            # 用一个更紧的 scale 逼出 clip，检查 STE 的 0 梯度
            x3 = torch.randn(16, 64, device=dev, requires_grad=True)
            tight = (x3.detach().abs().amax(dim=1) * 0.4 / 127.0).to(torch.float32)
            y3 = torch.fake_quantize_per_channel_affine(x3, tight, zp, 0, -127, 127)
            g3 = torch.ones_like(y3)
            y3.backward(g3)
            entry["clipped_grad_zero_frac"] = float((x3.grad == 0).to(torch.float32).mean())
            entry["status"] = STATUS_OK
        except Exception as exc:
            entry["status"] = STATUS_ERROR
            entry["exception"] = type(exc).__name__
            entry["message"] = str(exc)[:300]
        out[key] = entry
    out["why_it_matters"] = (
        "这几项在 CUDA 上都 ok，就说明 QAT（Q5）在 V100 上可以真跑；"
        "任何一项在 CUDA 上失败，Q5 就要退回 CPU-only，并在证据里写明。"
    )
    return out


def probe_fused_obs_fake_quant() -> Dict[str, Any]:
    """FusedMovingAvgObsFakeQuantize（对应 FusedObsFakeQuant.cu）。"""
    import warnings

    import torch

    out: Dict[str, Any] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torch.ao.quantization import FusedMovingAvgObsFakeQuantize
            from torch.ao.quantization.observer import (
                MovingAveragePerChannelMinMaxObserver,
            )
    except Exception as exc:
        return {
            "status": STATUS_UNAVAILABLE,
            "exception": type(exc).__name__,
            "message": str(exc)[:300],
        }
    for dev in ("cpu", "cuda"):
        key = "on_" + dev
        if dev == "cuda" and not torch.cuda.is_available():
            out[key] = {"status": STATUS_UNAVAILABLE, "reason": "没有 CUDA 设备"}
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = FusedMovingAvgObsFakeQuantize(
                    observer=MovingAveragePerChannelMinMaxObserver,
                    quant_min=-127,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    ch_axis=0,
                    averaging_constant=1.0,
                ).to(dev)
                w = torch.randn(16, 64, device=dev, requires_grad=True)
                y = mod(w)
                y.sum().backward()
            manual_scale = (w.detach().abs().amax(dim=1) / 127.0)
            out[key] = {
                "status": STATUS_OK,
                "forward": True,
                "backward": w.grad is not None,
                "scale_max_abs_diff_vs_amax_over_127": float(
                    (mod.scale.reshape(-1).to(dev) - manual_scale).abs().max()
                ),
                "note": (
                    "scale 与 amax/127 不完全相等属正常：融合 observer 的 qparams "
                    "计算路径与 PerChannelMinMaxObserver 不同。要和自写 STE 对齐，"
                    "用非融合的 FakeQuantize。"
                ),
            }
        except Exception as exc:
            out[key] = {
                "status": STATUS_ERROR,
                "exception": type(exc).__name__,
                "message": str(exc)[:300],
            }
    out["status"] = STATUS_OK
    return out


def probe_convert_fx() -> Dict[str, Any]:
    """FX 图模式量化：convert_fx 的产物能不能 .cuda()。**预期是不能。**

    这一项失败才是符合预期的。它的价值在于把「为什么 V100 上不能部署 INT8」
    从一句结论变成一条现场记录。
    """
    import warnings

    import torch
    import torch.nn as nn

    out: Dict[str, Any] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torch.ao.quantization import get_default_qconfig_mapping
            from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

            model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
            example = (torch.randn(4, 16),)
            backends = []
            try:
                backends = list(torch.backends.quantized.supported_engines)
            except Exception:
                backends = []
            out["supported_engines"] = backends
            qmap = get_default_qconfig_mapping("fbgemm")
            prepared = prepare_fx(model, qmap, example_inputs=example)
            prepared(*example)
            converted = convert_fx(prepared)
            out["convert_fx_ok"] = True
            out["cpu_forward_ok"] = bool(converted(*example).shape == (4, 8))
    except Exception as exc:
        return {
            "status": STATUS_ERROR,
            "convert_fx_ok": False,
            "exception": type(exc).__name__,
            "message": str(exc)[:300],
        }

    if not torch.cuda.is_available():
        out["cuda_move"] = {
            "status": STATUS_UNAVAILABLE,
            "reason": "本机没有 CUDA，无法验证 .cuda() 的失败方式",
        }
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                moved = converted.cuda()
                _ = moved(example[0].cuda())
            out["cuda_move"] = {
                "status": STATUS_OK,
                "moved_and_ran": True,
                "unexpected": True,
                "note": "与审计结论不符：请把这条输出带回来，06_QUANT_LOWBIT.md 要改。",
            }
        except Exception as exc:
            out["cuda_move"] = {
                "status": STATUS_OK,
                "moved_and_ran": False,
                "unexpected": False,
                "exception": type(exc).__name__,
                "message": str(exc)[:300],
                "note": "失败是**预期结果**：INT8 后端只有 CPU 实现。",
            }
    out["status"] = STATUS_OK
    return out


def probe_torchao() -> Dict[str, Any]:
    try:
        import torchao  # type: ignore
    except Exception as exc:
        return {
            "status": STATUS_UNAVAILABLE,
            "exception": type(exc).__name__,
            "message": str(exc)[:200],
            "expected_on_torch_2_1": "unavailable（torchao 最低要求 torch 2.5）",
        }
    return {
        "status": STATUS_OK,
        "version": getattr(torchao, "__version__", "unknown"),
        "note": "torchao 出现在 torch 2.1 环境里不符合审计结论，请核对实际 torch 版本。",
    }


def probe_bnb() -> Dict[str, Any]:
    """bitsandbytes 的三类功能各自可用与否。装不上是**预期结果**。"""
    import torch

    try:
        import bitsandbytes as bnb  # type: ignore
    except Exception as exc:
        return {
            "status": STATUS_UNAVAILABLE,
            "exception": type(exc).__name__,
            "message": str(exc)[:300],
            "note": (
                "内网装 bitsandbytes 有风险且非必需：mm_quant 的所有实验都是纯 PyTorch。"
                "要装的话与 torch 2.1 相容的最高版本是 0.45.5。"
            ),
        }

    out: Dict[str, Any] = {
        "status": STATUS_OK,
        "version": getattr(bnb, "__version__", "unknown"),
    }
    cuda_ok = torch.cuda.is_available()
    cc = None
    if cuda_ok:
        p = torch.cuda.get_device_properties(0)
        cc = (p.major, p.minor)
    out["capability"] = "{}.{}".format(*cc) if cc else None

    # 1) 8-bit 优化器：要求算力 >= 6.0，V100 满足
    try:
        import torch.nn as nn

        lin = nn.Linear(64, 64)
        if cuda_ok:
            lin = lin.cuda()
        opt = bnb.optim.AdamW8bit(lin.parameters(), lr=1e-3)
        lin(torch.randn(8, 64, device="cuda" if cuda_ok else "cpu")).sum().backward()
        opt.step()
        out["adamw8bit"] = {"status": STATUS_OK, "stepped": True, "min_capability": "6.0"}
    except Exception as exc:
        out["adamw8bit"] = {
            "status": STATUS_UNAVAILABLE,
            "exception": type(exc).__name__,
            "message": str(exc)[:300],
            "min_capability": "6.0",
        }

    # 2) NF4 / 4-bit：要求算力 >= 6.0，V100 满足；compute dtype 只能 fp16
    try:
        import torch.nn as nn

        lin4 = bnb.nn.Linear4bit(
            64, 64, bias=False, compute_dtype=torch.float16, quant_type="nf4"
        )
        if cuda_ok:
            lin4 = lin4.cuda()
            y = lin4(torch.randn(8, 64, device="cuda", dtype=torch.float16))
            out["nf4"] = {"status": STATUS_OK, "forward": list(y.shape), "min_capability": "6.0"}
        else:
            out["nf4"] = {
                "status": STATUS_UNAVAILABLE,
                "reason": "4-bit 只有 CUDA kernel，本机无 CUDA",
                "min_capability": "6.0",
            }
    except Exception as exc:
        out["nf4"] = {
            "status": STATUS_UNAVAILABLE,
            "exception": type(exc).__name__,
            "message": str(exc)[:300],
            "min_capability": "6.0",
        }

    # 3) LLM.int8()：要求算力 >= 7.5，**V100（7.0）用不了**
    try:
        lin8 = bnb.nn.Linear8bitLt(64, 64, bias=False, has_fp16_weights=False)
        if cuda_ok:
            lin8 = lin8.cuda()
            y = lin8(torch.randn(8, 64, device="cuda", dtype=torch.float16))
            out["llm_int8"] = {
                "status": STATUS_OK,
                "forward": list(y.shape),
                "min_capability": "7.5",
                "note": "在 V100（7.0）上跑通不符合审计结论，请把输出带回来。",
            }
        else:
            out["llm_int8"] = {
                "status": STATUS_UNAVAILABLE,
                "reason": "只有 CUDA kernel，本机无 CUDA",
                "min_capability": "7.5",
            }
    except Exception as exc:
        out["llm_int8"] = {
            "status": STATUS_UNAVAILABLE,
            "exception": type(exc).__name__,
            "message": str(exc)[:300],
            "min_capability": "7.5",
            "expected_on_v100": "unavailable（V100 算力 7.0 < 7.5）",
        }
    return out


PROBES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "environment": probe_environment,
    "arch": probe_arch,
    "int_mm": probe_int_mm,
    "weight_int8pack_mm": probe_weight_int8pack_mm,
    "fake_quant": probe_fake_quant,
    "fused_obs_fake_quant": probe_fused_obs_fake_quant,
    "convert_fx": probe_convert_fx,
    "torchao": probe_torchao,
    "bnb": probe_bnb,
}


def summarize(results: Dict[str, Any]) -> List[str]:
    """一句话结论列表，直接可以抄进证据字段。"""
    lines: List[str] = []
    env = results.get("environment", {})
    if env.get("cuda_available"):
        lines.append(
            "GPU {}，算力 {}，INT8 Tensor Core: {}".format(
                env.get("device_name"),
                env.get("capability"),
                "有" if env.get("has_int8_tensor_core") else "无（任何 INT8 提速都是假的）",
            )
        )
        lines.append("BF16: {}".format("支持" if env.get("bf16_supported") else "不支持 => 只能 fp16"))
    else:
        lines.append("无 CUDA：本次只覆盖 CPU 路径，GPU 结论全部待目标机。")
    im = results.get("int_mm", {}).get("on_cuda", {})
    if im.get("status") == STATUS_OK:
        lines.append(
            "torch._int_mm on CUDA: 可调用，结果{}".format(
                "正确" if im.get("matches_fp64_reference_exactly") else "**错误**，不要用"
            )
        )
    else:
        lines.append("torch._int_mm on CUDA: {}".format(im.get("reason") or im.get("status")))
    fq = results.get("fake_quant", {}).get("on_cuda", {})
    lines.append(
        "fake-quant on CUDA: {} => QAT(Q5) {}".format(
            fq.get("status"),
            "可在 V100 上真跑" if fq.get("status") == STATUS_OK else "退回 CPU-only",
        )
    )
    cm = results.get("convert_fx", {}).get("cuda_move", {})
    if cm:
        lines.append(
            "convert_fx 产物 .cuda(): {}".format(
                "失败（预期）" if cm.get("moved_and_ran") is False else cm.get("status")
            )
        )
    bnb = results.get("bnb", {})
    if bnb.get("status") == STATUS_UNAVAILABLE:
        lines.append("bitsandbytes: 未安装（预期；所有实验不依赖它）")
    else:
        lines.append(
            "bitsandbytes {}: AdamW8bit={} NF4={} LLM.int8()={}".format(
                bnb.get("version"),
                bnb.get("adamw8bit", {}).get("status"),
                bnb.get("nf4", {}).get("status"),
                bnb.get("llm_int8", {}).get("status"),
            )
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="探测 INT8 / fake-quant / bitsandbytes 在本机的可用性（信息收集器，永远退出 0）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tag",
        default="quant_probe",
        help="run 目录标签，结果写到 <MM_RUNS_ROOT>/<tag>/logs/int8_caps.json",
    )
    p.add_argument(
        "--out",
        default="",
        help="显式输出路径；'-' 表示只打到 stdout，不落盘。留空则用 --tag 的默认位置",
    )
    p.add_argument(
        "--only",
        default="",
        help="只跑这些探针，逗号分隔。可选：" + ",".join(CHECKS),
    )
    p.add_argument(
        "--skip",
        default="",
        help="跳过这些探针，逗号分隔",
    )
    p.add_argument("--quiet", action="store_true", help="不打印摘要，只写文件")
    return p


def main(argv: List[str]) -> int:
    use_utf8()
    args = build_parser().parse_args(argv)

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    for name in only + skip:
        if name not in CHECKS:
            print("未知探针名 {!r}；可选：{}".format(name, ", ".join(CHECKS)), file=sys.stderr)
            return 2

    selected = [c for c in CHECKS if (not only or c in only) and c not in skip]

    results: Dict[str, Any] = {}
    for name in CHECKS:
        if name not in selected:
            results[name] = {"status": STATUS_SKIPPED}
            continue
        results[name] = _guard(PROBES[name])

    payload = {
        "schema": "mm_quant.int8_caps.v1",
        "checks_run": selected,
        "results": results,
        "summary": summarize(results),
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out == "-":
        print(text)
    else:
        if args.out:
            out_path = Path(args.out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = paths.run_dir(args.tag) / "logs" / "int8_caps.json"
        try:
            out_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            print("写不进 {}：{}".format(out_path, exc), file=sys.stderr)
            print("下一步：export MM_RUNS_ROOT=<一个可写目录>，或用 --out - 只打屏。",
                  file=sys.stderr)
            return 2
        if not args.quiet:
            print("结果已写入 {}".format(out_path))

    if not args.quiet:
        print("")
        print("一句话结论：")
        for line in payload["summary"]:
            print("  - " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
