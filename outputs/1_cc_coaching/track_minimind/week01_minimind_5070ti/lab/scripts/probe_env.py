"""环境探针 → probe.json（Day 0）。无 GPU 时 GPU 字段为 null，退出码仍为 0。

用法：
  python scripts/probe_env.py [--out probe.json] [--minimind-root DIR] [--skip-hash]
检查项：python/torch/cuda 版本、GPU 名、compute capability、torch.cuda.get_arch_list()、is_bf16_supported()、
SDPA 后端开关、一次 CUDA 矩阵乘（捕获 "no kernel image"/sm_120 不支持）、MiniMind commit 是否等于锁定值、
4 个数据文件是否存在、大小是否与周卡一致、sha256（--skip-hash 可跳过；GB 级文件约需 1–3 分钟）。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from mm_probe.minimind_env import DATA_FILES, PINNED_COMMIT, find_minimind_root, git_head, sha256_file  # noqa: E402


def _version(mod_name: str):
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def probe_torch() -> dict:
    out = {"torch": None, "torch_cuda_version": None, "cudnn": None, "cuda_available": False, "gpu_name": None,
           "compute_capability": None, "arch_list": None, "bf16_supported": None, "sdpa": None,
           "cuda_matmul_ok": None, "cuda_matmul_error": None, "cpu_sdpa_available": None, "total_mem_mb": None}
    try:
        import torch
    except ImportError:
        return out
    out["torch"] = torch.__version__
    out["torch_cuda_version"] = torch.version.cuda
    out["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    out["cpu_sdpa_available"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    try:
        out["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception as e:  # noqa: BLE001
        out["arch_list"] = f"error: {e}"
    bc = torch.backends.cuda
    sdpa = {}
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled", "is_flash_attention_available"):
        fn = getattr(bc, name, None)
        try:
            sdpa[name] = bool(fn()) if callable(fn) else None
        except Exception as e:  # noqa: BLE001
            sdpa[name] = f"error: {e}"
    out["sdpa"] = sdpa
    out["cuda_available"] = bool(torch.cuda.is_available())
    if not out["cuda_available"]:
        return out
    try:
        out["gpu_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        out["compute_capability"] = f"{cap[0]}.{cap[1]}"
        out["total_mem_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
        out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:  # noqa: BLE001
        out["cuda_matmul_error"] = f"device query failed: {e}"
    try:
        a = torch.randn(64, 64, device="cuda")
        b = (a @ a).float().sum().item()
        out["cuda_matmul_ok"] = bool(b == b)
    except Exception as e:  # noqa: BLE001
        out["cuda_matmul_ok"] = False
        out["cuda_matmul_error"] = str(e)
    return out


def probe_minimind(root_arg, skip_hash: bool) -> dict:
    out = {"minimind_root": None, "minimind_commit": None, "pinned_commit": PINNED_COMMIT, "commit_matches": None,
           "tokenizer_loads": None, "data_files": {}}
    root = find_minimind_root(root_arg, require=False)
    if root is None:
        return out
    out["minimind_root"] = str(root)
    head = git_head(root)
    out["minimind_commit"] = head
    out["commit_matches"] = (head == PINNED_COMMIT) if head else None
    try:
        from mm_probe.minimind_env import load_tokenizer

        tok = load_tokenizer(root)
        out["tokenizer_loads"] = (len(tok) == 6400 and tok.bos_token_id == 1 and tok.eos_token_id == 2 and tok.pad_token_id == 0)
    except Exception as e:  # noqa: BLE001
        out["tokenizer_loads"] = f"error: {e}"
    for name, expected in DATA_FILES.items():
        p = root / "dataset" / name
        rec = {"exists": p.exists(), "size": None, "size_matches_card": None, "sha256": None}
        if p.exists():
            rec["size"] = p.stat().st_size
            rec["size_matches_card"] = rec["size"] == expected
            if not skip_hash:
                rec["sha256"] = sha256_file(p)
        out["data_files"][name] = rec
    return out


def build_probe(args) -> dict:
    t = probe_torch()
    m = probe_minimind(args.minimind_root, args.skip_hash)
    warnings = []
    cap = t.get("compute_capability")
    arch = t.get("arch_list") or []
    if cap == "12.0" and isinstance(arch, list) and "sm_120" not in arch:
        warnings.append("GPU 是 sm_120 (RTX 50 系) 但 torch 编译的 arch_list 不含 sm_120：安装 torch>=2.7 的 cu128 wheel")
    if t["cuda_matmul_ok"] is False:
        warnings.append(f"CUDA 矩阵乘失败：{t['cuda_matmul_error']}")
    if t["cuda_available"] and t["bf16_supported"] is False:
        warnings.append("GPU 不支持 bf16：训练 dtype 改用 float16 + GradScaler")
    if m["minimind_root"] is None:
        warnings.append("MINIMIND_ROOT 未设置：MiniMind 相关检查与依赖 tokenizer 的测试将跳过")
    elif m["commit_matches"] is False:
        warnings.append(f"MiniMind commit {m['minimind_commit']} ≠ 锁定 {PINNED_COMMIT}：git -C $MINIMIND_ROOT checkout {PINNED_COMMIT}")
    for name, rec in m["data_files"].items():
        if not rec["exists"]:
            warnings.append(f"数据缺失：{name}（运行 scripts/setup_minimind.*）")
        elif rec["size_matches_card"] is False:
            warnings.append(f"数据大小与周卡记录不一致：{name} size={rec['size']}")
    return {
        "probe_version": 1,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        **t,
        "transformers": _version("transformers"),
        "datasets": _version("datasets"),
        "huggingface_hub": _version("huggingface_hub"),
        "matplotlib": _version("matplotlib"),
        **m,
        "warnings": warnings,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="环境探针 → probe.json")
    p.add_argument("--out", default=str(LAB / "runs" / "probe.json"))
    p.add_argument("--minimind-root", default=None)
    p.add_argument("--skip-hash", action="store_true", help="跳过 GB 级文件的 sha256")
    args = p.parse_args(argv)
    result = build_probe(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
