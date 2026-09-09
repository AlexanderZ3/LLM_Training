"""路径合同：本仓库（Windows，无 GPU）与 V100 服务器（内网）之间唯一的路径来源。

设计约束
--------
1. 这个文件里**没有任何字面绝对路径**。所有位置来自四个环境变量，缺省时回退到
   周包内的同名目录。你在 V100 上设一次环境变量，全部脚本同时对齐。
2. 找不到东西时抛出的异常必须**说清楚下一步做什么**，而不是只报一个 FileNotFoundError。
   代码在这台机器上写、在另一台机器上第一次运行，报错信息就是唯一的排错入口。
3. 这个模块不 import torch。它必须在任何环境里都能 import，包括只装了标准库的
   环境——否则 Day 0 的路径自检会因为 torch 装不上而无法运行，那是本末倒置。

环境变量
--------
MM_DATA_ROOT     只读数据根目录，见 ../../datasets/README.md
MM_WEIGHTS_ROOT  只读权重根目录，见 ../../weights/README.md
MM_RUNS_ROOT     可写运行产物根目录，见 ../../runs/README.md
MINIMIND_ROOT    MiniMind 仓库的克隆位置（git clone 得到的那个目录）
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "PackageRoot",
    "package_root",
    "data_root",
    "weights_root",
    "runs_root",
    "minimind_root",
    "manifest_path",
    "load_manifest",
    "dataset_dir",
    "dataset_file",
    "weights_dir",
    "run_dir",
    "require_free_space",
    "describe",
    "PathContractError",
]

# 环境变量名集中在这里，别处不许再写字符串字面量。
ENV_DATA = "MM_DATA_ROOT"
ENV_WEIGHTS = "MM_WEIGHTS_ROOT"
ENV_RUNS = "MM_RUNS_ROOT"
ENV_MINIMIND = "MINIMIND_ROOT"

# run_dir() 在每个 run 目录下固定建这几个子目录，脚本不再自己拼。
RUN_SUBDIRS = ("ckpt", "logs", "profiles")


class PathContractError(RuntimeError):
    """路径合同被破坏。异常消息里必须包含用户下一步该做什么。"""


class PackageRoot:
    """周包根目录 = 这个文件往上四层（src/mm_v100/paths.py -> mm_v100 -> src -> lab -> 周包）。"""

    @staticmethod
    def resolve() -> Path:
        return Path(__file__).resolve().parents[3]


def package_root() -> Path:
    return PackageRoot.resolve()


def _root_from_env(var: str, fallback_subdir: str) -> Path:
    """读环境变量；没设就回退到周包内的同名目录。返回值不保证存在。"""
    raw = os.environ.get(var, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (package_root() / fallback_subdir).resolve()


def data_root() -> Path:
    return _root_from_env(ENV_DATA, "datasets")


def weights_root() -> Path:
    return _root_from_env(ENV_WEIGHTS, "weights")


def runs_root() -> Path:
    return _root_from_env(ENV_RUNS, "runs")


def minimind_root() -> Path:
    """MiniMind 仓库位置。这一个没有合理的包内回退，未设就是错误。"""
    raw = os.environ.get(ENV_MINIMIND, "").strip()
    if not raw:
        raise PathContractError(
            f"环境变量 {ENV_MINIMIND} 没设。\n"
            f"下一步：在 V100 上 git clone MiniMind，然后\n"
            f"  export {ENV_MINIMIND}=/你克隆到的路径\n"
            f"本周锁定的 commit 见 lab/README.md 第 1 节。"
        )
    p = Path(raw).expanduser().resolve()
    if not (p / "model").is_dir():
        raise PathContractError(
            f"{ENV_MINIMIND}={p} 下面没有 model/ 目录，这不像 MiniMind 仓库根目录。\n"
            f"下一步：确认你指的是 git clone 出来的那个目录本身，不是它的父目录。"
        )
    return p


def manifest_path() -> Path:
    """数据清单。先看 MM_DATA_ROOT 下有没有（拷贝时可以一起带过去），再回退到包内。"""
    candidate = data_root() / "DATA_MANIFEST.json"
    if candidate.is_file():
        return candidate
    return package_root() / "datasets" / "DATA_MANIFEST.json"


def load_manifest() -> Dict[str, Any]:
    p = manifest_path()
    if not p.is_file():
        raise PathContractError(
            f"找不到数据清单 {p}。\n"
            f"下一步：确认周包完整，或把 datasets/DATA_MANIFEST.json 拷到 {data_root()} 下。"
        )
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dataset_dir(group: str = "minimind_dataset") -> Path:
    """某个数据组的目录。不存在时给出可执行的下一步。"""
    d = data_root() / group
    if not d.is_dir():
        raise PathContractError(
            f"数据目录不存在：{d}\n"
            f"下一步（二选一）：\n"
            f"  1) 把 {group}/ 拷到 {data_root()} 下；\n"
            f"  2) 或者 export {ENV_DATA}=<你实际放数据的父目录>\n"
            f"目录长什么样见 datasets/README.md 第 2 节；拷完跑 "
            f"python lab/scripts/check_data_layout.py 确认。"
        )
    return d


def dataset_file(name: str, group: str = "minimind_dataset") -> Path:
    """一个具体数据文件。会顺带核对清单里登记的字节数。"""
    p = dataset_dir(group) / name
    if not p.is_file():
        known = _manifest_names(group)
        hint = ""
        if known and name not in known:
            hint = (
                f"\n注意：'{name}' 不在清单里。清单登记的文件是：\n  "
                + "\n  ".join(known)
            )
        raise PathContractError(
            f"数据文件不存在：{p}{hint}\n"
            f"下一步：python lab/scripts/check_data_layout.py 会列出缺哪些。"
        )
    expected = _manifest_bytes(group, name)
    if expected is not None:
        actual = p.stat().st_size
        if actual != expected:
            raise PathContractError(
                f"字节数对不上：{p}\n"
                f"  期望 {expected:,} B，实际 {actual:,} B（差 {actual - expected:+,} B）\n"
                f"下一步：多半是传输被截断，或者用了文本模式传输把换行改了。"
                f"重新用二进制方式拷这个文件，再跑 check_data_layout.py。"
            )
    return p


def _manifest_group(group: str) -> Optional[Dict[str, Any]]:
    try:
        return load_manifest()["groups"].get(group)
    except (PathContractError, KeyError, json.JSONDecodeError):
        return None


def _manifest_names(group: str) -> List[str]:
    g = _manifest_group(group)
    if not g:
        return []
    return [f["path"] for f in g["files"]]


def _manifest_bytes(group: str, name: str) -> Optional[int]:
    g = _manifest_group(group)
    if not g:
        return None
    for f in g["files"]:
        if f["path"] == name:
            return int(f["bytes"])
    return None


def weights_dir(name: str, must_exist: bool = True) -> Path:
    """权重子目录。name 例如 'minimind_tokenizer'、'qat_base'。"""
    d = weights_root() / name
    if must_exist and not d.is_dir():
        raise PathContractError(
            f"权重目录不存在：{d}\n"
            f"下一步：内网下不动权重，要在外网机器上下好再拷进来。"
            f"具体怎么拷见 weights/README.md。\n"
            f"或者 export {ENV_WEIGHTS}=<你实际放权重的目录>"
        )
    return d


def run_dir(tag: str, create: bool = True) -> Path:
    """一次运行的产物目录，固定带 ckpt/ logs/ profiles/ 三个子目录。

    tag 用 day0_probe、day1_pretrain 这种；不要带日期或随机数，
    重跑同一天就覆盖同一个目录，这样证据字段里的路径是稳定的。
    """
    d = runs_root() / tag
    if create:
        for sub in RUN_SUBDIRS:
            (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def require_free_space(gb: float, where: Optional[Path] = None) -> float:
    """磁盘不够就在训练开始前退出，而不是训到一半塞满盘。返回剩余 GiB。"""
    target = Path(where) if where is not None else runs_root()
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_bytes = shutil.disk_usage(str(probe)).free
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < gb:
        raise PathContractError(
            f"磁盘空间不够：{probe} 只剩 {free_gb:.1f} GiB，这一步需要 {gb:.1f} GiB。\n"
            f"下一步：清理 {runs_root()} 下旧的 run 目录，或把 {ENV_RUNS} 指向大容量分区。"
        )
    return free_gb


def describe() -> str:
    """把当前解析出的四个根目录打成一张表。Day 0 第一条命令就是打印这个。

    只打印路径和存在与否，不打印任何数据内容——所以这个输出可以直接贴出来。
    """
    rows = [
        (ENV_DATA, data_root(), "只读数据"),
        (ENV_WEIGHTS, weights_root(), "只读权重"),
        (ENV_RUNS, runs_root(), "可写产物"),
    ]
    lines = ["路径合同当前解析结果：", ""]
    for var, path, what in rows:
        src = "环境变量" if os.environ.get(var, "").strip() else "包内默认"
        exists = "存在" if path.exists() else "不存在"
        lines.append(f"  {var:<16} {src:<8} {exists:<6} {what:<8} {path}")
    raw_mm = os.environ.get(ENV_MINIMIND, "").strip()
    if raw_mm:
        p = Path(raw_mm).expanduser().resolve()
        exists = "存在" if p.exists() else "不存在"
        lines.append(f"  {ENV_MINIMIND:<16} {'环境变量':<8} {exists:<6} {'仓库':<8} {p}")
    else:
        lines.append(f"  {ENV_MINIMIND:<16} {'未设置':<8} {'—':<6} {'仓库':<8} —")
    lines.append("")
    lines.append(f"  清单文件           {manifest_path()}")
    return "\n".join(lines)
